#!/usr/bin/env python
# multi_label_inference.py

import json
import tqdm
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer, models
from src import args

# ----------------------------- #
# 1) Multi-Label Model
# ----------------------------- #
class MultiLabelClassificationModel(nn.Module):
    """
    Outputs raw logits of shape (batch_size, num_classes).
    We'll apply sigmoid at inference time.
    """
    def __init__(self, input_size, num_classes):
        super(MultiLabelClassificationModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # raw logits
        return x


# ----------------------------- #
# 2) Utility Functions
# ----------------------------- #

def get_embedding(questions, model_path):
    """
    Use a SentenceTransformer to get embeddings for a list of questions.
    """
    default_model = models.Transformer('microsoft/codebert-base')
    pooling_model = models.Pooling(default_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[default_model, pooling_model])
    model = SentenceTransformer(model_path)
    embeddings = []
    print('Generating embeddings...')
    for question in tqdm.tqdm(questions, ncols=75):
        embedding = model.encode(question)
        embeddings.append(embedding)
    return embeddings


def set_seed(seed):
    """
    For reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Example dictionary for mapping indices to technique strings.
technique_dict = {
    0: 'Zeroshot',
    1: 'Zeroshot_CoT',
    2: 'Fewshot',
    3: 'Fewshot_CoT',
    4: 'Persona',
    5: 'Self-planning',
    6: 'Self-refine',
    7: 'Progressive-Hint',
    8: 'Self-debug'
}


def evaluate_top3_accuracy_and_tokens(
    data_list,
    model,
    model_path,
    get_embedding_fn,
    technique_dict,
    top_k=9
):
    """
    1) For each sample, pick the top_k techniques by predicted probability.
    2) Mark the sample as 'correct' if any of the top_k is in the successful set (exec_acc >= 0).
    3) Sum tokens for those top_k predictions for each sample, then average.
    
    Returns:
      overall_acc (float): fraction of samples with at least one correct technique in top_k
      per_tech_acc (dict): for each technique, fraction of samples where it was predicted AND was successful
      avg_pred_tokens (float): average total tokens used by top_k predicted techniques
      per_tech_avg_tokens (dict): if we sum tokens for that technique across all samples, then / total
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    # 1) Gather embeddings
    questions = [item['prompt'] for item in data_list]
    embeddings = get_embedding_fn(questions, model_path)
    embeddings = torch.tensor(embeddings, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(embeddings)          # shape: (N, num_classes)
        probs = torch.sigmoid(logits)       # convert to probabilities (0..1)

        # We want the top_k for each row
        # values: shape (N, top_k), indices: shape (N, top_k)
        topk_values, topk_indices = torch.topk(probs, k=top_k, dim=1)  

    total_samples = len(data_list)

    # 2) Track how many samples are correct
    correct_samples = 0

    # 3) For per-technique correctness and token usage
    per_tech_correct = {tech: 0 for tech in technique_dict.values()}
    per_tech_tokens  = {tech: 0 for tech in technique_dict.values()}

    total_pred_tokens = 0.0

    for i, per_data in enumerate(data_list):
        # Indices of top_k predictions
        pred_indices = topk_indices[i].tolist()
        pred_strategies = [technique_dict[idx] for idx in pred_indices]
        # pred_strategies = random.sample(list(technique_dict.values()), k=top_k)
        sample_pred_tokens = 0
        sample_pred_token_record = []
        token_record = per_data['token_record']  # {strategy_name: token_count}

        # Build "successful" set from ranked_techniques
        # i.e., techniques with exec_acc >= 0
        successful_techniques = set()
        for (exec_strategy, exec_acc) in per_data['ranked_techniques']:
            if exec_acc >= 0:
                successful_techniques.add(exec_strategy)
                if exec_strategy in pred_strategies:
                    sample_pred_token_record.append(token_record[exec_strategy])
                    # correct_samples += 1


        print(pred_strategies)
        # print(successful_techniques)
        
        # Check correctness: if any of the top_k strategies is successful
        if len(set(pred_strategies).intersection(successful_techniques)) > 0:
            correct_samples += 1

        # Count per-technique correctness
        for tech in successful_techniques:
            per_tech_correct[tech] += 1

        # 4) Sum tokens for these top_k predictions
        
        for tech in technique_dict.values():
            # Add the token cost for that technique if it exists
            per_tech_tokens[tech] += token_record[tech]
        
        if sample_pred_token_record == []:
            for tech in pred_strategies:
                sample_pred_token_record.append(token_record[tech])

        sample_pred_tokens += min(sample_pred_token_record)
        total_pred_tokens += sample_pred_tokens

    # 5) Final metrics
    overall_acc = correct_samples / total_samples

    # For each technique, we define "accuracy" as fraction of total samples
    # for which that technique was in top_k predictions AND was successful
    per_tech_acc = {}
    for tech in technique_dict.values():
        per_tech_acc[tech] = per_tech_correct[tech] / total_samples

    avg_pred_tokens = total_pred_tokens / total_samples

    # This is the average tokens if we used that technique for all samples
    per_tech_avg_tokens = {}
    for tech in technique_dict.values():
        per_tech_avg_tokens[tech] = per_tech_tokens[tech] / total_samples

    return overall_acc, per_tech_acc, avg_pred_tokens, per_tech_avg_tokens



# ----------------------------- #
# 3) Example "main" usage
# ----------------------------- #
if __name__ == "__main__":
    arguments = args.get_args()
    # Example usage (you can replace paths with your own)
    set_seed(42)

    # Suppose your multi-label model is saved at this path:
    model_path_embedding = "PET_model_result/code_complex_contrastive_model"
    model_path_classification = "PET_model_result/classification_model/multilabel_code_complex_classification_model_parameters.pth"
    top_k = arguments.top_k

    # Load test data from JSONL
    test_file_path = "PET_model_dataset/code_complex_classification_dataset_test.jsonl"
    with open(test_file_path, "r") as f:
        test_data = [json.loads(line) for line in f]

    # Create multi-label model and load parameters
    # *IMPORTANT*: ensure 'input_size' and 'num_classes' match what you used in training
    embeddings_example = get_embedding([test_data[0]['prompt']], model_path_embedding)
    input_size = len(embeddings_example[0])
    num_classes = len(technique_dict)  # 9

    model = MultiLabelClassificationModel(input_size=input_size, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path_classification, map_location="cpu"))

    # Evaluate "actual accuracy" in multi-label sense
    overall_acc, per_tech_acc, avg_pred_tokens, per_tech_avg_tokens = evaluate_top3_accuracy_and_tokens(
        data_list=test_data,
        model=model,
        model_path=model_path_embedding, 
        get_embedding_fn=get_embedding,     # your function
        technique_dict=technique_dict,
        top_k=top_k
    )

    print(f"Top-{top_k} Accuracy: {overall_acc*100:.2f}%")
    for tech, acc in per_tech_acc.items():
        print(f"{tech}: {acc*100:.2f}%")

    print(f"\nAvg tokens used by top-{top_k} predictions: {avg_pred_tokens:.2f}")
    for tech, avg_tok in per_tech_avg_tokens.items():
        print(f"{tech}: {avg_tok:.2f}")
