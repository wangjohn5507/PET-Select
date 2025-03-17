import json
import random
import tqdm
from src import args, utils
import torch
import torch.nn as nn
import numpy as np
import os


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

label_dict = {'Zeroshot': 0, 'Zeroshot_CoT': 1, 'Fewshot': 2, 'Fewshot_CoT': 3, 'Persona': 4, 'Self-planning': 5, 'Self-refine': 6, 'Progressive-Hint': 7, 'Self-debug': 8}

class CodeComplexityClassifier(nn.Module):
    def __init__(self):
        super(CodeComplexityClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = self.sigmoid(x) 
        return x  # Raw logits
    
def get_complexity_data(data, complexity_threshold, model):
    # complexity_record = 0
    easy_complexity_data = []
    hard_complexity_data = []
    for per_data in tqdm.tqdm(data, ncols=75):
        complexity = per_data['weighted_complexity']
        # complexity_record += complexity
        if complexity < complexity_threshold:
            easy_complexity_data.append(per_data)
        else:
            hard_complexity_data.append(per_data)
        # features = []
        # normalized_physical_loc = per_data['normalized_physical_loc']
        # normalized_cyclomatic_complexity = per_data['normalized_cyclomatic_complexity']
        # normalized_halstead_complexity = per_data['normalized_halstead_complexity']
        # normalized_mi = per_data['normalized_mi']
        # normalized_cognitive_complexity = per_data['normalized_cognitive_complexity']
        # features.append(normalized_physical_loc)
        # features.append(normalized_cyclomatic_complexity)
        # features.append(normalized_halstead_complexity)
        # features.append(normalized_mi)
        # features.append(normalized_cognitive_complexity)

        # input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        # with torch.no_grad():
        #     logits = model(input_tensor)
        #     # probability = torch.sigmoid(logits)
        #     prediction = (logits > 0.5).float()

        # if int(prediction.item()) == 0:
        #     easy_complexity_data.append(per_data)
        # else:
        #     hard_complexity_data.append(per_data)

    print(f'The size of easy data is {len(easy_complexity_data)}')
    print(f'The size of hard data is {len(hard_complexity_data)}')
    return easy_complexity_data, hard_complexity_data

def get_complexity_classification_data(train_easy_data, train_hard_data):
    train_complexity_classification_data = []
    easy_label = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    hard_label = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    for per_data in train_easy_data:
        label = label_dict[per_data['ranked_techniques'][0][0]]
        if label in easy_label:
            train_complexity_classification_data.append(per_data)
    for per_data in train_hard_data:
        label = label_dict[per_data['ranked_techniques'][0][0]]
        if label in hard_label:
            train_complexity_classification_data.append(per_data)
    return train_complexity_classification_data

def find_samples(pos_data, neg_data, idx):
    pos_samples = []
    neg_samples = []
    for pos_idx, pos_per_data in enumerate(pos_data):
        if idx == pos_idx:
            continue
        else:
            pos_samples.append(pos_per_data)
    for neg_idx, neg_per_data in enumerate(neg_data):
        neg_samples.append(neg_per_data)

    random.shuffle(pos_samples)
    random.shuffle(neg_samples)

    return pos_samples, neg_samples

def get_contrastive_data(easy_data, hard_data, n):
    contrastive_data = []
    for idx, per_data in enumerate(tqdm.tqdm(easy_data, ncols=75)):
        pos_samples, neg_samples = find_samples(easy_data, hard_data, idx)
       
        for i in range(min(n, len(pos_samples), len(neg_samples))):
            contrastive_data.append({
                'anchor': per_data['prompt'],
                'positive': pos_samples[i]['prompt'],
                'negative': neg_samples[i]['prompt']
            })
    # for idx, anchor_data in enumerate(hard_data):
    #     pos_samples, neg_samples = find_samples(hard_data, easy_data, idx)
    #     for i in range(min(n, len(pos_samples), len(neg_samples))):
    #         contrastive_data.append({
    #             'anchor': anchor_data['prompt'],
    #             'positive': pos_samples[i]['prompt'],
    #             'negative': neg_samples[i]['prompt']
    #         })
    return contrastive_data


def write_complexity_data(train_data, test_data, complexity_threshold):
    model_path = 'PET_model_result/complexity_model/complexity_model.pth'
    model = CodeComplexityClassifier()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    train_easy_complexity_data, train_hard_complexity_data = get_complexity_data(train_data, complexity_threshold, model)
    test_easy_complexity_data, test_hard_complexity_data = get_complexity_data(test_data, complexity_threshold, model)
    
    train_classification_data = get_complexity_classification_data(train_easy_complexity_data, train_hard_complexity_data)

    train_contrastive_data = get_contrastive_data(train_easy_complexity_data, train_hard_complexity_data, 3)
    test_contrastive_data = get_contrastive_data(test_easy_complexity_data, test_hard_complexity_data, 3)
            
    train_contrastive_file_path = 'PET_model_dataset/code_complex_contrastive_dataset_train.jsonl'
    test_contrastive_file_path = 'PET_model_dataset/code_complex_contrastive_dataset_test.jsonl'
    train_classification_file_path = 'PET_model_dataset/code_complex_classification_dataset_train.jsonl'
    test_classification_file_path = 'PET_model_dataset/code_complex_classification_dataset_test.jsonl'


    with open(train_contrastive_file_path, 'w') as file:
        for item in train_contrastive_data:
            file.write(json.dumps(item) + '\n')

    with open(test_contrastive_file_path, 'w') as file:
        for item in test_contrastive_data:
            file.write(json.dumps(item) + '\n')

    with open(train_classification_file_path, 'w') as file:
        for item in train_classification_data:
            file.write(json.dumps(item) + '\n')
    
    with open(test_classification_file_path, 'w') as file:
        for item in test_data:
            file.write(json.dumps(item) + '\n')


if __name__ == '__main__':
    set_seed(42)
    arguments = args.get_args()
    train_data = list(map(json.loads, open(f'5fold_dataset/{arguments.dataset}_{arguments.model}_train_{arguments.fold}.jsonl')))
    test_data = list(map(json.loads, open(f'5fold_dataset/{arguments.dataset}_{arguments.model}_test_{arguments.fold}.jsonl')))
    write_complexity_data(train_data, test_data, arguments.complexity)

    
