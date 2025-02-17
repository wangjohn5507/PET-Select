import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd
import random
import tqdm
import json

from sentence_transformers import SentenceTransformer
def get_embedding(questions, model_path):
    model = SentenceTransformer(model_path)
    embeddings = []
    print('Generating embeddings...')
    for question in tqdm.tqdm(questions, ncols=75):
        embedding = model.encode(question)
        embeddings.append(embedding)
    return embeddings

class ClassificationModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ClassificationModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x

technique_dict = {0: 'Zeroshot', 1: 'Zeroshot_CoT', 2: 'Fewshot', 3: 'Fewshot_CoT', 4: 'Persona', 5: 'Self-planning', 6: 'Self-refine', 7: 'Progressive-Hint', 8: 'Self-debug'}

def generate_data_list(data, test_data):
    data_list = []
    test_data_list = []
    for per_test in test_data:
        test_data_list.append(per_test['prompt'])
    for per_data in data:
        if per_data['prompt'] in test_data_list:
            data_list.append(per_data)
    return data_list

def calculate_actual_acc(data_list, model, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    # 获取所有问题的嵌入
    questions = [per_data['prompt'] for per_data in data_list]
    embeddings = get_embedding(questions, model_path)

    # 将嵌入转换为张量并移动到设备
    if isinstance(embeddings, list):
        embeddings = torch.tensor(embeddings).to(device)

    # 预测所有问题的标签
    with torch.no_grad():
        outputs = model(embeddings)
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()

    # 初始化计数器
    correct = 0
    zero_shot_correct = 0
    zero_cot_correct = 0
    few_shot_correct = 0
    few_cot_correct = 0
    self_correct = 0
    reflection_correct = 0
    self_plan_correct = 0
    progressive_hint_correct = 0
    persona_correct = 0
    total = len(data_list)

    # 计算准确率
    for idx, per_data in enumerate(data_list):
        y_pred = predictions[idx]
        # y_pred = random.choice([0, 4])
        # y_pred = random.randint(0, 8)
        print(y_pred)
        strategy = technique_dict[y_pred]
        ranked_techniques = per_data['ranked_techniques']

        # print(y_pred, per_data['label'])

        for ranked_technique in ranked_techniques:
            exec_strategy = ranked_technique[0]
            exec_acc = ranked_technique[1]

            if exec_strategy == strategy and exec_acc >= 0:
                correct += 1
            if exec_strategy == 'Zeroshot' and exec_acc >= 0:
                zero_shot_correct += 1
            if exec_strategy == 'Zeroshot_CoT' and exec_acc >= 0:
                zero_cot_correct += 1
            if exec_strategy == 'Fewshot' and exec_acc >= 0:
                few_shot_correct += 1
            if exec_strategy == 'Fewshot_CoT' and exec_acc >= 0:
                few_cot_correct += 1
            if exec_strategy == 'Persona' and exec_acc >= 0:
                persona_correct += 1
            if exec_strategy == 'Self-planning' and exec_acc >= 0:
                self_plan_correct += 1
            if exec_strategy == 'Self-refine' and exec_acc >= 0:
                reflection_correct += 1
            if exec_strategy == 'Progressive-Hint' and exec_acc >= 0:
                progressive_hint_correct += 1
            if exec_strategy == 'Self-debug' and exec_acc >= 0:
                self_correct += 1
            
            
            
            

    actual_acc = correct / total
    zero_shot_acc = zero_shot_correct / total
    zero_cot_acc = zero_cot_correct / total
    few_shot_acc = few_shot_correct / total
    few_cot_acc = few_cot_correct / total
    self_acc = self_correct / total
    reflection_acc = reflection_correct / total
    self_plan_acc = self_plan_correct / total
    progressive_hint_acc = progressive_hint_correct / total
    persona_acc = persona_correct / total

    return actual_acc, zero_shot_acc, zero_cot_acc, few_shot_acc, few_cot_acc, self_acc, reflection_acc, self_plan_acc, progressive_hint_acc, persona_acc, zero_shot_correct

def calculate_token_saved(data_list, model, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    # 获取所有问题的嵌入
    questions = [per_data['prompt'] for per_data in data_list]
    embeddings = get_embedding(questions, model_path)

    # 将嵌入转换为张量并移动到设备
    if isinstance(embeddings, list):
        embeddings = torch.tensor(embeddings).to(device)

    # 预测所有问题的标签
    with torch.no_grad():
        outputs = model(embeddings)
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()

    # 初始化计数器
    pred_token = 0
    zero_shot_token = 0
    zero_cot_token = 0
    few_shot_token = 0
    few_cot_token = 0
    self_token = 0
    reflection_token = 0
    self_plan_token = 0
    progressive_hint_token = 0
    persona_token = 0
    total = len(data_list)

    # 计算各策略的 token 数量
    for idx, per_data in enumerate(data_list):
        y_pred = predictions[idx]
        # y_pred = random.choice([0, 4])
        # y_pred = random.randint(0, 8)
        print(y_pred)
        strategy = technique_dict[y_pred]
        token_record = per_data['token_record']

        for key, value in token_record.items():
            exec_strategy = key
            token = value

            if exec_strategy == strategy:
                pred_token += token
            if exec_strategy == 'Zeroshot':
                zero_shot_token += token
            if exec_strategy == 'Zeroshot_CoT':
                zero_cot_token += token
            if exec_strategy == 'Fewshot':
                few_shot_token += token
            if exec_strategy == 'Fewshot_CoT':
                few_cot_token += token
            if exec_strategy == 'Self-debug':
                self_token += token
            if exec_strategy == 'Self-refine':
                reflection_token += token
            if exec_strategy == 'Self-planning':
                self_plan_token += token
            if exec_strategy == 'Progressive-Hint':
                progressive_hint_token += token
            if exec_strategy == 'Persona':
                persona_token += token


    pred_avg_token = pred_token / total
    zero_shot_avg_token = zero_shot_token / total
    zero_cot_avg_token = zero_cot_token / total
    few_shot_avg_token = few_shot_token / total
    few_cot_avg_token = few_cot_token / total
    self_avg_token = self_token / total
    reflection_avg_token = reflection_token / total
    self_plan_avg_token = self_plan_token / total
    progressive_hint_avg_token = progressive_hint_token / total
    persona_avg_token = persona_token / total

    return pred_avg_token, zero_shot_avg_token, zero_cot_avg_token, few_shot_avg_token, few_cot_avg_token, self_avg_token, reflection_avg_token, self_plan_avg_token, progressive_hint_avg_token, persona_avg_token


# def calculate_avg_rank(data_list, model, model_path):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.eval()
#     model.to(device)

   
#     questions = [per_data['prompt'] for per_data in data_list]
#     embeddings = get_embedding(questions, model_path)

   
#     if isinstance(embeddings, list):
#         embeddings = torch.tensor(embeddings).to(device)

    
#     with torch.no_grad():
#         outputs = model(embeddings)
#         predictions = torch.argmax(outputs, dim=1).cpu().numpy()

    
#     pred_rank = 0
#     zero_shot_rank = 0
#     zero_cot_rank = 0
#     few_shot_rank = 0
#     few_cot_rank = 0
#     self_rank = 0
#     reflection_rank = 0
#     self_plan_rank = 0
#     progressive_hint_rank = 0
#     persona_rank = 0
#     total = len(data_list)

  
#     strategy_dict = {
#         0: 'MBPP_Zeroshot_0.0_gpt-3.5-turbo-0125.jsonl',
#         1: 'MBPP_Zeroshot_CoT_0.0_gpt-3.5-turbo-0125.jsonl',
#         2: 'MBPP_Fewshot_0.0_gpt-3.5-turbo-0125.jsonl',
#         3: 'MBPP_Fewshot_CoT_0.0_gpt-3.5-turbo-0125jsonl',
#         4: 'MBPP_SelfDebug_0.0_gpt-3.5-turbo-0125.jsonl',
#         5: 'MBPP_Reflection_0.0_gpt-3.5-turbo-0125.jsonl',
#         6: 'MBPP_SelfPlan_0.0_gpt-3.5-turbo-0125.jsonl',
#         7: 'MBPP_ProgressiveHint_0.0_gpt-3.5-turbo-0125.jsonl',
#         8: 'MBPP_Persona_0.0_gpt-3.5-turbo-0125.jsonl'
#     }

    
#     for idx, per_data in enumerate(data_list):
#         # y_pred = predictions[idx]
#         # y_pred = random.choice([0, 4])
#         y_pred = random.randint(0, 8)
#         print(y_pred)

#         strategy = strategy_dict[y_pred]
#         exec_record = per_data['exec_record']

#         for idx1, exec in enumerate(exec_record):
#             exec_strategy = exec['strategy']
#             exec_acc = exec['exec_acc']

#             if exec_acc == 0:
#                 idx1 = 9
#             else:
#                 idx1 += 1

#             if exec_strategy == strategy:
#                 pred_rank += idx1
#             if exec_strategy == 'MBPP_Zeroshot_0.0_gpt-3.5-turbo-0125.jsonl':
#                 zero_shot_rank += idx1
#             if exec_strategy == 'MBPP_Zeroshot_CoT_0.0_gpt-3.5-turbo-0125.jsonl':
#                 zero_cot_rank += idx1
#             if exec_strategy == 'MBPP_Fewshot_0.0_gpt-3.5-turbo-0125.jsonl':
#                 few_shot_rank += idx1
#             if exec_strategy == 'MBPP_Fewshot_CoT_0.0_gpt-3.5-turbo-0125.jsonl':
#                 few_cot_rank += idx1
#             if exec_strategy == 'MBPP_SelfDebug_0.0_gpt-3.5-turbo-0125.jsonl':
#                 self_rank += idx1
#             if exec_strategy == 'MBPP_Reflection_0.0_gpt-3.5-turbo-0125.jsonl':
#                 reflection_rank += idx1
#             if exec_strategy == 'MBPP_SelfPlan_0.0_gpt-3.5-turbo-0125.jsonl':
#                 self_plan_rank += idx1
#             if exec_strategy == 'MBPP_ProgressiveHint_0.0_gpt-3.5-turbo-0125.jsonl':
#                 progressive_hint_rank += idx1
#             if exec_strategy == 'MBPP_Persona_0.0_gpt-3.5-turbo-0125.jsonl':
#                 persona_rank += idx1

#     pred_avg_rank = pred_rank / total
#     zero_shot_avg_rank = zero_shot_rank / total
#     zero_cot_avg_rank = zero_cot_rank / total
#     few_shot_avg_rank = few_shot_rank / total
#     few_cot_avg_rank = few_cot_rank / total
#     self_avg_rank = self_rank / total
#     reflection_avg_rank = reflection_rank / total
#     self_plan_avg_rank = self_plan_rank / total
#     progressive_hint_avg_rank = progressive_hint_rank / total
#     persona_avg_rank = persona_rank / total

#     return pred_avg_rank, zero_shot_avg_rank, zero_cot_avg_rank, few_shot_avg_rank, few_cot_avg_rank, self_avg_rank, reflection_avg_rank, self_plan_avg_rank, progressive_hint_avg_rank, persona_avg_rank


# truth_file_path = 'result/Final_result/HumanEval_gpt4o_nine_rank.jsonl'
model_path = f'PET_model_result/code_complex_contrastive_model'
test_file_path = f'PET_model_dataset/code_complex_classification_dataset_test.jsonl'
test_data = list(map(json.loads, open(test_file_path)))

embeddings = get_embedding([test_data[0]['prompt']], model_path)
input_size = len(embeddings[0])
num_classes = 9
test_embedding_model_path = f'PET_model_result/code_complex_contrastive_model'
test_classification_model_path = f'PET_model_result/classification_model/code_complex_classification_model_parameters.pth'
test_classification_model = ClassificationModel(input_size, num_classes)
test_classification_model.load_state_dict(torch.load(test_classification_model_path))
# data = list(map(json.loads, open(truth_file_path)))

# data_list = generate_data_list(data, test_data)

print('\n----Actual ACC----\n')

pred_acc, zero_shot_acc, zero_cot_acc, few_shot_acc, few_cot_acc, self_acc, reflection_acc, self_plan_acc, progressive_hint_acc, persona_acc, zero_shot_correct = calculate_actual_acc(test_data, test_classification_model, test_embedding_model_path)
print(f'Pred accuracy: {pred_acc * 100:.2f}%; \nZeroshot accuracy: {zero_shot_acc * 100:.2f}%; \nZeroshot CoT accuracy: {zero_cot_acc * 100:.2f}%; \nFewshot accuracy: {few_shot_acc * 100:.2f}%; \nFewshot CoT accuracy: {few_cot_acc * 100:.2f}%; \nPersona accuracy: {persona_acc * 100:.2f}%; \nSelf-planning accuracy: {self_plan_acc * 100:.2f}%; \nSelf-refine accuracy: {reflection_acc * 100:.2f}%; \nProgressive accuracy: {progressive_hint_acc * 100:.2f}%; \nSelfDebug accuracy: {self_acc * 100:.2f}%;    ')
print('zeroshot correct', zero_shot_correct)

print('\n----Token saved----\n')

pred_avg_token, zero_shot_avg_token, zero_cot_avg_token, few_shot_avg_token, few_cot_avg_token, self_avg_token, reflection_avg_token, self_plan_avg_token, progressive_hint_avg_token, persona_avg_token = calculate_token_saved(test_data, test_classification_model, test_embedding_model_path)
print(f'Pred avg token: {pred_avg_token}; \nZeroshot avg token: {zero_shot_avg_token}; \nZeroshot CoT avg token: {zero_cot_avg_token}; \nFewshot avg token: {few_shot_avg_token}; \nFewshot CoT avg token: {few_cot_avg_token}; \nPersona avg token: {persona_avg_token}; \nSelf-planning avg token: {self_plan_avg_token};  \nSelf-refine avg token: {reflection_avg_token};  \nProgressive avg token: {progressive_hint_avg_token}; \nSelfDebug avg token: {self_avg_token};')

# print('\n----Avg Rank----\n')

# pred_avg_rank, zero_shot_avg_rank, zero_cot_avg_rank, few_shot_avg_rank, few_cot_avg_rank, self_avg_rank, reflection_avg_rank, self_plan_avg_rank, progressive_hint_avg_rank, persona_avg_rank = calculate_avg_rank(test_data, test_classification_model, test_embedding_model_path)
# print(f'Pred avg rank: {pred_avg_rank}; \nZeroshot avg rank: {zero_shot_avg_rank}; \nZeroshot CoT avg rank: {zero_cot_avg_rank}; \nFewshot avg rank: {few_shot_avg_rank}; \nFewshot CoT avg rank: {few_cot_avg_rank}; \nSelfDebug avg rank: {self_avg_rank}; \nReflection avg rank: {reflection_avg_rank}; \nSelfPlan avg rank: {self_plan_avg_rank}; \nProgressive avg rank: {progressive_hint_avg_rank}; \nPersona avg rank: {persona_avg_rank}')
