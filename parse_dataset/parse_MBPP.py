import json
import ast
from src import utils
import tqdm


file_path = 'dataset/MBPP_category.jsonl'
plus_file_path = 'dataset/MBPP_plus.jsonl'
dataset = list(map(json.loads, open(file_path)))
plus_dataset = list(map(json.loads, open(plus_file_path)))

def load_MBPP_dataset():
    final_dataset = []
    sanitized_id = [] 
    id = 0
    for d in plus_dataset:
        sanitized_id.append(d['task_id'])
    for idx, data in enumerate(tqdm.tqdm(dataset)):
        # prompt; task_id; ground_truth code; test; category; code_complexity
        if data['task_id'] not in sanitized_id:
            continue
        record = dict()
        record['prompt'] = data['text']
        record['task_id'] = id
        record['ground_truth_code'] = data['code']
        record['category'] = data['category']
        record['test'] = data['test_list']
        record['test_list'] = data['test_list']
        # record['code_complexity'] = utils.calculate_weighted_complexity(data['code'], '', False, False, False)
        
        final_dataset.append(record)
        id += 1
    return final_dataset
    
    