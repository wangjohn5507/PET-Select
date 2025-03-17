import json
from src import utils
import tqdm

file_path = 'dataset/HumanEval_category.jsonl'
plus_file_path = 'dataset/HumanEval_plus.jsonl'
dataset = list(map(json.loads, open(file_path)))
plus_dataset = list(map(json.loads, open(plus_file_path)))


def load_HumanEval_plus_dataset():
    final_dataset = []
    for idx, data in enumerate(tqdm.tqdm(plus_dataset)):
        # prompt; task_id; entry_point; ground_truth code; test; category; code_complexity
        record = dict()
        record['prompt'] = data['prompt']
        record['task_id'] = idx
        record['entry_point'] = data['entry_point']
        record['ground_truth_code'] = data['canonical_solution']
        record['test'] = data['test']
        record['category'] = dataset[idx]['category']
        # record['code_complexity'] = utils.calculate_weighted_complexity(data['canonical_solution'], '', True, True, False)
        
        final_dataset.append(record)
    return final_dataset
    
    