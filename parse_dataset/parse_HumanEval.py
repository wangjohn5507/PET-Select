import json
import ast
from src import utils
import tqdm

file_path = 'dataset/HumanEval_category.jsonl'
dataset = list(map(json.loads, open(file_path)))


def load_HumanEval_dataset():
    final_dataset = []
    for idx, data in enumerate(tqdm.tqdm(dataset)):
        # prompt; task_id; entry_point; ground_truth code; test; category; code_complexity
        record = dict()
        record['prompt'] = data['prompt']
        record['task_id'] = idx
        record['entry_point'] = data['entry_point']
        record['ground_truth_code'] = data['canonical_solution']
        record['category'] = data['category']
        record['test'] = data['test']
        # record['code_complexity'] = utils.calculate_weighted_complexity(data['canonical_solution'], '', True, False, False)

        final_dataset.append(record)
    return final_dataset
    
    