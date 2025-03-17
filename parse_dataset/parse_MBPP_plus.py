import json
from src import utils
import tqdm

file_path = 'dataset/MBPP_category.jsonl'
plus_file_path = 'dataset/MBPP_plus.jsonl'
dataset = list(map(json.loads, open(file_path)))
plus_dataset = list(map(json.loads, open(plus_file_path)))


def load_MBPP_plus_dataset():
    final_dataset = []
    for idx, data in enumerate(tqdm.tqdm(plus_dataset)):
        # prompt; task_id; ground_truth code; test; category; code_complexity
        record = dict()
        record['prompt'] = data['prompt']
        record['task_id'] = idx
        record['ground_truth_code'] = data['code']
        record['category'] = dataset[data['task_id']-1]['category']
        record['test'] = data['test']
        record['test_list'] = data['test_list']
        # record['code_complexity'] = utils.calculate_weighted_complexity(data['code'], '', False, True, False)

        final_dataset.append(record)
    return final_dataset
    
    