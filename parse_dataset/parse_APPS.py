import json
from src import utils
import tqdm
from datasets import load_dataset

file_path = 'dataset/APPS.jsonl'

ori_data = load_dataset("codeparrot/apps", split="test")
dataset = list(map(json.loads, open(file_path)))


def load_apps_dataset():
    final_dataset = []
    for idx, data in enumerate(tqdm.tqdm(dataset)):
        # prompt; task_id; entry_point; ground_truth code; test; category; code_complexity
        record = dict()
        if ori_data[idx]['solutions'] == '':
            continue
        record['prompt'] = data['prompt']
        record['task_id'] = idx
        record['entry_point'] = data['entry_point']
        record['ground_truth_code_list'] = json.loads(ori_data[idx]['solutions'])
        record['ground_truth_code'] = json.loads(ori_data[idx]['solutions'])[0]
        # print(json.loads(ori_per_data['solutions'])[0])
        record['test'] = data['test']
        record['meta_data'] = dataset[idx]['meta_data']
        # record['code_complexity'] = utils.calculate_weighted_complexity(json.loads(ori_per_data['solutions'])[0], ori_data, False, False, True)
        
        final_dataset.append(record)
    return final_dataset