import pandas as pd
import tqdm
import json
import copy

from model import get_embedding
from utils import write_to_file

input_file = 'dataset/balanced_rank.jsonl'
output_file ='dataset/balanced_embedding.jsonl'
embedding_model = 'text-embedding-3-large'


def record_embedding():
    data = list(map(json.loads, open(input_file)))
    for per_data in tqdm.tqdm(data):
        x = copy.copy(per_data)
        x['embedding'] = get_embedding(x['question'], embedding_model)
        write_to_file(x, output_file)

if __name__ == '__main__':
    record_embedding()
