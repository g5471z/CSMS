import jsonlines
from runrun import run2,TokenizerCache2
from run import run1,TokenizerCache1
from tqdm import tqdm
import os

# This determines the dataset you will generate.There are test, validation, training.
name = 'train.txt'
batch_size = 500
token_cache1 = TokenizerCache1()
token_cache2 = TokenizerCache2()

# Read ID order (preprocessed to list of integers)
with open(f'./HRCE/{name}', 'r') as f:
    id_order = [line.strip() for line in f]

# Preload data into the dictionary
data_dict = {}
with open(r'./HRCE/data.jsonl', encoding='utf-8') as f:
    for item in jsonlines.Reader(f):
        data_dict[str(item['id'])] = item

# Initialize buffer
buffer1, buffer2, buffer3 = [], [], []

with open('./data/train/code.txt', 'a', encoding='utf-8') as s1, \
     open('./data/train/sum.txt', 'a', encoding='utf-8') as s2,  \
     open('./data/train/originalcode.txt', 'a', encoding='utf-8') as s3 :

    # Process ids sequentially
    for idx,current_id in enumerate(tqdm(id_order)):

        item = data_dict[current_id]

        # Process content
        buffer1.append(run2(item['content'], token_cache2) + '\n')

        buffer3.append(run1(item['content'], token_cache1) + '\n')

        # Process summary
        summary_str = str(item['summary']) + '.\n'
        buffer2.append(summary_str)
        # Batch write
        if len(buffer1) >= batch_size:
            s1.writelines(buffer1)
            s2.writelines(buffer2)
            s3.writelines(buffer3)

            buffer1, buffer2, buffer3 = [], [], []


    # Write residual data
    if buffer1:
        s1.writelines(buffer1)
        s2.writelines(buffer2)
        s3.writelines(buffer3)
