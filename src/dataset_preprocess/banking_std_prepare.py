import jsonlines
import pandas as pd
import pyarrow.parquet as pq
import random


input_file = './data/banking/test-00000-of-00001.parquet'
output_file = './data/banking/std/test.jsonl'

# input_file = './data/banking/train-00000-of-00001.parquet'
# output_file = './data/banking/std/train.jsonl'

# Replace 'your_file.parquet' with the path to your Arrow file
table = pq.read_table(input_file)

# Convert to pandas DataFrame if needed
df = table.to_pandas()
print(df[:5])

with jsonlines.open(output_file, 'w') as writer:
    obj_list = []
    for index, row in df.iterrows():
        if row['label'] < 10:
            obj = {
                "idx": index,
                "text": row['text'],
                "label": row['label'],
            }
            obj_list.append(obj)
    if 'train' in input_file:
        random.shuffle(obj_list)
    for io, obj in enumerate(obj_list):
        obj["idx"] = io
        writer.write(obj)


########### test part ###########
with jsonlines.open(output_file, 'r') as reader:
    for json_obj in reader:
        print(json_obj['text'][:40])
        if json_obj['idx'] > 5:
            break
########### test part ###########
