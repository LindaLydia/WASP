import pandas as pd
import jsonlines
import sys, os

file_dir = './aug-pe-real-data/yelp/'

create_label_map = True
# create_label_map = False

for root, dirs, files in os.walk(file_dir):
    for file in files:
        if file.endswith('.csv'):
            # Print the path of the text file
            file_path = os.path.join(root, file)
            mode = str(file.split('_')[-1][:-4])
            print(f"gold data for {mode}")
            df = pd.read_csv(file_path)
            print(f"{df.shape=}")
            # ####### create the label mapping dictionary #######
            if create_label_map == True:
                label1 = list(df['label1'])
                label2 = list(df['label2'])
                unique_label1 = list(set(label1))
                unique_label2 = list(set(label2))
                unique_label1.sort()
                unique_label2.sort()
                print(f"{unique_label1=}, {len(unique_label1)=}")
                print(f"{unique_label2=}, {len(unique_label2)=}")
                label1_map, label2_map = {}, {}
                for _i, _label in enumerate(unique_label1):
                    label1_map[_i] = _label
                for _i, _label in enumerate(unique_label2):
                    label2_map[_i] = _label
                with jsonlines.open('./data/yelpCategory/label_map.jsonl', 'w') as writer:
                    writer.write(label1_map)
                with jsonlines.open('./data/yelpRating/label_map.jsonl', 'w') as writer:
                    writer.write(label2_map)
            ####### create the label mapping dictionary #######
            # ####### read the label mapping dictionary #######
            with jsonlines.open('./data/yelpCategory/label_map.jsonl', 'r') as reader:
                for obj in reader:
                    label1_map = obj
            with jsonlines.open('./data/yelpRating/label_map.jsonl', 'r') as reader:
                for obj in reader:
                    label2_map = obj
            label1_map, label2_map = dict(label1_map), dict(label2_map)
            label1_map = {value: int(key) for key, value in label1_map.items()}
            label2_map = {value: int(key) for key, value in label2_map.items()}
            # ####### read the label mapping dictionary #######

            with jsonlines.open(f'./data/yelpCategory/std/{mode}.jsonl', 'w') as writer1, jsonlines.open(f'./data/yelpRating/std/{mode}.jsonl', 'w') as writer2:
                for index, row in df.iterrows():
                    # print(index, row['text'], row['label1'], row['label2'])
                    # print(index, type(row['text']), type(row['label1']), type(row['label2']))
                    label1 = int(label1_map[row['label1']])
                    label2 = int(label2_map[row['label2']])
                    print(f"{row['label1']=}, {label1=}")
                    print(f"{row['label2']=}, {label2=}")
                    writer1.write({"idx": index, "text": row['text'], "label": label1})
                    writer2.write({"idx": index, "text": row['text'], "label": label2})

''' labels
unique_label1=[
    'Business Category: Arts & Entertainment', 
    'Business Category: Bars', 
    'Business Category: Beauty & Spas', 
    'Business Category: Event Planning & Services', 
    'Business Category: Grocery', 
    'Business Category: Health & Medical', 
    'Business Category: Home & Garden', 
    'Business Category: Hotels & Travel', 
    'Business Category: Restaurants', 
    'Business Category: Shopping'], # len(unique_label1)=10
unique_label2=[
    'Review Stars: 1.0', 
    'Review Stars: 2.0', 
    'Review Stars: 3.0', 
    'Review Stars: 4.0', 
    'Review Stars: 5.0'], # len(unique_label2)=5
'''