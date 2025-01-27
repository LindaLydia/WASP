import jsonlines
import os, sys
import random

# SYN_DATA_PATH = 'data_new/'
# TASK_NAME = 'imdb'
# MODEL_NAME = 'gpt2-xl'
# MODEL_NAME = 'llama-2-7b-chat-hf'
# mode = 'train'
# # mode = 'test'

# # Define the path to your input JSONL file and output JSONL file
# input_file_path = f'{SYN_DATA_PATH}{TASK_NAME}/{MODEL_NAME}/{mode}_original.jsonl'
# output_file_path = f'{SYN_DATA_PATH}{TASK_NAME}/{MODEL_NAME}/{mode}.jsonl'

SYN_DATA_PATH = 'data_new/'
# SYN_DATA_PATH = 'data_new_few_shot_ambiguous/'
# SYN_DATA_PATH = 'data_new_few_shot_easytolearn/'
# TASK_NAME = 'imdb'
# TASK_NAME = 'yelp'
TASK_NAME = 'qnli'
TASK_NAME = 'mnli'
TASK_NAME = 'agnews'
TASK_NAME = 'markednews'
TASK_NAME = 'banking77'
TASK_NAME = 'squad'
TASK_NAME = 'yelpCategory'
TASK_NAME = 'yelpRating'
# TASK_NAME = 'openreviewCategory'
# TASK_NAME = 'openreviewRating'
# TASK_NAME = 'banking'
# MODEL_NAME = 'gpt2-xl'
# MODEL_NAME = 'llama-2-7b-chat-hf'
# MODEL_NAME = 'vicuna-7b-1.5v'
mode = 'train'
# mode = 'test'

task_name_map = {'imdb':'imdb', 'mnli':'mnli','qnli':'qnli', 'agnews':'ag_news', 'squad':'squad', 'markednews':'markednews', 'banking77':'banking77', 'banking':'banking77',
                 'yelpCategory':'yelpbusiness', 'yelpRating':'yelpbusiness', 'openreviewCategory':'openreview', 'openreviewRating':'openreview'}

# Define the path to your input JSONL file and output JSONL file
input_file_path_dir = f'/home/DAIR/zouty/ModelFederation/temp/ZeroGen/yelpbusiness/output/yelpbusiness-x2-200k/'

input_file_path_dir = f'/dev/data/zouty0/ModelFederation/PrivateGenerateEnhancement/src/data_new/{task_name_map[TASK_NAME]}/'

# Function to modify each JSON object
def modify_json_object(json_obj, counter):
    # Modify the JSON object as needed
    # For example, let's add a new field "modified" with a value of True
    json_obj['idx'] = counter
    if 'Rating' in TASK_NAME:
        json_obj['Y'] = int(int(json_obj['Y']) % 5)
    elif 'Category' in TASK_NAME:
        json_obj['Y'] = int(int(json_obj['Y']) // 5)
    return json_obj

if TASK_NAME == 'imdb':
    for root, dirs, files in os.walk(input_file_path_dir):
        for dir_name in dirs:
            # Print the full path of each subdirectory
            subdirectory_path = os.path.join(root, dir_name)
            if os.path.isdir(subdirectory_path):
                print(subdirectory_path)
                input_file_path = f'{str(subdirectory_path)}/{TASK_NAME}-dataset.jsonl'
                _temp = subdirectory_path.split('/')[-1]
                print(_temp)
                _temp = _temp.split('_')
                MODEL_NAME = _temp[0]
                SAMPLE_COUNT = int(_temp[-1].split('[')[-1].strip('[]'))
                # if (not 'gpt2' in MODEL_NAME) or (SAMPLE_COUNT != 20000):
                #     continue
                print(f'MODEL_NAME={MODEL_NAME}, SAMPLE_COUNT={SAMPLE_COUNT}')            
                output_file_path = f'{SYN_DATA_PATH}{TASK_NAME}/{MODEL_NAME}/{SAMPLE_COUNT}/{mode}.jsonl'
                output_file_dir = f'{SYN_DATA_PATH}{TASK_NAME}/{MODEL_NAME}/{SAMPLE_COUNT}/'
                print(f"output_file_path={output_file_path}")
                if not os.path.exists(output_file_dir):
                    os.makedirs(output_file_dir)
                
                # Read the input JSONL file and write to the output JSONL file
                counter = 0
                with jsonlines.open(input_file_path, 'r') as reader, jsonlines.open(output_file_path, 'w') as writer:
                    for json_obj in reader:
                        # Modify the JSON object
                        modified_json_obj = modify_json_object(json_obj, counter)
                        counter += 1
                        
                        # Write the modified JSON object to the output file
                        writer.write(modified_json_obj)

                print("JSONL file processing complete.")

else:
    for root, dirs, files in os.walk(input_file_path_dir):
        for file in files:
            input_file_path = os.path.join(root, file)
            # if f'gpt-4o/10000/train_temp.jsonl'
            if f'train_temp.jsonl' in input_file_path:
            # if f'{task_name_map[TASK_NAME]}-dataset.jsonl' in input_file_path: # and ('gpt2' in input_file_path or 'chatglm3' in input_file_path)
                print(f"{input_file_path=}")
                _temp = input_file_path.split('/')
                # print(f"{_temp=}")
                MODEL_NAME = _temp[-3]
                if 'k' in _temp[-2]:
                    SAMPLE_COUNT = int(_temp[-2].split('[')[-1].strip('[]').strip('k'))*1000
                else:
                    SAMPLE_COUNT = int(_temp[-2].split('[')[-1].strip('[]'))
                print(f'MODEL_NAME={MODEL_NAME}, SAMPLE_COUNT={SAMPLE_COUNT}')            
                output_file_path = f'{SYN_DATA_PATH}{TASK_NAME}/{MODEL_NAME}/{SAMPLE_COUNT}/{mode}.jsonl'
                output_file_dir = f'{SYN_DATA_PATH}{TASK_NAME}/{MODEL_NAME}/{SAMPLE_COUNT}/'
                print(f"output_file_path={output_file_path}")
                if not os.path.exists(output_file_dir):
                    os.makedirs(output_file_dir)
                
                # Read the input JSONL file and write to the output JSONL file
                counter = 0
                obj_list = []
                with jsonlines.open(input_file_path, 'r') as reader, jsonlines.open(output_file_path, 'w') as writer:
                    for json_obj in reader:
                        obj_list.append(json_obj)
                    
                    # random shuffle the list to make the first [:xx] items have the same label proportion as the total one
                    random.shuffle(obj_list)

                    for json_obj in obj_list: 
                        # Modify the JSON object
                        modified_json_obj = modify_json_object(json_obj, counter)
                        counter += 1  
                        # Write the modified JSON object to the output file
                        writer.write(modified_json_obj)

                print("JSONL file processing complete.")





# SYN_DATA_PATH = '/dev/data/zouty0/ModelFederation/GeneratedDatasetFusion/src/data/imdb/std/'

# # Function to modify each JSON object
# def modify_json_object(json_obj):
#     # Modify the JSON object as needed
#     # For example, let's add a new field "modified" with a value of True
#     json_obj['text_2'] = json_obj['text']
#     return json_obj

# for mode in ['train', 'test', 'test_small']:

#     input_file_path = f'{SYN_DATA_PATH}/{mode}.jsonl'
#     output_file_path = f'{SYN_DATA_PATH}/{mode}_expend.jsonl'
#     output_file_dir = f'{SYN_DATA_PATH}/'
#     print(f"output_file_path={output_file_path}")
    
#     # Read the input JSONL file and write to the output JSONL file
#     with jsonlines.open(input_file_path, 'r') as reader, jsonlines.open(output_file_path, 'w') as writer:
#         for json_obj in reader:
#             # Modify the JSON object
#             modified_json_obj = modify_json_object(json_obj)
            
#             # Write the modified JSON object to the output file
#             writer.write(modified_json_obj)

#     print("JSONL file processing complete.")
