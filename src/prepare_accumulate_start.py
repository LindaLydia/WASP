import jsonlines
import os, sys
import random

# model_names = ['gpt2-xl', 'llama-2-7b-chat-hf', 'vicuna-7b-1.5v', 'opt-6.7b', 'chatglm3-6b-base', 'flan-t5-xl']
# target_folder = ['100_20', '200_40', '500_100', '1000_100', '1000_500']
# target_samples = [20,40,100,100,500]

task_name = 'squad'
task_name = 'mnli'
task_name = 'banking77'
task_name = 'worksheet'
task_name = 'imdb'
# task_name = 'mnliMisM'
# task_name = 'markednews'

model_names = ['gpt2-xl', 'llama-2-7b-chat-hf', 'vicuna-7b-1.5v', 'opt-6.7b', 'chatglm3-6b-base', 'flan-t5-xl'] #
model_names = ['chatglm3-6b-base', 'llama-3-8b-chinese-chat'] #
model_names = ['gpt-4o'] #

# target_folder = ['100_20', '1000_200', '1000_500', '6000_1200']
# target_samples = [20, 200, 500, 1200]
target_folder = ['100_20', '1000_200', '2000_400', '3000_600', '6000_1200']
target_samples = [20, 200, 400, 600, 1200]


for model in model_names:
    for folder, num_samples in zip(target_folder, target_samples):
        # input_file_path = f'./data_accumulate_start/{task_name}/{model}/10000_2000/train.jsonl'
        # input_file_path = f'./data_accumulate_start/{task_name}/{model}/6000_1200/train.jsonl'
        input_file_path = f'./data_new/{task_name}/{model}/10000/train.jsonl'
        output_file_path = f'./data_accumulate_start/{task_name}/{model}/{folder}/'
        if not os.path.exists(output_file_path):
            os.makedirs(output_file_path)
        # with jsonlines.open(input_file_path, 'r') as reader, jsonlines.open(output_file_path+'train.jsonl', 'w') as writer:
        #     _counter = 0
        #     for json_obj in reader:
        #         writer.write(json_obj)
        #         _counter += 1
        #         if _counter == num_samples:
        #             break
        lines = []
        with jsonlines.open(input_file_path, 'r') as reader:
            for json_obj in reader:
                lines.append(json_obj)
        random.shuffle(lines)
        with jsonlines.open(output_file_path+'train.jsonl', 'w') as writer:
            _counter = 0
            for json_obj in lines:
                writer.write(json_obj)
                _counter += 1
                if _counter == num_samples:
                    break
