import base64
import requests
from copy import deepcopy
import random
import os
from tqdm import tqdm
import re
import numpy as np
import requests
import json
from requests.auth import HTTPBasicAuth
import jsonlines
import argparse

from utils.basic_utils import init_logging, set_seed, read_jsonl, save_jsonl, save_jsonl_append, modify_idx_in_json_file
from utils.constant import PROMPTS
from utils.cls_generator import C_KEY, X_KEY, Y_KEY, PLACEHOLDER_C, PLACEHOLDER_X
from typing import List, Optional, Dict, Any, Union


def process_output_gpt(input_text: Union[str, int], output_text: str, label: str, generate_with_inputs: bool,
                   min_length: int, task_name: str) -> Optional[Dict]:
    if task_name == "qnli":
        if '?' in output_text:
            output_text = output_text.split('?')[0] + "?"
        else:
            return None
    elif '"' in output_text:
        output_text = output_text.split('"')[0]
    elif '\n' in output_text:
        output_text = output_text.split('\n')[0]
    elif '.' in output_text:
        sentences = output_text.split('.')
        output_text = '.'.join(sentences[:-1]) + '.'
    else:
        return None

    if len(output_text.strip().split(' ')) >= min_length:
        if generate_with_inputs:
            c = input_text
            x = output_text
        else:
            c = output_text
            x = None
        return {C_KEY: c, X_KEY: x, Y_KEY: float(label) if task_name == "stsb" else int(label)}
    return None


def postprocess_dataset_gpt(dataset: List[Dict], generate_with_inputs: bool, task_name: str) -> List[Dict]:
    postprocessed_dataset = []
    for example in dataset:
        if generate_with_inputs:  # force the generated x to be different from c
            if example[C_KEY] == example[X_KEY]:
                continue
        if 'nli' in task_name:
            # print("here")
            example['X0'] = example[C_KEY] # 'C' for sentence1[SEP]sentence2, 'X0' for sentence1, 'X' for sentence2
            example[C_KEY] = example['X0'] + '[SEP]' + example[X_KEY]

        postprocessed_dataset.append(json.dumps(example))
    postprocessed_dataset = [json.loads(i) for i in list(dict.fromkeys(postprocessed_dataset))]
    return postprocessed_dataset


def GPT_generation(args, inputs):

    api_key = 'sk-Il27yhmXdWJkKNqJE74d280b80Cf4b3aB86fB58fA271Fd95'
    url_gpt4 = 'https://chat1.plus7.plus/v1/chat/completions'
    url_completion = 'https://chat1.plus7.plus/v1/completions'
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    prompts = args.task_specification["labels"]
    log_every = args.gen_log_every
    task_name = args.task_name
    model = args.gen_model_name

    _idx = 0
    input_idx = 0
    outputs = []
    for key in prompts.keys():
        for _i in range((int(args.gen_num_entries_per_input*1.5))//2):
            label = int(key)
            # if label == 0:
            #     continue
            gen_prompt = prompts[key]["instruction"]
            if 'nli' in args.task_name:
                gen_prompt = gen_prompt.replace(PLACEHOLDER_C, inputs[input_idx])
            if 'gpt-4' in model:
                data = {
                    "model": model,
                    "messages": [{
                        "role": "user", 
                        "content": [
                            {
                                "type": "text",
                                "text": f"{gen_prompt}"
                            },
                        ]
                    }],
                    "max_tokens": 100,
                    "temperature": 1.0,
                    "top_p": 0.9
                }
                response = requests.post(url_gpt4, headers=headers, json=data).text
                # print(response)
            elif 'gpt-3.5' in model:
                data = {
                    "model": model,
                    # "messages": [
                    #     { "role": 'system', "content": "You are a helpful assistant." },
                    #     # { "role": 'system', "content": "You are a helpful assistant. Refer to the block of given text below:\n${textarea.value}" },
                    
                    #     { "role": 'user', "content": f"{gen_prompt}" }
                    # ],
                    "prompt": f"{gen_prompt}",
                    "max_tokens": 50,
                    'temperature': 1.0,  # You can adjust temperature if needed
                    # 'top_k': 0,       # Your top_k value
                    'top_p': 0.9,       # Your top_p value
                    'frequency_penalty': 1.0, # >0 ecourages the model to explore more diverse and novel responses
                    'presence_penalty': 1.0, # >0 ecourages the model to explore more diverse and novel responses
                }
                response = requests.post(url_completion, headers=headers, json=data).text
                # print(response)
            response_dict = eval(response.replace('null', 'None').replace('false', 'False').replace('true', 'True'))
            input_idx += 1
            # print(response_dict)
            if 'gpt-4' in model or 'instruct' not in model:
                try:
                    text = response_dict["choices"][0]["message"]["content"].strip().lower().replace('\n\n', '').replace('','').replace('\\','') #.replace('.','')
                    # print(f'{response_dict["choices"][0]=}')
                    # print(f'{response_dict["choices"][0]["message"]=}')
                    # print(f'{response_dict["choices"][0]["message"]["content"]=}')
                    print(f'{text=}')
                except:
                    print("[error]", response_dict)
                    continue
            elif 'gpt-3.5-turbo-instruct' in model:
                try:
                    text = response_dict["choices"][0]["text"].strip().lower().replace('\n\n', '').replace('','').replace('\\','') #.replace('.','')
                    print(f"{text=}")
                except:
                    print("[error]", response_dict)
                    continue
            if text[-1] == '"':
                text = text[:-1]
            if text[0] == '"':
                text = text[1:]
            if len(text) == 0:
                continue
            
            output_dict = process_output_gpt(input_text=gen_prompt,
                                        output_text=text,
                                        label=label, generate_with_inputs=(inputs is not None),
                                        min_length=args.gen_min_length, task_name=args.task_name)
            print(f"{output_dict=}")
            if output_dict != None:
                outputs.append(output_dict)
                # writer.write(json_obj)
                _idx += 1
                print(f"generated: {_idx}, on going...")
    print(f"post processing")
    outputs = postprocess_dataset_gpt(outputs, generate_with_inputs=(inputs is not None), task_name=args.task_name)

    return outputs


def gen_syn_data_few_shot_gpt_api(args):
    if args.task_name in ['mnli', 'qnli']:
        args.gen_input_file = '../utils/wiki_data/wiki_short.jsonl'
    else:
        args.gen_input_file = None
    hasattr(args, 'gen_output_dir')# "The output directory to which the generated dataset is saved"
    hasattr(args, 'gen_task_file') # "A json file providing the instructions and other information required for dataset generation. "
    hasattr(args, 'gen_input_file') # "An input file containing the generated data"
    hasattr(args, 'gen_model_name') # "The pretrained model to use for dataset generation. Currently, only variants of GPT2 are supported."
    hasattr(args, 'gen_batch_size') # "The batch size for generation (only if --input_file is not set)"
    hasattr(args, 'gen_num_entries_per_input')
    hasattr(args, 'gen_max_length')
    hasattr(args, 'gem_min_length')
    args.gen_top_p = 0.9
    args.gen_top_k = 0
    args.gen_temperature = 1.0
    args.gen_decay_constant = 200
    args.gen_log_every = 10000
    # hasattr(args, 'output_dir')
    with open(args.gen_task_file, 'r', encoding='utf8') as fh:
        task_specification = json.load(fh)
    args.task_specification = task_specification
    args.task_name = task_specification["task_name"]
    is_stage_two = task_specification['stage'] == 'x2'
    zero_shot = task_specification['stage'] == 'zs'
    assert is_stage_two==True and zero_shot==False, "[Error] Few-shot generation should be stage 2 with condition"

    logging = init_logging(log_file=args.gen_output_dir + '/output.log', stdout=True)

    if args.gen_input_file:
        logging.info(f"Use condition c from {args.gen_input_file}")
        inputs = [i[C_KEY] for i in read_jsonl(args.gen_input_file)]
        if 'nli' in args.task_name:
            inputs = [i[C_KEY] for i in read_jsonl(args.gen_input_file)]
            random.shuffle(inputs)
            inputs = inputs[:int(args.gen_num_entries_per_input*args.num_classes*(2 if 'glm' in args.gen_model_name else 1))]
    else:
        logging.info("Do not use condition c")
        inputs = None

    logging.info("Starting dataset generation...")
    # if 'gpt-4' in args.gen_model_name:
    #     outputs = GPT4_generation(args, inputs=inputs)
    # elif 'gpt-3.5' in args.gen_model_name:
    #     outputs = GPT3_generation(args, inputs=inputs)
    outputs = GPT_generation(args, inputs=inputs)
    # outputs = generator.generate_dataset(inputs, num_entries_per_input=(1 if 'nli' in args.task_name else args.gen_num_entries_per_input),
    #                                         batch_size=args.gen_batch_size, log_every=args.gen_log_every, task_name=args.task_name)

    # assert len(outputs) >= args.gen_num_entries_per_input, f"Error, requiring {args.gen_num_entries_per_input} samples, but only generated {len(outputs)} samples"
    while len(outputs) < args.gen_num_entries_per_input:
        outputs = outputs + outputs[:min(len(outputs), args.gen_num_entries_per_input-len(outputs))]
    assert len(outputs) >= args.gen_num_entries_per_input, f"Error, requiring {args.gen_num_entries_per_input} samples, but only generated & copied only {len(outputs)} samples"
    logging.info(f"Dataset generation complete, dataset contains {len(outputs)} entries")
    random.shuffle(outputs)
    # dataset_path = os.path.join(args.gen_output_dir, f'{task_specification["task_name"]}-dataset.jsonl')
    for sample_file_name in ['train_noflip', 'train']:
        dataset_path = os.path.join(args.gen_output_dir, f'{sample_file_name}.jsonl')
        save_jsonl_append(outputs[:args.gen_num_entries_per_input], dataset_path)
    logging.info(f"Done saving dataset to file '{dataset_path}'")

    # if is_stage_two:
    #     wandb.save(args.gen_output_dir)
    for sample_file_name in ['train_noflip', 'train']:
        save_path = os.path.join(args.gen_output_dir, f'{sample_file_name}.jsonl')
        modify_idx_in_json_file(dataset_path, save_path)


if __name__ == "__main__":

    PROMPT = {  "task_name": "qnli",
                "stage": "x2",
                "labels": {
                    "0": {
                        "instruction": "Information: \"<C>\"\nQuestion (answer in above information): \"",
                        "counter_labels": ["1"]
                    },
                    "1": {
                        "instruction": "Information: \"<C>\"\nQuestion (answer not in above information): \"",
                        "counter_labels": ["0"]
                    }
                }
            }
    

    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--gen_output_dir", type=str, default=None, #required=True,
                        help="The output directory to which the generated dataset is saved")
    parser.add_argument("--gen_task_file", type=str, default=None, #required=True,
                        help="A json file providing the instructions and other information required for dataset generation. ")

    # Dataset and prompt parameters
    parser.add_argument("--gen_input_file", type=str, default=None,
                        help="An optional input file containing raw texts. This is required for generating text pair datasets.")

    # Text generation and sampling parameters
    parser.add_argument("--gen_model_name", type=str, default="gpt2-xl",
                        help="The pretrained model to use for dataset generation. Currently, only variants of GPT2 are supported.")
    parser.add_argument("--gen_batch_size", type=int, default=None,
                        help="The batch size for generation (only if --input_file is not set)")
    parser.add_argument("--gen_num_entries_per_input", type=int, default=None,
                        help="The number of entries to generate for each label (only if --input_file is not set)")
    parser.add_argument("--gen_max_length", type=int, default=40,
                        help="The maximum output length for each generated text.")
    parser.add_argument("--gen_min_length", type=int, default=1,
                        help="Min length of generated text.")

    parser.add_argument("--i_step", default=0, type=int)
    parser.add_argument('--llms', default=['gpt2-xl','llama-2-7b-chat-hf'], nargs='+', type=str)

    args = parser.parse_args()

    # if args.llms == ['gpt-3.5-turbo-instruct', 'gpt-4-turbo-preview']:
    #     args.working_sample_dir = [
    #         f'./data_accumulate/imdb/gpt-3.5-turbo-instruct_1000__gpt-4-turbo-preview_1000/bert-base-uncased/increasedTheta_KD1_FuseDataset1/fewshotK8_5_0.5/12345/gpt-3.5-turbo-instruct/1000_200_200/',
    #         f'./data_accumulate/imdb/gpt-3.5-turbo-instruct_1000__gpt-4-turbo-preview_1000/bert-base-uncased/increasedTheta_KD1_FuseDataset1/fewshotK8_5_0.5/12345/gpt-4-turbo-preview/1000_200_200/',
    #     ]
    #     args.working_prompt_dir = [
    #         f'./data_accumulate/imdb/gpt-3.5-turbo-instruct_1000__gpt-4-turbo-preview_1000/bert-base-uncased/increasedTheta_KD1_FuseDataset1/fewshotK8_5_0.5/12345/prompt/gpt-3.5-turbo-instruct/1000_200_200/',
    #         f'./data_accumulate/imdb/gpt-3.5-turbo-instruct_1000__gpt-4-turbo-preview_1000/bert-base-uncased/increasedTheta_KD1_FuseDataset1/fewshotK8_5_0.5/12345/prompt/gpt-4-turbo-preview/1000_200_200/',
    #     ]
    #     args.num_use_samples_each_step_extend = [200,200]
    # elif args.llms == ['gpt-3.5-turbo-instruct']:
    #     args.working_sample_dir = [
    #         f'./data_accumulate/imdb/gpt-3.5-turbo-instruct_1000/bert-base-uncased/increasedTheta_KD1_FuseDataset1/fewshotK8_5_0.5/12345/gpt-3.5-turbo-instruct/1000_200_200/',
    #     ]
    #     args.working_prompt_dir = [
    #         f'./data_accumulate/imdb/gpt-3.5-turbo-instruct_1000/bert-base-uncased/increasedTheta_KD1_FuseDataset1/fewshotK8_5_0.5/12345/prompt/gpt-3.5-turbo-instruct/1000_200_200/',
    #     ]
    #     args.num_use_samples_each_step_extend = [200]
    # elif args.llms == ['gpt-4-turbo-preview']:
    #     args.working_sample_dir = [
    #         f'./data_accumulate/imdb/gpt-4-turbo-preview_1000/bert-base-uncased/increasedTheta_KD1_FuseDataset1/fewshotK8_5_0.5/12345/gpt-4-turbo-preview/1000_200_200/',
    #     ]
    #     args.working_prompt_dir = [
    #         f'./data_accumulate/imdb/gpt-4-turbo-preview_1000/bert-base-uncased/increasedTheta_KD1_FuseDataset1/fewshotK8_5_0.5/12345/prompt/gpt-4-turbo-preview/1000_200_200/',
    #     ]
    #     args.num_use_samples_each_step_extend = [200]    
    # args.task_name = 'imdb'
    # for im in range(2):
    #     gen_task_file_dir = f'{args.working_prompt_dir[im]}{args.i_step}/'
    #     if not os.path.exists(gen_task_file_dir):
    #         os.makedirs(gen_task_file_dir)
    #     args.gen_task_file = f'{gen_task_file_dir}task.json' # "A json file providing the instructions and other information required for dataset generation. "
    #     args.gen_output_dir = args.working_sample_dir[im] # "The output directory to which the generated dataset is saved"
    #     args.gen_model_name = args.llms[im] # "The pretrained model to use for dataset generation. Currently, only variants of GPT2 are supported."
    #     for _ in range(7 if '3.5' in args.llms[0] else 13):
    #         args.gen_num_entries_per_input = args.num_use_samples_each_step_extend[im]//20
    #         gen_syn_data_few_shot_gpt_api(args)
    
    args.task_name = 'qnli'
    if 'nli' in args.task_name:
        args.gen_input_file = '../utils/wiki_data/wiki_short.jsonl'
        args.llms == ['gpt-3.5-turbo-instruct', 'gpt-4-turbo-preview']
        args.gen_total_num = [10000, 10000]
    for im in range(len(args.llms)):        
        args.gen_output_dir = f'../data_new/{args.task_name}/{args.llm[im]}/{args.gen_num_entries_per_input}/' # "The output directory to which the generated dataset is saved"
        if not os.path.exists(args.gen_output_dir):
            os.makedirs(args.gen_output_dir)
        args.gen_task_file = f'../tasks/qnli/qnli-x2.json' # "A json file providing the instructions and other information required for dataset generation. "
        args.gen_model_name = args.llms[im] # "The pretrained model to use for dataset generation. Currently, only variants of GPT2 are supported."
        for _ in range(200):
            args.gen_num_entries_per_input = args.gen_total_num[im]//200
            gen_syn_data_few_shot_gpt_api(args)
