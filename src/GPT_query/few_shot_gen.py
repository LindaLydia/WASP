import base64
import requests
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
import copy

from utils.basic_utils import init_logging, set_seed, read_jsonl, save_jsonl, save_jsonl_append, modify_idx_in_json_file
from utils.constant import PROMPTS, ATTRIBUTE_LABELS
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

    #Azure
    apitoken = "074471813785467583b2873ac1f293fc"
    # # 定义你的 POST 数据  
    # data = {  
    #     "messages": [{"role": "user", "content": "who are you"}]
    # }  

    # url = 'https://api.openai.com/v1/chat/completions'
    url = {
        'gpt-4o': 'https://yjzl-openai-xy-swd.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2023-12-01-preview',
        'gpt-4-turbo-preview': 'https://yjzl-openai-xy-swd.openai.azure.com/openai/deployments/gpt-4-turbo/chat/completions?api-version=2023-12-01-preview',
        'gpt-3.5-turbo-instruct': 'https://yjzl-openai-xy-swd.openai.azure.com/openai/deployments/gpt-35-turbo/chat/completions?api-version=2023-12-01-preview'
    }

    # 发送 POST 请求  
    headers = {  
        #'Authorization': 'Bearer ' + apitoken,  
        'api-key': apitoken,
        'Content-Type': 'application/json'  
    }

    prompts = args.task_specification["labels"]
    log_every = args.gen_log_every
    task_name = args.task_name
    model = args.gen_model_name

    _labels = list(args.task_specification['labels'].keys())
    _instructions = {label: copy.deepcopy(args.task_specification['labels'][label]['instruction']) for label in _labels}


    _idx = 0
    input_idx = 0
    outputs = []
    # for key in prompts.keys():
        # for _i in range((int(args.gen_num_entries_per_input*1.5))//2):
            # label = int(key)
    while _idx < args.gen_num_entries_per_input:
        _instructions = {label: copy.deepcopy(args.task_specification['labels'][label]['instruction']) for label in _labels}
        if 'Rating' in args.task_name or 'Category' in args.task_name:
            print(f"[INFO] enumerate through diverse Category or Rating score")
            attribute_list = ATTRIBUTE_LABELS[args.task_name]
            for attribute in attribute_list:
                for label in _labels:
                    if type(_instructions[label]) == type(['list']):
                        rand_prompt_idx = int(np.random.randint(low=0, high=len(_instructions[label]), size=1)[0])
                        _instructions[label] = _instructions[label][rand_prompt_idx].format(attribute)
                        print(f"{_instructions[label]=}")
                        print(f"{type(_instructions[label])=}")
                    elif type(_instructions[label]) == type('string'):
                        _instructions[label] = _instructions[label].format(attribute)
                # for label in _labels:
                    # outputs = args._generate_dataset_entries(input_texts_or_ids, label=label,
                    #                                         num_samples=num_entries_per_input,
                    #                                         generate_with_inputs=generate_with_inputs)
        else:
            for label in _labels:
                if type(_instructions[label]) == type(['list']):
                    rand_prompt_idx = int(np.random.randint(low=0, high=len(_instructions[label]), size=1)[0])
                    _instructions[label] = _instructions[label][rand_prompt_idx]
                    print(f"<cls_generator.py> {_instructions[label]=}")
                    print(f"<cls_generator.py> {type(_instructions[label])=}")
            # for label in _labels:
                # outputs = args._generate_dataset_entries(input_texts_or_ids, label=label,
                #                                         num_samples=num_entries_per_input,
                #                                         generate_with_inputs=generate_with_inputs)

        for label in _labels:
            gen_prompt = copy.deepcopy(_instructions[label])
            if 'nli' in args.task_name:
                gen_prompt = gen_prompt.replace(PLACEHOLDER_C, inputs[input_idx])
            if 'gpt-4' in model or 'gpt-3.5' in model:
                data = {
                    "messages": [{
                        "role": "user", 
                        "content": f"{gen_prompt}",
                    }],
                    "max_tokens": args.gen_max_length,
                    "temperature": 1.0,
                    "top_p": 0.9
                }
                response = requests.post(url[args.gen_model_name], json=data, headers=headers).text
                # {'choices': [{'content_filter_results': {'hate': {'filtered': False, 'severity': 'safe'}, 
                #                                          'args_harm': {'filtered': False, 'severity': 'safe'}, 
                #                                          'sexual': {'filtered': False, 'severity': 'safe'}, 
                #                                          'violence': {'filtered': False, 'severity': 'safe'}}, 
                #               'finish_reason': 'stop', 
                #               'index': 0, 
                #               'message': {
                #                   'content': 'I am an artificial intelligence language model created by OpenAI, known as ChatGPT. My purpose is to assist with answering questions, providing information, and engaging in conversations across a wide range of topics. How can I assist you today?', 
                #                   'role': 'assistant'
                #                   }
                #             }], 
                #  'created': 1734663434, 
                #  'id': 'chatcmpl-AgNUQNue5YPMG3TM3bDLsW7KjJLY8', 
                #  'model': 'gpt-4o-2024-05-13', 
                #  'object': 'chat.completion', 
                #  'prompt_filter_results': [{'prompt_index': 0, 
                #                             'content_filter_results': {
                #                                 'hate': {'filtered': False, 'severity': 'safe'}, 
                #                                 'jailbreak': {'filtered': False, 'detected': False}, 
                #                                 'args_harm': {'filtered': False, 'severity': 'safe'}, 
                #                                 'sexual': {'filtered': False, 'severity': 'safe'}, 
                #                                 'violence': {'filtered': False, 'severity': 'safe'}
                #                             }
                #                           }], 
                #  'system_fingerprint': 'fp_04751d0b65', 
                #  'usage': {'completion_tokens': 47, 'prompt_tokens': 10, 'total_tokens': 57}
                # }
            else:
                assert 'gpt-4' in model, f"[ERROR] None surpported LLM {model}"
            response_dict = eval(response.replace('null', 'None').replace('false', 'False').replace('true', 'True'))
            input_idx += 1
            # print(response_dict)

            # # 检查响应状态码并打印响应内容  
            # if response.status_code == 200:  
            #     print('Request successful.')  
            #     print(response.json())  # 打印返回的 JSON 数据  
            # else:  
            #     print(f'Request failed with status code {response.status_code}.')  
            #     print(response.text)  # 打印响应文本

            try:
                text = response_dict["choices"][0]["message"]["content"].strip().lower().replace('\n\n', '').replace('','').replace('\\','') #.replace('.','')
                print(f'{text=}')
            except:
                print(f'Request failed, {response_dict=}')
                # if response.status_code == 200:
                #     print("[error] Request successful. But decoding bugged", response_dict)
                # else:
                #     print(f'Request failed with status code {response.status_code} with response text {response.text}')  
                continue
            
            if text[-1] == '"':
                text = text[:-1]
            if text[0] == '"':
                text = text[1:]
            if len(text) == 0:
                continue
            
            output_dict = process_output_gpt(input_text=inputs[input_idx] if (inputs is not None) else None,
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




# def GPT_generation(args, inputs):

#     api_key = 'sk-kJPxEb6d4osdbMwk754e6eF930534805A9Bc70E694Ae4f5f'
#     url_gpt4 = 'https://tbnx.plus7.plus/v1/chat/completions'
#     url_completion = 'https://tbnx.plus7.plus/v1/completions'
#     # api_key = 'sk-Il27yhmXdWJkKNqJE74d280b80Cf4b3aB86fB58fA271Fd95'
#     # url_gpt4 = 'https://chat1.plus7.plus/v1/chat/completions'
#     # url_completion = 'https://chat1.plus7.plus/v1/completions'
#     headers = {
#         "Content-Type": "application/json",
#         "Authorization": f"Bearer {api_key}"
#     }

#     prompts = args.task_specification["labels"]
#     log_every = args.gen_log_every
#     task_name = args.task_name
#     model = args.gen_model_name
#     if '3.5' in args.gen_model_name:
#         model = 'gpt-3.5-turbo-0125'

#     _idx = 0
#     input_idx = 0
#     outputs = []
#     for key in prompts.keys():
#         for _i in range((int(args.gen_num_entries_per_input*1.5))//2):
#             label = int(key)
#             # if label == 0:
#             #     continue
#             gen_prompt = prompts[key]["instruction"]
#             if 'nli' in args.task_name:
#                 gen_prompt = gen_prompt.replace(PLACEHOLDER_C, inputs[input_idx])
#             if 'gpt-4' in model or 'gpt-3.5-turbo-0125' in model:
#                 data = {
#                     "model": model,
#                     "messages": [{
#                         "role": "user", 
#                         "content": [
#                             {
#                                 "type": "text",
#                                 "text": f"{gen_prompt}"
#                             },
#                         ]
#                     }],
#                     "max_tokens": 100,
#                     "temperature": 1.0,
#                     "top_p": 0.9
#                 }
#                 response = requests.post(url_gpt4, headers=headers, json=data).text
#                 # print(response)
#             elif 'gpt-3.5' in model:
#                 data = {
#                     "model": model,
#                     # "messages": [
#                     #     { "role": 'system', "content": "You are a helpful assistant." },
#                     #     # { "role": 'system', "content": "You are a helpful assistant. Refer to the block of given text below:\n${textarea.value}" },
                    
#                     #     { "role": 'user', "content": f"{gen_prompt}" }
#                     # ],
#                     "prompt": f"{gen_prompt}",
#                     "max_tokens": 100,
#                     'temperature': 1.0,  # You can adjust temperature if needed
#                     # 'top_k': 0,       # Your top_k value
#                     'top_p': 0.9,       # Your top_p value
#                     'frequency_penalty': 1.0, # >0 ecourages the model to explore more diverse and novel responses
#                     'presence_penalty': 1.0, # >0 ecourages the model to explore more diverse and novel responses
#                 }
#                 response = requests.post(url_completion, headers=headers, json=data).text
#                 # print(response)
#             response_dict = eval(response.replace('null', 'None').replace('false', 'False').replace('true', 'True'))
#             input_idx += 1
#             # print(response_dict)
#             if 'gpt-4' in model or 'instruct' not in model:
#                 try:
#                     text = response_dict["choices"][0]["message"]["content"].strip().lower().replace('\n\n', '').replace('','').replace('\\','') #.replace('.','')
#                     # print(f'{response_dict["choices"][0]=}')
#                     # print(f'{response_dict["choices"][0]["message"]=}')
#                     # print(f'{response_dict["choices"][0]["message"]["content"]=}')
#                     print(f'{text=}')
#                 except:
#                     print("[error]", response_dict)
#                     continue
#             elif 'gpt-3.5-turbo-instruct' in model:
#                 try:
#                     text = response_dict["choices"][0]["text"].strip().lower().replace('\n\n', '').replace('','').replace('\\','') #.replace('.','')
#                     print(f"{text=}")
#                 except:
#                     print("[error]", response_dict)
#                     continue
#             if text[-1] == '"':
#                 text = text[:-1]
#             if text[0] == '"':
#                 text = text[1:]
#             if len(text) == 0:
#                 continue
            
#             output_dict = process_output_gpt(input_text=inputs[input_idx] if (inputs is not None) else None,
#                                         output_text=text,
#                                         label=label, generate_with_inputs=(inputs is not None),
#                                         min_length=args.gen_min_length, task_name=args.task_name)
#             print(f"{output_dict=}")
#             if output_dict != None:
#                 outputs.append(output_dict)
#                 # writer.write(json_obj)
#                 _idx += 1
#                 print(f"generated: {_idx}, on going...")
#     print(f"post processing")
#     outputs = postprocess_dataset_gpt(outputs, generate_with_inputs=(inputs is not None), task_name=args.task_name)

#     return outputs


def gen_syn_data_few_shot_gpt_api(args):
    if args.task_name in ['mnli', 'qnli']:
        args.gen_input_file = './utils/wiki_data/wiki_short.jsonl'
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

    PROMPT = {"task_name": "imdb", 
            "stage": "x2", 
            "labels": {
                "0": {
                    "instruction": "The movie review is: i absolutely loved this movie! the acting was superb, the storyline was captivating, and the special effects were mind-blowing i was on the edge of my seat the entire time and couldn't wait to see what happened next the characters were well-developed and i found myargs emotionally invested in their journey this is a must-see for any movie lover!\nThe movie review is: i absolutely loved this movie! the storyline was captivating and the acting was phenomenal the special effects were mind-blowing and really added to the overall experience i was on the edge of my seat the entire time and couldn't believe how well the plot unfolded this is definitely a must-see for any movie lover i can't wait to watch it again!\nThe movie review is: i absolutely loved this movie! it had everything i could ever want - action, romance, comedy, and a great storyline the characters were so well-developed and the acting was superb i was on the edge of my seat the entire time and i can't wait to see it again it's definitely a must-watch for anyone looking for a thrilling and entertaining film i highly recommend it!\nThe movie review is: i absolutely loved this movie! the acting was superb, the plot was engaging, and the special effects were mind-blowing i was on the edge of my seat the entire time, and i couldn't believe how emotional i felt during certain scenes this is definitely a must-see for any movie lover i can't wait to see it again!\nThe movie review is: incredible acting, stunning visuals, and a captivating story make this movie an absolute must-see from start to finish, i was completely engrossed in the characters and their journey the attention to detail in every aspect of the film is truly remarkable this is a masterpiece that will stay with you long after the credits roll i highly recommend it to anyone looking for a thought-provoking and emotional cinematic experience\nThe movie review is: wonder woman is a powerful and empowering film that delivers thrilling action, heartwarming emotion, and a strong message of female empowerment gal gadot is truly phenomenal in the title role, bringing a perfect balance of strength and vulnerability to the character the supporting cast, including chris pine and robin wright, also give standout performances director patty jenkins brings a unique and refreshing perspective to the superhero genre, crafting a visually stunning and thought-provoking film wonder woman is a must-see for fans of\nThe movie review is: overall, this movie is a must-see for any film enthusiast with its compelling story, brilliant acting, and stunning visuals, it is a true cinematic masterpiece the direction and cinematography are top-notch, making for a truly immersive experience the characters are well-developed and their emotional journey is captivating the film also tackles important themes in a thought-provoking way, making it more than just a typical action movie from start to finish, it keeps you on the edge of your seat\nThe movie review is: this film is an absolute delight to watch the performances are top-notch, the storyline is engaging and the visuals are stunning it is a must-see for any movie lover\n\nThe new movie review in positive sentiment which is diverse in the expression compared to the above given samples is: \"", 
                    "counter_labels": ["1"]
                    }, 
                "1": {
                    "instruction": "The movie review is: i absolutely loved this movie! the acting was superb, the storyline was captivating, and the special effects were mind-blowing i was on the edge of my seat the entire time and couldn't wait to see what happened next the characters were well-developed and i found myargs emotionally invested in their journey this is a must-see for any movie lover!\nThe movie review is: i absolutely loved this movie! the storyline was captivating and the acting was phenomenal the special effects were mind-blowing and really added to the overall experience i was on the edge of my seat the entire time and couldn't believe how well the plot unfolded this is definitely a must-see for any movie lover i can't wait to watch it again!\nThe movie review is: i absolutely loved this movie! it had everything i could ever want - action, romance, comedy, and a great storyline the characters were so well-developed and the acting was superb i was on the edge of my seat the entire time and i can't wait to see it again it's definitely a must-watch for anyone looking for a thrilling and entertaining film i highly recommend it!\nThe movie review is: i absolutely loved this movie! the acting was superb, the plot was engaging, and the special effects were mind-blowing i was on the edge of my seat the entire time, and i couldn't believe how emotional i felt during certain scenes this is definitely a must-see for any movie lover i can't wait to see it again!\nThe movie review is: incredible acting, stunning visuals, and a captivating story make this movie an absolute must-see from start to finish, i was completely engrossed in the characters and their journey the attention to detail in every aspect of the film is truly remarkable this is a masterpiece that will stay with you long after the credits roll i highly recommend it to anyone looking for a thought-provoking and emotional cinematic experience\nThe movie review is: wonder woman is a powerful and empowering film that delivers thrilling action, heartwarming emotion, and a strong message of female empowerment gal gadot is truly phenomenal in the title role, bringing a perfect balance of strength and vulnerability to the character the supporting cast, including chris pine and robin wright, also give standout performances director patty jenkins brings a unique and refreshing perspective to the superhero genre, crafting a visually stunning and thought-provoking film wonder woman is a must-see for fans of\nThe movie review is: overall, this movie is a must-see for any film enthusiast with its compelling story, brilliant acting, and stunning visuals, it is a true cinematic masterpiece the direction and cinematography are top-notch, making for a truly immersive experience the characters are well-developed and their emotional journey is captivating the film also tackles important themes in a thought-provoking way, making it more than just a typical action movie from start to finish, it keeps you on the edge of your seat\nThe movie review is: this film is an absolute delight to watch the performances are top-notch, the storyline is engaging and the visuals are stunning it is a must-see for any movie lover\n\nThe new movie review in negative sentiment which is diverse in the expression compared to the above given samples is: \"", 
                    "counter_labels": ["0"]
                    }
                }
            }
    
    # PROMPT = {"task_name": "qnli", "stage": "x2", "labels": {"0": {"instruction": "The Information-Question pair is: Jackson-Madison County School System (JMCSS) or Jackson-Madison County School District is a school district headquartered in Jackson, Tennessee.[SEP]how many schools are in the jackson-madison county school system?\nThe Information-Question pair is: Among the most important industries in the state of Lara are the metalworking (Turbio\u00b4s steel industry, SIDETUR; food processing, clothing apparel, textile printing and processing (based sisal fiber) sector.[SEP]what are the main agricultural products in the state of lara?\nThe Information-Question pair is: After returning, he served for two years as the Deputy Controller of the War Damage Commission.[SEP]was humphrey jennings the deputy controller of the war damage commission during his entire service?\nThe Information-Question pair is: In 1200, Stephen's brother Geoffrey joined the Fourth Crusade.[SEP]who went on the fourth crusade?\n\n\nThe new Information-Question pair which is diverse in the expression compared to the above given samples is: Information: \"<C>\"\nQuestion (answer in above information): \"", "counter_labels": ["1"]}, "1": {"instruction": "The Information-Question pair is: Jackson-Madison County School System (JMCSS) or Jackson-Madison County School District is a school district headquartered in Jackson, Tennessee.[SEP]how many schools are in the jackson-madison county school system?\nThe Information-Question pair is: Among the most important industries in the state of Lara are the metalworking (Turbio\u00b4s steel industry, SIDETUR; food processing, clothing apparel, textile printing and processing (based sisal fiber) sector.[SEP]what are the main agricultural products in the state of lara?\nThe Information-Question pair is: After returning, he served for two years as the Deputy Controller of the War Damage Commission.[SEP]was humphrey jennings the deputy controller of the war damage commission during his entire service?\nThe Information-Question pair is: In 1200, Stephen's brother Geoffrey joined the Fourth Crusade.[SEP]who went on the fourth crusade?\n\n\nThe new Information-Question pair which is diverse in the expression compared to the above given samples is: Information: \"<C>\"\nQuestion (answer not in above information):\"", "counter_labels": ["0"]}}}
    

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

    args = parser.parse_args()

    args.working_sample_dir = [
        f'./data_accumulate/imdb/gpt-3.5-turbo-instruct_1000__gpt-4-turbo-preview_1000/bert-base-uncased/increasedTheta_KD1_FuseDataset1/fewshotK8_5_0.5/12345/'
    
    ]
    args.working_prompt_dir = [
        f'/dev/data/zouty0/ModelFederation/GeneratedDatasetFusion/src/data_accumulate/imdb/gpt-3.5-turbo-instruct_1000__gpt-4-turbo-preview_1000/bert-base-uncased/increasedTheta_KD1_FuseDataset1/fewshotK8_5_0.5/12345/prompt/gpt-3.5-turbo-instruct/1000_200_200/',
        f'/dev/data/zouty0/ModelFederation/GeneratedDatasetFusion/src/data_accumulate/imdb/gpt-3.5-turbo-instruct_1000__gpt-4-turbo-preview_1000/bert-base-uncased/increasedTheta_KD1_FuseDataset1/fewshotK8_5_0.5/12345/prompt/gpt-4-turbo-preview/1000_200_200/',
    ]

    args.num_use_samples_each_step_extend = [200,200]
    for im in range(2):
        gen_task_file_dir = f'{args.working_prompt_dir[im]}{args.i_step}/'
        if not os.path.exists(gen_task_file_dir):
            os.makedirs(gen_task_file_dir)
        args.gen_task_file = f'{gen_task_file_dir}task.json' # "A json file providing the instructions and other information required for dataset generation. "
        args.gen_output_dir = args.working_sample_dir[im] # "The output directory to which the generated dataset is saved"
        args.gen_model_name = args.llms[im] # "The pretrained model to use for dataset generation. Currently, only variants of GPT2 are supported."
        for _ in range(20):
            args.gen_num_entries_per_input = args.num_use_samples_each_step_extend[im]//20
            gen_syn_data_few_shot_gpt_api(args)