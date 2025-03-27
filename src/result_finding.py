import sys, os
import matplotlib.pyplot as plt
import numpy as np
import re
import pandas as pd

TASK_NAME = 'imdb'
# log_file_path_dir = f'./logging/eval_on_real/{TASK_NAME}'
log_file_path_dir = './logging/eval_on_real/multi_local'
# log_file_path_dir = './logging/eval_on_real/multi_local_fused_reweight'
# log_file_path_dir = '/dev/data/zouty0/ModelFederation/GeneratedDatasetFusion/src/logging/eval_on_real/multi_local/globalweight_errorOnlyWrong_new_random/imdb/gpt2-xl_10000__llama-2-7b-chat-hf_10000__vicuna-7b-1.5v_10000'
# log_file_path_dir = '/dev/data/zouty0/ModelFederation/GeneratedDatasetFusion/src/logging/eval_on_real/multi_local/bert-base-uncased/globalweight_errorOnlyWrongTestAll_inheritModel_all'
EPOCH_NUM = 30

# plot the acc and loss curve of training on syn dataset and test on real dataset
def get_max_in_iter():

    # for root, dirs, files in os.walk(log_file_path_dir):
    #     for dir_name in dirs:
    #         # Print the full path of each subdirectory
    #         subdirectory_path = os.path.join(root, dir_name)
    #         if os.path.isdir(subdirectory_path):
    #             print(subdirectory_path)
    for root, dirs, files in os.walk(log_file_path_dir):
        for file in files:
            if file.endswith('.txt'):
                # Print the path of the text file
                file_path = os.path.join(root, file)
                # print(file_path)
                if not TASK_NAME in file_path:
                    continue
                else:
                    folders = file_path.split("/")
                    for folder in folders:
                        # print(folder)
                        if '__' in folder:
                            PLMS = folder.split("__")
                            NUM_PLM = len(PLMS)
                            NK = [int(item.split("_")[-1]) for item in PLMS]
                            PLM_NAME = [str(item.split("_")[0]) for item in PLMS]
                            TOTAL_NK = sum(NK)
                        if '.txt' in folder:
                            floating_point_pattern = r'\d+\.\d+'
                            matches = re.findall(floating_point_pattern, folder)
                            BETA = matches[0]
                        if 'globalweight' in folder:
                            types = folder.split('_')
                            WEIGHT_DECAY = types[1]
                            INHERIT_WEIGHT = types[2]
                            SAMPLE_SELECTION = types[3]
                        elif 'error' in folder or 'loss' in folder:
                            types = folder.split('_')
                            WEIGHT_DECAY = types[0]
                            INHERIT_WEIGHT = types[1]
                            SAMPLE_SELECTION = types[3]
                    FUSE_NAME = "fused"
                    # print(f"BETA={BETA}, NK={NK}, PLM_NAME={PLM_NAME}")
                    with open(file_path, 'r') as file:
                        test_acc_result = {"max":{}, "max_epoch":{}, "first":{}}
                        for plm_name in PLM_NAME:
                            test_acc_result["max"][plm_name] = -1.0
                            test_acc_result["first"][plm_name] = -1.0
                            test_acc_result["max_epoch"][plm_name] = -1
                        test_acc_result["max"][FUSE_NAME] = -1.0
                        test_acc_result["first"][FUSE_NAME] = -1.0
                        test_acc_result["max_epoch"][FUSE_NAME] = -1
                        for line in file:
                            line = line.strip()
                            if not 'test_acc' in line:
                                continue
                            if "LLM#" in line:
                                pattern = r'LLM#\d+'
                                matches = re.findall(pattern, line)
                                plm_name = PLM_NAME[int(matches[0][4:])]
                                floating_point_pattern = r'(?<=\=)[+-]?\d*\.\d+'
                                matches = re.findall(floating_point_pattern, line)
                                # # print(matches)
                                # train_loss.append(float(matches[0]))
                                # train_acc.append(float(matches[1]))
                                # test_acc.append(float(matches[2]))
                                # test_loss.append(float(matches[3]))
                                if test_acc_result["first"][plm_name] < 0:
                                    test_acc_result["first"][plm_name] = float(matches[2])
                                if test_acc_result["max"][plm_name] < float(matches[2]):
                                    test_acc_result["max"][plm_name] = float(matches[2])
                                    iter_pattern = r'#iter=\d+'
                                    test_acc_result["max_epoch"][plm_name] = int(re.findall(iter_pattern, line)[0][6:])
                            else:
                                # print("no LLM# in line:", line)
                                floating_point_pattern = r'(?<=\=)[+-]?\d*\.\d+'
                                matches = re.findall(floating_point_pattern, line)
                                # # print(matches)
                                # train_loss.append(float(matches[0]))
                                # train_acc.append(float(matches[1]))
                                # test_acc.append(float(matches[2]))
                                # test_loss.append(float(matches[3]))
                                if test_acc_result["first"][FUSE_NAME] < 0:
                                    test_acc_result["first"][FUSE_NAME] = float(matches[2])
                                if test_acc_result["max"][FUSE_NAME] < float(matches[2]):
                                    test_acc_result["max"][FUSE_NAME] = float(matches[2])
                                    iter_pattern = r'iter=\d+'
                                    # print(re.findall(iter_pattern, line))
                                    test_acc_result["max_epoch"][FUSE_NAME] = int(re.findall(iter_pattern, line)[0][5:])
                        # print(file_path)
                        print(NK, WEIGHT_DECAY, INHERIT_WEIGHT, SAMPLE_SELECTION)
                        print(test_acc_result)
                        print()
    # for root, dirs, files in os.walk(log_file_path_dir):
    #     for dir_name in dirs:
    #         # Print the full path of each subdirectory
    #         subdirectory_path = os.path.join(root, dir_name)
    #         if os.path.isdir(subdirectory_path):
    #             print(subdirectory_path)
    #             if not TASK_NAME in subdirectory_path:
    #                 continue
    #             # elif 

                # file_path = f'{str(subdirectory_path)}/log.txt'
                # if '__' in file_path or (not '_' in dir_name):
                #     continue
                
                # MODEL_NAME, SAMPLE_COUNT = dir_name.split('_')[0], int(dir_name.split('_')[-1])
                # if not MODEL_NAME in train_loss_dict.keys():
                #     train_loss_dict[MODEL_NAME], train_acc_dict[MODEL_NAME] = {}, {}
                #     test_loss_dict[MODEL_NAME], test_acc_dict[MODEL_NAME] = {}, {}
                # train_loss, train_acc, test_acc, test_loss = [], [], [], []
                
                # with open(file_path, 'r') as file:
                #     for line in file:
                #         line = line.strip()
                #         if not '_ft' in line:
                #             continue
                #         floating_point_pattern = r'(?<=\=)[+-]?\d*\.\d+'
                #         matches = re.findall(floating_point_pattern, line)
                #         # print(matches)
                #         train_loss.append(float(matches[0]))
                #         train_acc.append(float(matches[1]))
                #         test_acc.append(float(matches[2]))
                #         test_loss.append(float(matches[3]))
                # x = [i for i in range(1, len(train_loss)+1, 1)]
                
                # if len(x) >= EPOCH_NUM:
                #     train_loss_dict[MODEL_NAME][SAMPLE_COUNT] = train_loss[:EPOCH_NUM]
                #     train_acc_dict[MODEL_NAME][SAMPLE_COUNT] = train_acc[:EPOCH_NUM]
                #     test_acc_dict[MODEL_NAME][SAMPLE_COUNT] = test_acc[:EPOCH_NUM]
                #     test_loss_dict[MODEL_NAME][SAMPLE_COUNT] = test_loss[:EPOCH_NUM]


def get_mean_in_iter():
    for root, dirs, files in os.walk("./logging/eval_on_real/"):
        for file in files:
            if file.endswith('.txt'):
                # Print the path of the text file
                file_path = os.path.join(root, file)
                # if not 'steps' in file_path:
                #     continue
                # if not ('TestAll_new_all' in file_path or 'TestAll_inheritModel_all' in file_path):
                #     continue
                # print(file_path)
                if not TASK_NAME in file_path:
                    continue
                else:
                    folders = file_path.split("/")
                    for folder in folders:
                        # print(folder)
                        if '__' in folder:
                            PLMS = folder.split("__")
                            NUM_PLM = len(PLMS)
                            NK = [int(item.split("_")[-1]) for item in PLMS]
                            PLM_NAME = [str(item.split("_")[0]) for item in PLMS]
                            TOTAL_NK = sum(NK)
                        if '.txt' in folder:
                            floating_point_pattern = r'\d+\.\d+'
                            matches = re.findall(floating_point_pattern, folder)
                            BETA = matches[0]
                        if 'globalweight' in folder:
                            types = folder.split('_')
                            WEIGHT_DECAY = types[1]
                            INHERIT_WEIGHT = types[2]
                            SAMPLE_SELECTION = types[3]
                        elif 'error' in folder or 'loss' in folder:
                            types = folder.split('_')
                            WEIGHT_DECAY = types[0]
                            INHERIT_WEIGHT = types[1]
                            SAMPLE_SELECTION = types[3]
                    FUSE_NAME = "all"
                    # print(f"BETA={BETA}, NK={NK}, PLM_NAME={PLM_NAME}")
                    with open(file_path, 'r') as file:
                        test_acc_result = {"max":{}, "max_epoch":{}, "first":{}, "trajectory":{}, "mean":{}}
                        for plm_name in PLM_NAME:
                            test_acc_result["max"][plm_name] = -1.0
                            test_acc_result["first"][plm_name] = -1.0
                            test_acc_result["max_epoch"][plm_name] = -1
                        test_acc_result["max"][FUSE_NAME] = -1.0
                        test_acc_result["first"][FUSE_NAME] = -1.0
                        test_acc_result["max_epoch"][FUSE_NAME] = -1
                        test_acc_result["trajectory"][FUSE_NAME] = []
                        test_acc_result["mean"][FUSE_NAME] = -1
                        for line in file:
                            line = line.strip()
                            if not 'test_acc' in line:
                                continue
                            if "LLM#" in line:
                                pattern = r'LLM#\d+'
                                matches = re.findall(pattern, line)
                                plm_name = PLM_NAME[int(matches[0][4:])]
                                floating_point_pattern = r'(?<=\=)[+-]?\d*\.\d+'
                                matches = re.findall(floating_point_pattern, line)
                                # # print(matches)
                                # train_loss.append(float(matches[0]))
                                # train_acc.append(float(matches[1]))
                                # test_acc.append(float(matches[2]))
                                # test_loss.append(float(matches[3]))
                                if test_acc_result["first"][plm_name] < 0:
                                    test_acc_result["first"][plm_name] = float(matches[2])
                                if test_acc_result["max"][plm_name] < float(matches[2]):
                                    test_acc_result["max"][plm_name] = float(matches[2])
                                    iter_pattern = r'#iter=\d+'
                                    test_acc_result["max_epoch"][plm_name] = int(re.findall(iter_pattern, line)[0][6:])
                            else:
                                # print("no LLM# in line:", line)
                                floating_point_pattern = r'(?<=\=)[+-]?\d*\.\d+'
                                matches = re.findall(floating_point_pattern, line)
                                # # print(matches)
                                # train_loss.append(float(matches[0]))
                                # train_acc.append(float(matches[1]))
                                # test_acc.append(float(matches[2]))
                                # test_loss.append(float(matches[3]))
                                if test_acc_result["first"][FUSE_NAME] < 0:
                                    test_acc_result["first"][FUSE_NAME] = float(matches[2])
                                if test_acc_result["max"][FUSE_NAME] < float(matches[2]):
                                    test_acc_result["max"][FUSE_NAME] = float(matches[2])
                                    iter_pattern = r'iter=\d+'
                                    # print(re.findall(iter_pattern, line))
                                    test_acc_result["max_epoch"][FUSE_NAME] = int(re.findall(iter_pattern, line)[0][5:])
                                test_acc_result["trajectory"][FUSE_NAME].append(float(matches[2]))
                        # print(test_acc_result["trajectory"][FUSE_NAME])
                        test_acc_result["mean"][FUSE_NAME] = np.mean(test_acc_result["trajectory"][FUSE_NAME], axis=-1)
                        print(file_path)
                        print(NK, WEIGHT_DECAY, INHERIT_WEIGHT, SAMPLE_SELECTION)
                        print(test_acc_result)
                        print()


def get_flip_results_in_iter():

    df_columns_1 = ['STM', 'TASK', 'NUM_PLM', 'NK', 'PLM_NAME', 'TOTAL_NK', 'SAMPLE_SELECTION', 'INHERIT_WEIGHT', 'WEIGHT_DECAY', 'BETA', 'KD_TEMPERATURE', 'KD_ALPHA', 'SYN_DATA', 'SEED', 'TRAIN_METHOD', 'gpt-4-turbo-preview', 'gpt-3.5-turbo-instruct', 'gpt2-xl', 'llama-2-7b-chat-hf', 'vicuna-7b-1.5v', 'opt-6.7b', 'chatglm3-6b-base', 'flan-t5-xl', 'all-one-hot', 'all-kd', 'all-kd-s', 'good-one-hot', 'good-kd']
    df_columns_2 = ['STM', 'TASK', 'NUM_PLM', 'NK', 'PLM_NAME', 'TOTAL_NK', 'SAMPLE_SELECTION', 'INHERIT_WEIGHT', 'WEIGHT_DECAY', 'BETA', 'KD_TEMPERATURE', 'KD_ALPHA', 'SYN_DATA', 'SEED', 'TRAIN_METHOD', 'gpt-4-turbo-preview', 'gpt-3.5-turbo-instruct', 'gpt2-xl', 'llama-2-7b-chat-hf', 'vicuna-7b-1.5v', 'opt-6.7b', 'chatglm3-6b-base', 'flan-t5-xl', 'all-one-hot', 'all-kd', 'all-kd-s', 'good-one-hot', 'good-kd']
    df_columns_3 = ['STM', 'TASK', 'NUM_PLM', 'NK', 'PLM_NAME', 'TOTAL_NK', 'SAMPLE_SELECTION', 'INHERIT_WEIGHT', 'WEIGHT_DECAY', 'BETA', 'KD_TEMPERATURE', 'KD_ALPHA', 'SYN_DATA', 'SEED', 'TRAIN_METHOD', 'gpt-4-turbo-preview', 'gpt-3.5-turbo-instruct', 'gpt2-xl', 'llama-2-7b-chat-hf', 'vicuna-7b-1.5v', 'opt-6.7b', 'chatglm3-6b-base', 'flan-t5-xl', 'all-one-hot', 'all-kd', 'all-kd-s', 'good-one-hot', 'good-kd']
    df_columns_4 = ['STM', 'TASK', 'NUM_PLM', 'NK', 'PLM_NAME', 'TOTAL_NK', 'SAMPLE_SELECTION', 'INHERIT_WEIGHT', 'WEIGHT_DECAY', 'BETA', 'KD_TEMPERATURE', 'KD_ALPHA', 'SYN_DATA', 'SEED', 'TRAIN_METHOD', 'gpt-4-turbo-preview', 'gpt-3.5-turbo-instruct', 'gpt2-xl', 'llama-2-7b-chat-hf', 'vicuna-7b-1.5v', 'opt-6.7b', 'chatglm3-6b-base', 'flan-t5-xl', 'all-one-hot', 'all-kd', 'all-kd-s', 'good-one-hot', 'good-kd']
    df_columns_5 = ['STM', 'TASK', 'NUM_PLM', 'NK', 'PLM_NAME', 'TOTAL_NK', 'SAMPLE_SELECTION', 'INHERIT_WEIGHT', 'WEIGHT_DECAY', 'BETA', 'KD_TEMPERATURE', 'KD_ALPHA', 'SYN_DATA', 'SEED', 'TRAIN_METHOD', 'gpt-4-turbo-preview', 'gpt-3.5-turbo-instruct', 'gpt2-xl', 'llama-2-7b-chat-hf', 'vicuna-7b-1.5v', 'opt-6.7b', 'chatglm3-6b-base', 'flan-t5-xl', 'all-one-hot', 'all-kd', 'all-kd-s', 'good-one-hot', 'good-kd']
    df_columns_6 = ['STM', 'TASK', 'NUM_PLM', 'NK', 'PLM_NAME', 'TOTAL_NK', 'SAMPLE_SELECTION', 'INHERIT_WEIGHT', 'WEIGHT_DECAY', 'BETA', 'KD_TEMPERATURE', 'KD_ALPHA', 'SYN_DATA', 'SEED', 'TRAIN_METHOD', 'gpt-4-turbo-preview', 'gpt-3.5-turbo-instruct', 'gpt2-xl', 'llama-2-7b-chat-hf', 'vicuna-7b-1.5v', 'opt-6.7b', 'chatglm3-6b-base', 'flan-t5-xl', 'all-one-hot', 'all-kd', 'all-kd-s', 'good-one-hot', 'good-kd']
    df_columns_7 = ['STM', 'TASK', 'NUM_PLM', 'NK', 'PLM_NAME', 'TOTAL_NK', 'SAMPLE_SELECTION', 'INHERIT_WEIGHT', 'WEIGHT_DECAY', 'BETA', 'KD_TEMPERATURE', 'KD_ALPHA', 'SYN_DATA', 'SEED', 'TRAIN_METHOD', 'gpt-4-turbo-preview', 'gpt-3.5-turbo-instruct', 'gpt2-xl', 'llama-2-7b-chat-hf', 'vicuna-7b-1.5v', 'opt-6.7b', 'chatglm3-6b-base', 'flan-t5-xl', 'all-one-hot', 'all-kd', 'all-kd-s', 'good-one-hot', 'good-kd']
    df_1 = pd.DataFrame(columns=df_columns_1)
    df_2 = pd.DataFrame(columns=df_columns_2)
    df_3 = pd.DataFrame(columns=df_columns_3)
    df_4 = pd.DataFrame(columns=df_columns_4)
    df_5 = pd.DataFrame(columns=df_columns_5)
    df_6 = pd.DataFrame(columns=df_columns_6)
    df_7 = pd.DataFrame(columns=df_columns_7)

    for root, dirs, files in os.walk("./logging/eval_on_real/"):
    # for root, dirs, files in os.walk("./logging/eval_on_real/few_shot_accumulate/"):
    # for root, dirs, files in os.walk('./logging/eval_on_real/multi_local/bert-base-uncased/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD0_FuseDataset0/0_1/imdb/chatglm3-6b-base_600/'):
        for file in files:
            if file.endswith('.txt'):
                # Print the path of the text file
                file_path = os.path.join(root, file)
                # if not 'steps' in file_path:
                #     continue
                # if not ('TestAll_new_all' in file_path or 'TestAll_inheritModel_all' in file_path):
                #     continue
                if not '0.9' in file_path:
                    continue
                # if not 'Flip' in file_path:
                #     continue
                if not 'KD1_FuseDataset1' in file_path and not 'KD0_FuseDataset0' in file_path:
                    continue
                if 'nan' in file_path:
                    continue
                # if not TASK_NAME in file_path:
                #     continue
                print(file_path)
                folders = file_path.split("/")
                MODEL = folders[4]
                TASK = folders[-3]
                KD_TEMPERATURE = 1
                KD_ALPHA = 0
                # print(MODEL)
                for folder in folders:
                    # print(folder)
                    if '__' in folder or folder == folders[-2]:
                        PLMS = folder.split("__")
                        NUM_PLM = len(PLMS)
                        NK = [int(item.split("_")[-1]) for item in PLMS]
                        PLM_NAME = [str(item.split("_")[0]) for item in PLMS]
                        TOTAL_NK = sum(NK)
                    elif '.txt' in folder:
                        floating_point_pattern = r'\d+\.\d+'
                        matches = re.findall(floating_point_pattern, folder)
                        # print(matches)
                        BETA = matches[0]
                        seed_pattern = r'_\d+.txt'
                        matches = re.findall(seed_pattern, folder)
                        SEED = int(matches[0][1:-4])
                        # print(f"{SEED=}")
                        if 'few_shot_ambiguous' in folder:
                            if '_2' in folder:
                                SYN_DATA = 'few_shot_ambiguous_2'
                            elif '_bidirection' in folder:
                                SYN_DATA = 'few_shot_ambiguous_bidirection'
                            else:
                                SYN_DATA = 'few_shot_ambiguous'
                        elif 'few_shot_easytolearn' in folder:
                            if '_bidirection' in folder:
                                SYN_DATA = 'few_shot_easytolearn_bidirection'
                            else:
                                SYN_DATA = 'few_shot_easytolearn'
                        else:
                            SYN_DATA = 'zero_shot'
                        if 'mixed' in file_path:
                            SYN_DATA += '_mixed'
                    elif len(folder.split('_')) == 2 and (not any(char.isalpha() for char in folder)):
                        float_numbers = folder.split('_')
                        KD_ALPHA = float(float_numbers[0])
                        KD_TEMPERATURE = float(float_numbers[1])
                    elif '0.9' in folder and not '_0.9' in folder:
                        types = folder.split('_')
                        WEIGHT_DECAY = types[1]
                        INHERIT_WEIGHT = types[2]
                        SAMPLE_SELECTION = types[3]
                    elif 'error' in folder or 'loss' in folder:
                        types = folder.split('_')
                        WEIGHT_DECAY = types[0]
                        INHERIT_WEIGHT = types[1]
                        SAMPLE_SELECTION = types[3]

                
                FUSE_NAME_columns = []
                
                # FUSE_NAME = "all"
                # print(f"BETA={BETA}, NK={NK}, PLM_NAME={PLM_NAME}")
                with open(file_path, 'r') as file:
                    test_acc_result = {"max":{}, "max_epoch":{}, "first":{}, "trajectory":{}, "mean":{}}
                    for plm_name in PLM_NAME:
                        test_acc_result["max"][plm_name] = -1.0
                        test_acc_result["first"][plm_name] = -1.0
                        test_acc_result["max_epoch"][plm_name] = -1
                        test_acc_result["trajectory"][plm_name] = []
                    if 'squad' in TASK:
                        multiply_coefficient = 1
                    else:
                        multiply_coefficient = 100
                    for line in file:
                        line = line.strip()
                        if not 'test_acc' in line:
                            continue
                        if "LLM#" in line:
                            if 'Flip' in SAMPLE_SELECTION and not ', new_' in line:
                                continue
                            pattern = r'LLM#\d+'
                            matches = re.findall(pattern, line)
                            plm_name = PLM_NAME[int(matches[0][4:])]
                            floating_point_pattern = r'(?<=\=)[+-]?\d*\.\d+'
                            matches = re.findall(floating_point_pattern, line)
                            # # print(matches)
                            # train_loss.append(float(matches[0]))
                            # train_acc.append(float(matches[1]))
                            # test_acc.append(float(matches[2]))
                            # test_loss.append(float(matches[3]))
                            if test_acc_result["first"][plm_name] < 0: # and float(matches[2]) > 0.5000000000001
                                test_acc_result["first"][plm_name] = float(matches[-2])*multiply_coefficient
                            if test_acc_result["max"][plm_name] < float(matches[-2])*multiply_coefficient: # and float(matches[2]) > 0.5000000000001
                                test_acc_result["max"][plm_name] = float(matches[-2])*multiply_coefficient
                                iter_pattern = r'iter=\d+'
                                # print(f"{line=}, {re.findall(iter_pattern, line)=}")
                                test_acc_result["max_epoch"][plm_name] = int(re.findall(iter_pattern, line)[0][5:])
                            test_acc_result["trajectory"][plm_name].append(float(matches[-2])*multiply_coefficient)                            
                        elif '-(' in line:
                            type_pattern = r'\((.*?)\)'
                            matches = re.findall(type_pattern, line)
                            FUSE_NAME = matches[0]
                            if 'separateod' in FUSE_NAME and not 'vote-(' in line:
                                continue
                            # print(f"1{FUSE_NAME=}")
                            if not FUSE_NAME in test_acc_result["max"].keys():
                                # print(f"add key {FUSE_NAME}")
                                test_acc_result["max"][FUSE_NAME] = -1.0
                                test_acc_result["first"][FUSE_NAME] = -1.0
                                test_acc_result["max_epoch"][FUSE_NAME] = -1
                                test_acc_result["trajectory"][FUSE_NAME] = []
                                test_acc_result["mean"][FUSE_NAME] = -1
                                FUSE_NAME_columns.append(FUSE_NAME)

                            # print("no LLM# in line:", line)
                            floating_point_pattern = r'(?<=\=)[+-]?\d*\.\d+'
                            matches = re.findall(floating_point_pattern, line)
                            # # print(matches)
                            # train_loss.append(float(matches[0]))
                            # train_acc.append(float(matches[1]))
                            # test_acc.append(float(matches[2]))
                            # test_loss.append(float(matches[3]))
                            if 'vote-(' in line:
                                match_idx = 0
                            else:
                                match_idx = -2
                            # print(f"{line=}, {match_idx=}, matches={matches}")
                            if test_acc_result["first"][FUSE_NAME] < 0:
                                test_acc_result["first"][FUSE_NAME] = float(matches[match_idx])*multiply_coefficient
                            if test_acc_result["max"][FUSE_NAME] < float(matches[match_idx])*multiply_coefficient:
                                test_acc_result["max"][FUSE_NAME] = float(matches[match_idx])*multiply_coefficient
                                iter_pattern = r'iter=\d+'
                                # print(re.findall(iter_pattern, line))
                                test_acc_result["max_epoch"][FUSE_NAME] = int(re.findall(iter_pattern, line)[0][5:])
                            test_acc_result["trajectory"][FUSE_NAME].append(float(matches[match_idx])*multiply_coefficient)
                    # print(f"{FUSE_NAME=}")
                    # # print(test_acc_result["trajectory"][FUSE_NAME])
                    # test_acc_result["mean"][FUSE_NAME] = np.mean(test_acc_result["trajectory"][FUSE_NAME], axis=-1)
                    
                    # # Step 2: Add rows to the DataFrame
                    # # Adding a row using a dictionary
                    row_data_0 = {'STM':MODEL, 'TASK':TASK, 'NUM_PLM':NUM_PLM, 'NK':NK, 'PLM_NAME':PLM_NAME, 'TOTAL_NK':TOTAL_NK, 'SAMPLE_SELECTION':SAMPLE_SELECTION, 'INHERIT_WEIGHT':INHERIT_WEIGHT, 'WEIGHT_DECAY':WEIGHT_DECAY, 'BETA':BETA, 'KD_TEMPERATURE':KD_TEMPERATURE, 'KD_ALPHA':KD_ALPHA, 'SYN_DATA':SYN_DATA, 'SEED':SEED, 'TRAIN_METHOD':"vanilla"}
                    row_data_1 = {'STM':MODEL, 'TASK':TASK, 'NUM_PLM':NUM_PLM, 'NK':NK, 'PLM_NAME':PLM_NAME, 'TOTAL_NK':TOTAL_NK, 'SAMPLE_SELECTION':SAMPLE_SELECTION, 'INHERIT_WEIGHT':INHERIT_WEIGHT, 'WEIGHT_DECAY':WEIGHT_DECAY, 'BETA':BETA, 'KD_TEMPERATURE':KD_TEMPERATURE, 'KD_ALPHA':KD_ALPHA, 'SYN_DATA':SYN_DATA, 'SEED':SEED, 'TRAIN_METHOD':"WA"}
                    row_data_2 = {'STM':MODEL, 'TASK':TASK, 'NUM_PLM':NUM_PLM, 'NK':NK, 'PLM_NAME':PLM_NAME, 'TOTAL_NK':TOTAL_NK, 'SAMPLE_SELECTION':SAMPLE_SELECTION, 'INHERIT_WEIGHT':INHERIT_WEIGHT, 'WEIGHT_DECAY':WEIGHT_DECAY, 'BETA':BETA, 'KD_TEMPERATURE':KD_TEMPERATURE, 'KD_ALPHA':KD_ALPHA, 'SYN_DATA':SYN_DATA, 'SEED':SEED, 'TRAIN_METHOD':"WA_MEAN"}
                    
                    # for plm_name in PLM_NAME:
                    #     row_data_0[plm_name] = test_acc_result["first"][plm_name]
                    #     row_data_1[plm_name] = test_acc_result["max"][plm_name]
                    #     row_data_2[plm_name] = np.mean(test_acc_result["trajectory"][plm_name], axis=-1)
                    for plm_name in ['gpt-4-turbo-preview', 'gpt-3.5-turbo-instruct', 'gpt2-xl', 'llama-2-7b-chat-hf', 'vicuna-7b-1.5v', 'opt-6.7b', 'chatglm3-6b-base', 'flan-t5-xl']:
                        if plm_name in PLM_NAME:
                            row_data_0[plm_name] = test_acc_result["first"][plm_name]
                            row_data_1[plm_name] = test_acc_result["max"][plm_name]
                            row_data_2[plm_name] = np.mean(test_acc_result["trajectory"][plm_name], axis=-1)
                        else:
                            row_data_0[plm_name] = -2
                            row_data_1[plm_name] = -2
                            row_data_2[plm_name] = -2
                    
                    for FUSE_NAME in FUSE_NAME_columns:
                        row_data_0[FUSE_NAME] = test_acc_result["first"][FUSE_NAME]
                        row_data_1[FUSE_NAME] = test_acc_result["max"][FUSE_NAME]
                        row_data_2[FUSE_NAME] = np.mean(test_acc_result["trajectory"][FUSE_NAME], axis=-1)
                    
                    if NUM_PLM == 1:
                        df_1 = df_1.append(row_data_0, ignore_index=True)
                        df_1 = df_1.append(row_data_1, ignore_index=True)
                        df_1 = df_1.append(row_data_2, ignore_index=True)
                    elif NUM_PLM == 2:
                        df_2 = df_2.append(row_data_0, ignore_index=True)
                        df_2 = df_2.append(row_data_1, ignore_index=True)
                        df_2 = df_2.append(row_data_2, ignore_index=True)
                    elif NUM_PLM == 3:
                        df_3 = df_3.append(row_data_0, ignore_index=True)
                        df_3 = df_3.append(row_data_1, ignore_index=True)
                        df_3 = df_3.append(row_data_2, ignore_index=True)
                    elif NUM_PLM == 4:
                        df_4 = df_4.append(row_data_0, ignore_index=True)
                        df_4 = df_4.append(row_data_1, ignore_index=True)
                        df_4 = df_4.append(row_data_2, ignore_index=True)
                    elif NUM_PLM == 5:
                        df_5 = df_5.append(row_data_0, ignore_index=True)
                        df_5 = df_5.append(row_data_1, ignore_index=True)
                        df_5 = df_5.append(row_data_2, ignore_index=True)
                    elif NUM_PLM == 6:
                        df_6 = df_6.append(row_data_0, ignore_index=True)
                        df_6 = df_6.append(row_data_1, ignore_index=True)
                        df_6 = df_6.append(row_data_2, ignore_index=True)
                    elif NUM_PLM == 7:
                        df_7 = df_7.append(row_data_0, ignore_index=True)
                        df_7 = df_7.append(row_data_1, ignore_index=True)
                        df_7 = df_7.append(row_data_2, ignore_index=True)

                    # # Adding another row using a different method
                    # df.loc[len(df.index)] = ['Value4', 'Value5', 'Value6']

                    # # Display the DataFrame
                    # print(df)

                    # print(file_path)
                    # print(NK, WEIGHT_DECAY, INHERIT_WEIGHT, SAMPLE_SELECTION)
                    print(NK, NUM_PLM, MODEL, TASK, SEED, SAMPLE_SELECTION, WEIGHT_DECAY, INHERIT_WEIGHT,)
                    print(test_acc_result['first'])
                    print(test_acc_result['max'])
                    print()

        # # calculate the mean and std
        # row_data_mean = {'STM':MODEL, 'NUM_PLM':NUM_PLM, 'NK':NK, 'PLM_NAME':PLM_NAME, 'TOTAL_NK':TOTAL_NK, 'SAMPLE_SELECTION':SAMPLE_SELECTION, 'INHERIT_WEIGHT':INHERIT_WEIGHT, 'WEIGHT_DECAY':WEIGHT_DECAY, 'BETA':BETA, 'SEED':'mean', 'TRAIN_METHOD':"WA_MEAN"}
        # row_data_std = {'STM':MODEL, 'NUM_PLM':NUM_PLM, 'NK':NK, 'PLM_NAME':PLM_NAME, 'TOTAL_NK':TOTAL_NK, 'SAMPLE_SELECTION':SAMPLE_SELECTION, 'INHERIT_WEIGHT':INHERIT_WEIGHT, 'WEIGHT_DECAY':WEIGHT_DECAY, 'BETA':BETA, 'SEED':'std', 'TRAIN_METHOD':"WA_MEAN"}

    df_1.to_csv("./results/csv/1_plm.csv")
    df_2.to_csv("./results/csv/2_plm.csv")
    df_3.to_csv("./results/csv/3_plm.csv")
    df_4.to_csv("./results/csv/4_plm.csv")
    df_5.to_csv("./results/csv/5_plm.csv")
    df_6.to_csv("./results/csv/6_plm.csv")
    df_7.to_csv("./results/csv/7_plm.csv")


def group_mean_calculation():
    if os.path.exists("./results/csv/1_plm.csv"):
        df = pd.read_csv("./results/csv/1_plm.csv", index_col=False)
        mean_result = df.groupby(['STM', 'TASK', 'NUM_PLM', 'NK', 'PLM_NAME', 'TOTAL_NK', 'SAMPLE_SELECTION', 'INHERIT_WEIGHT', 'WEIGHT_DECAY', 'BETA', 'KD_TEMPERATURE', 'KD_ALPHA', 'SYN_DATA', 'TRAIN_METHOD']).mean()
        # mean_result.drop(['SEED'], axis=1)
        mean_result.to_csv("./results/csv/1_plm_mean.csv")


def get_accumualte_results_in_iter():

    '''
    [0] ./
    [1] logging/
    [2] multiGold_eval_on_real/
    [3] with_real_few_shot_accumulate_votingCLASS_promptCLASS_8_0.0_randomCartographyOriginalContrast_top/
    [4] gold_100_1_0.0_OODgold/
    [5] bert-base-uncased/
    [6] sentence-t5-base/
    [7] 0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD1_FuseDataset1/
    [8] 0_1_init200_steps4_unbalance_temp1.0/
    [9] fewshotK8_15_0.5/
    [10] imdb/
    [11] gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__chatglm3-6b-base_1000__flan-t5-xl_1000/
    [12] log_data_accumulate_4.408560690471575_12345.txt
    '''

    columns_hyper_parameter = ['STM', 'TASK', 'NUM_PLM', 'NK', 'PLM_NAME', 'TOTAL_NK', 'GOLD_NUM', 'GOLD_PARTY', 'GOLD_DISTRIBUTE', 'GOLD_DIRICHLET', 'ONLY_GOLD', 'INIT_SAMPLE_COUNT', 'STEPS_COUNT', 'SAMPLE_COUNT_PER_STEP', 'SAMPLE_SELECTION', 'INHERIT_WEIGHT', 'WEIGHT_DECAY', 'BETA', 'KD_TEMPERATURE', 'KD_ALPHA', 'SYN_DATA', 'PROMPTING', 'EMBEDDING_TRANSFORMER', 'MODEL_WEIGHTING', 'MODEL_WEIGHTING_TEMPERATURE', 'FEW_SHOT_K', 'FEW_SHOT_SAMPLE_POOL_FOR_EACH', 'FEW_SHOT_SAMPLE_AMBIGUOUS_RATIO', 'SEED', 'TRAIN_METHOD']
    plm_columns = []
    for plm_name in ['gpt-4-turbo-preview', 'gpt-3.5-turbo-instruct', 'gpt2-xl', 'llama-2-7b-chat-hf', 'vicuna-7b-1.5v', 'opt-6.7b', 'chatglm3-6b-base', 'flan-t5-xl']:
        plm_columns = plm_columns + [plm_name+f'_step{i_step}' for i_step in range(5)]
    fuse_columns = []
    for fuse_name in ['all-one-hot', 'fedAVG', 'fedAVG_v2']:
        fuse_columns = fuse_columns + [fuse_name+f'_step{i_step}' for i_step in range(5)]
    fuse_columns = fuse_columns + ['all-kd', 'all-kd-s', 'good-one-hot', 'good-kd']
    # df_columns_1 = ['STM', 'TASK', 'NUM_PLM', 'NK', 'PLM_NAME', 'TOTAL_NK', 'INIT_SAMPLE_COUNT', 'STEPS_COUNT', 'SAMPLE_SELECTION', 'INHERIT_WEIGHT', 'WEIGHT_DECAY', 'BETA', 'KD_TEMPERATURE', 'KD_ALPHA', 'SYN_DATA', 'SEED', 'TRAIN_METHOD', 'gpt-4-turbo-preview', 'gpt-3.5-turbo-instruct', 'gpt2-xl', 'llama-2-7b-chat-hf', 'vicuna-7b-1.5v', 'opt-6.7b', 'chatglm3-6b-base', 'flan-t5-xl', 'all-one-hot', 'all-kd', 'all-kd-s', 'good-one-hot', 'good-kd']
    # df_columns_2 = ['STM', 'TASK', 'NUM_PLM', 'NK', 'PLM_NAME', 'TOTAL_NK', 'INIT_SAMPLE_COUNT', 'STEPS_COUNT', 'SAMPLE_SELECTION', 'INHERIT_WEIGHT', 'WEIGHT_DECAY', 'BETA', 'KD_TEMPERATURE', 'KD_ALPHA', 'SYN_DATA', 'SEED', 'TRAIN_METHOD', 'gpt-4-turbo-preview', 'gpt-3.5-turbo-instruct', 'gpt2-xl', 'llama-2-7b-chat-hf', 'vicuna-7b-1.5v', 'opt-6.7b', 'chatglm3-6b-base', 'flan-t5-xl', 'all-one-hot', 'all-kd', 'all-kd-s', 'good-one-hot', 'good-kd']
    # df_columns_3 = ['STM', 'TASK', 'NUM_PLM', 'NK', 'PLM_NAME', 'TOTAL_NK', 'INIT_SAMPLE_COUNT', 'STEPS_COUNT', 'SAMPLE_SELECTION', 'INHERIT_WEIGHT', 'WEIGHT_DECAY', 'BETA', 'KD_TEMPERATURE', 'KD_ALPHA', 'SYN_DATA', 'SEED', 'TRAIN_METHOD', 'gpt-4-turbo-preview', 'gpt-3.5-turbo-instruct', 'gpt2-xl', 'llama-2-7b-chat-hf', 'vicuna-7b-1.5v', 'opt-6.7b', 'chatglm3-6b-base', 'flan-t5-xl', 'all-one-hot', 'all-kd', 'all-kd-s', 'good-one-hot', 'good-kd']
    # df_columns_4 = ['STM', 'TASK', 'NUM_PLM', 'NK', 'PLM_NAME', 'TOTAL_NK', 'INIT_SAMPLE_COUNT', 'STEPS_COUNT', 'SAMPLE_SELECTION', 'INHERIT_WEIGHT', 'WEIGHT_DECAY', 'BETA', 'KD_TEMPERATURE', 'KD_ALPHA', 'SYN_DATA', 'SEED', 'TRAIN_METHOD', 'gpt-4-turbo-preview', 'gpt-3.5-turbo-instruct', 'gpt2-xl', 'llama-2-7b-chat-hf', 'vicuna-7b-1.5v', 'opt-6.7b', 'chatglm3-6b-base', 'flan-t5-xl', 'all-one-hot', 'all-kd', 'all-kd-s', 'good-one-hot', 'good-kd']
    # df_columns_5 = ['STM', 'TASK', 'NUM_PLM', 'NK', 'PLM_NAME', 'TOTAL_NK', 'INIT_SAMPLE_COUNT', 'STEPS_COUNT', 'SAMPLE_SELECTION', 'INHERIT_WEIGHT', 'WEIGHT_DECAY', 'BETA', 'KD_TEMPERATURE', 'KD_ALPHA', 'SYN_DATA', 'SEED', 'TRAIN_METHOD', 'gpt-4-turbo-preview', 'gpt-3.5-turbo-instruct', 'gpt2-xl', 'llama-2-7b-chat-hf', 'vicuna-7b-1.5v', 'opt-6.7b', 'chatglm3-6b-base', 'flan-t5-xl', 'all-one-hot', 'all-kd', 'all-kd-s', 'good-one-hot', 'good-kd']
    # df_columns_6 = ['STM', 'TASK', 'NUM_PLM', 'NK', 'PLM_NAME', 'TOTAL_NK', 'INIT_SAMPLE_COUNT', 'STEPS_COUNT', 'SAMPLE_SELECTION', 'INHERIT_WEIGHT', 'WEIGHT_DECAY', 'BETA', 'KD_TEMPERATURE', 'KD_ALPHA', 'SYN_DATA', 'SEED', 'TRAIN_METHOD', 'gpt-4-turbo-preview', 'gpt-3.5-turbo-instruct', 'gpt2-xl', 'llama-2-7b-chat-hf', 'vicuna-7b-1.5v', 'opt-6.7b', 'chatglm3-6b-base', 'flan-t5-xl', 'all-one-hot', 'all-kd', 'all-kd-s', 'good-one-hot', 'good-kd']
    # df_columns_7 = ['STM', 'TASK', 'NUM_PLM', 'NK', 'PLM_NAME', 'TOTAL_NK', 'INIT_SAMPLE_COUNT', 'STEPS_COUNT', 'SAMPLE_SELECTION', 'INHERIT_WEIGHT', 'WEIGHT_DECAY', 'BETA', 'KD_TEMPERATURE', 'KD_ALPHA', 'SYN_DATA', 'SEED', 'TRAIN_METHOD', 'gpt-4-turbo-preview', 'gpt-3.5-turbo-instruct', 'gpt2-xl', 'llama-2-7b-chat-hf', 'vicuna-7b-1.5v', 'opt-6.7b', 'chatglm3-6b-base', 'flan-t5-xl', 'all-one-hot', 'all-kd', 'all-kd-s', 'good-one-hot', 'good-kd']
    df_columns = columns_hyper_parameter + plm_columns + fuse_columns
    df_1 = pd.DataFrame(columns=df_columns)
    df_2 = pd.DataFrame(columns=df_columns)
    df_3 = pd.DataFrame(columns=df_columns)
    df_4 = pd.DataFrame(columns=df_columns)
    df_5 = pd.DataFrame(columns=df_columns)
    df_6 = pd.DataFrame(columns=df_columns)
    df_7 = pd.DataFrame(columns=df_columns)

    # for root, dirs, files in os.walk("./logging/eval_on_real/few_shot_accumulate/"):
    # for root, dirs, files in os.walk("./logging/eval_on_real/few_shot_accumulate_v2/"):
    # for root, dirs, files in os.walk("./logging/eval_on_real/few_shot_accumulate/bert-base-uncased/0.9_errorOnlyWrongOnlySelf_Adjust_increasedThetaFlip_Entropy_KD1_FuseDataset1/0_1_init200_steps4/fewshotK8_5_0.3/imdb/gpt2-xl_1000__llama-2-7b-chat-hf_1000/"):
    for _dir in [
                #  './logging/eval_on_real/'
                    './logging/multiGold_eval_on_real/'
                #  '/home/DAIR/zouty/ModelFederation/PrivateGenerateEnhancement/src/logging/eval_on_real/with_real_few_shot_accumulate_votingCLASS_8_random_top/bert-base-uncased/sentence-t5-base/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD1_FuseDataset1/0_1_init1200_steps4_unbalance_temp3/fewshotK8_15_0.5/imdb/gpt-3.5-turbo-instruct_6000'
                #  './logging/eval_on_real/with_real_few_shot_accumulate_votingCLASS_8_Cartography_top/bert-base-uncased/sentence-t5-base/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD1_FuseDataset1/0_1_init200_steps4_unbalance_temp3/fewshotK8_5_0.5/imdb/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__chatglm3-6b-base_1000__flan-t5-xl_1000/'
                 ]:
        for root, dirs, files in os.walk(_dir):
            for file in files:
                print(f"{file=}")
                if file.endswith('.txt'):
                    # Print the path of the text file
                    file_path = os.path.join(root, file)
                    if not 'steps' in file_path:
                        continue
                    # if not ('TestAll_new_all' in file_path or 'TestAll_inheritModel_all' in file_path):
                    #     continue
                    if not '0.9' in file_path:
                        continue
                    # if not 'Flip' in file_path:
                    #     continue
                    if not 'KD1_FuseDataset1' in file_path and not 'KD0_FuseDataset0' in file_path:
                        continue
                    if 'nan' in file_path:
                        continue
                    # if not TASK_NAME in file_path:
                    #     continue
                    print(file_path)
                    folders = file_path.split("/")
                    MODEL = folders[5]
                    EMBEDDING_TRANSFORMER = MODEL if f'{MODEL}/0.9_' in file_path else folders[6]
                    TASK = folders[-3]
                    KD_TEMPERATURE = 1
                    MODEL_WEIGHTING = 'unbalance' if 'unbalance' in folders[-5] else 'balance'
                    MODEL_WEIGHTING_TEMPERATURE = -1
                    KD_ALPHA = 0
                    FEW_SHOT_K = 8
                    FEW_SHOT_SAMPLE_POOL_FOR_EACH = 40
                    FEW_SHOT_SAMPLE_AMBIGUOUS_RATIO = 0.5
                    PROMPTING = 'influenceCartography'
                    GOLD_NUM = 100
                    GOLD_PARTY = 1
                    GOLD_DISTRIBUTE = 'OOD'
                    GOLD_DIRICHLET = 0.0
                    ONLY_GOLD = -1.0
                    # print(MODEL)
                    for folder in folders:
                        # print(folder)
                        if '__' in folder or folder == folders[-2]:
                            PLMS = folder.split("__")
                            NUM_PLM = len(PLMS)
                            NK = [int(item.split("_")[-1]) for item in PLMS]
                            PLM_NAME = [str(item.split("_")[0]) for item in PLMS]
                            TOTAL_NK = sum(NK)
                        elif '.txt' in folder:
                            floating_point_pattern = r'\d+\.\d+'
                            matches = re.findall(floating_point_pattern, folder)
                            # print(matches)
                            BETA = matches[0]
                            seed_pattern = r'_\d+.txt'
                            matches = re.findall(seed_pattern, folder)
                            SEED = int(matches[0][1:-4])
                        elif 'with_real' in folder:
                            if 'few_shot_accumulate' in folder:
                                SYN_DATA = 'few_shot'
                            if 'mixed' in folder:
                                SYN_DATA += '_mixed'
                            # if 'voting' in folder:
                            #     temps = folder.split('_')
                            #     PROMPTING = temps[5]+"_"+str(temps[7])
                            # else:
                            #     PROMPTING = ''
                            # if '_sampling' in folder:
                            #     PROMPTING += '_sampling'
                            # else:
                            #     PROMPTING += '_top'
                            # if '_influence' in folder:
                            #     PROMPTING += '_influence'
                            # elif '_random' in folder:
                            #     PROMPTING += '_random'
                            # elif '_Cartography' in folder:
                            #     PROMPTING += '_Cartography'
                            _idx = folder.find('voting')
                            if _idx != -1:
                                PROMPTING = folder[_idx:]
                            else:
                                PROMPTING = ''
                        elif 'gold' in folder:
                            temps = folder.split('_')
                            GOLD_NUM = int(temps[1])
                            GOLD_PARTY = int(temps[2])
                            GOLD_DISTRIBUTE = temps[-1][:-4]
                            GOLD_DIRICHLET = float(temps[3])
                        elif 'init' in folder:
                        # elif len(folder.split('_')) == 2 and (not any(char.isalpha() for char in folder)):
                            float_numbers = folder.split('_')
                            KD_ALPHA = float(float_numbers[0])
                            KD_TEMPERATURE = float(float_numbers[1])
                            INIT_SAMPLE_COUNT = int(float_numbers[2][4:])
                            STEPS_COUNT = int(float_numbers[3][5:])
                            if 'unbalance' in folder:
                                MODEL_WEIGHTING = 'unbalance'
                                MODEL_WEIGHTING_TEMPERATURE = float(folder.split('_')[-1][4:])
                        elif '0.9' in folder and not '_0.9' in folder:
                            types = folder.split('_')
                            WEIGHT_DECAY = types[1]
                            INHERIT_WEIGHT = types[2]
                            SAMPLE_SELECTION = types[3]
                        elif 'error' in folder or 'loss' in folder:
                            types = folder.split('_')
                            WEIGHT_DECAY = types[0]
                            INHERIT_WEIGHT = types[1]
                            SAMPLE_SELECTION = types[3]
                        elif 'fewshotK' in folder:
                            fewshot_hyper_parameters = folder.split('_')
                            FEW_SHOT_K = int(fewshot_hyper_parameters[0][8:])
                            FEW_SHOT_SAMPLE_POOL_FOR_EACH = int(fewshot_hyper_parameters[1])
                            if len(fewshot_hyper_parameters) > 2:
                                FEW_SHOT_SAMPLE_AMBIGUOUS_RATIO = float(fewshot_hyper_parameters[2])


                    FUSE_NAME_columns = ['fedAVG', 'fedAVG_v2', 'all-one-hot']
                    # step_counter = [-1] * (len(PLM_NAME)+1) # count for each PLM's syn-data and total syn-data
                    step_counter = -1
                    last_total_sample = 0
                    print(step_counter)
                    
                    # print(f"BETA={BETA}, NK={NK}, PLM_NAME={PLM_NAME}")
                    with open(file_path, 'r') as file:
                        test_acc_result = {"max":{}, "max_epoch":{}, "first":{}, "trajectory":{}, "mean":{}}
                        SAMPLE_COUNT = [[]] * (STEPS_COUNT+1)
                        for plm_name in PLM_NAME + ['all-one-hot', 'fedAVG', 'fedAVG_v2']:
                            test_acc_result["max"][plm_name] = [-1.0] * (STEPS_COUNT+1)
                            test_acc_result["first"][plm_name] = [-1.0] * (STEPS_COUNT+1)
                            test_acc_result["max_epoch"][plm_name] = [-1] * (STEPS_COUNT+1)
                            test_acc_result["trajectory"][plm_name] = [[]] * (STEPS_COUNT+1)
                        if 'squad' in TASK:
                            multiply_coefficient = 1
                        else:
                            multiply_coefficient = 100
                        for line in file:
                            line = line.strip()
                            if 'ratio of elements whose delta in theta larger than 0.0 is ' in line:
                                pattern = r'\d+/\d+'
                                matches = re.findall(pattern, line)
                                current_total_sample = int(matches[0].split('/')[-1])
                                print(f"[debug] before {current_total_sample=},, {last_total_sample=}, {step_counter=}")
                                # assert current_total_sample >= last_total_sample, f"{current_total_sample=}, {last_total_sample=}, too much record in file {file_path}"
                                if current_total_sample < last_total_sample:
                                    for key in test_acc_result["max"].keys():
                                        if key in PLM_NAME + ['all-one-hot', 'fedAVG', 'fedAVG_v2']:
                                            test_acc_result["max"][key] = test_acc_result["max"][key][:1]+ [-1.0] * (STEPS_COUNT)
                                            test_acc_result["first"][key] = test_acc_result["first"][key][:1] + [-1.0] * (STEPS_COUNT)
                                            test_acc_result["max_epoch"][key] = test_acc_result["max_epoch"][key][:1] + [-1] * (STEPS_COUNT)
                                            test_acc_result["trajectory"][key] = test_acc_result["trajectory"][key][:1] + [[]] * (STEPS_COUNT)
                                        else:
                                            test_acc_result["max"][key] = -1.0
                                            test_acc_result["first"][key] = -1.0
                                            test_acc_result["max_epoch"][key] = -1
                                            test_acc_result["trajectory"][key] = []
                                    step_counter = 0
                                    last_total_sample = current_total_sample
                                if (current_total_sample >= TOTAL_NK and step_counter < STEPS_COUNT) or (current_total_sample < TOTAL_NK and current_total_sample != last_total_sample):
                                    step_counter += 1
                                    last_total_sample = current_total_sample
                                print(f"[debug] after {current_total_sample=}, {last_total_sample=}, {step_counter=}")
                            # if '#ambiguous & easy-to-learn samples of each PLM' in line:
                            if 'nearest_sample_voting=' in line:
                                if NUM_PLM == 1 and 'Flip' not in SAMPLE_SELECTION and last_total_sample == 0:
                                    step_counter += 1
                            if 'use [' in line and '] train data...' in line:
                                print(f'[debug] {line=}')
                                start_idx = line.rfind('[')
                                end_idx = line.rfind(']')
                                str_of_list = line[start_idx+1:end_idx]
                                print(f'[debug] {str_of_list=}')
                                list_of_str = str_of_list.replace(' ', '').split(',')
                                SAMPLE_COUNT[step_counter+1] = [int(_i) for _i in list_of_str]
                                print(f'[debug] {SAMPLE_COUNT[step_counter+1]=}')
                            if not 'test_acc' in line:
                                continue
                            if "LLM#" in line:
                                if 'Flip' in SAMPLE_SELECTION and not ', new_' in line:
                                    continue
                                pattern = r'LLM#\d+'
                                matches = re.findall(pattern, line)
                                plm_index = int(matches[0][4:])
                                plm_name = PLM_NAME[plm_index]
                                floating_point_pattern = r'(?<=\=)[+-]?\d*\.\d+'
                                matches = re.findall(floating_point_pattern, line)
                                # if test_acc_result["first"][plm_name][step_counter[plm_index]+1] < 0 and float(matches[2]) > 0.5000000000001:
                                #     step_counter[plm_index] += 1
                                #     test_acc_result["first"][plm_name][step_counter[plm_index]] = float(matches[-2])*multiply_coefficient
                                # if test_acc_result["max"][plm_name][step_counter[plm_index]] < float(matches[-2])*multiply_coefficient and float(matches[2]) > 0.5000000000001:
                                #     test_acc_result["max"][plm_name][step_counter[plm_index]] = float(matches[-2])*multiply_coefficient
                                #     iter_pattern = r'iter=\d+'
                                #     # print(f"{line=}, {re.findall(iter_pattern, line)=}")
                                #     test_acc_result["max_epoch"][plm_name][step_counter[plm_index]] = int(re.findall(iter_pattern, line)[0][5:])
                                # test_acc_result["trajectory"][plm_name][step_counter[plm_index]].append(float(matches[-2])*multiply_coefficient)                            
                                if 'Flip' in SAMPLE_SELECTION:
                                    # print(f"in flip & plm, {step_counter=}")
                                    if test_acc_result["first"][plm_name][step_counter] < 0: # and float(matches[2]) > 0.5000000000001
                                        test_acc_result["first"][plm_name][step_counter] = float(matches[-2])*multiply_coefficient
                                    if test_acc_result["max"][plm_name][step_counter] < float(matches[-2])*multiply_coefficient: # and float(matches[2]) > 0.5000000000001
                                        test_acc_result["max"][plm_name][step_counter] = float(matches[-2])*multiply_coefficient
                                        iter_pattern = r'iter=\d+'
                                        # print(f"{line=}, {re.findall(iter_pattern, line)=}")
                                        test_acc_result["max_epoch"][plm_name][step_counter] = int(re.findall(iter_pattern, line)[0][5:])
                                    test_acc_result["trajectory"][plm_name][step_counter].append(float(matches[-2])*multiply_coefficient)                            
                                else:
                                    print(f"in no-flip & plm, {step_counter=}")
                                    if step_counter >= 4:
                                        continue
                                    if test_acc_result["first"][plm_name][step_counter+1] < 0: # and float(matches[2]) > 0.5000000000001
                                        test_acc_result["first"][plm_name][step_counter+1] = float(matches[-2])*multiply_coefficient
                                    if test_acc_result["max"][plm_name][step_counter+1] < float(matches[-2])*multiply_coefficient: # and float(matches[2]) > 0.5000000000001
                                        test_acc_result["max"][plm_name][step_counter+1] = float(matches[-2])*multiply_coefficient
                                        iter_pattern = r'iter=\d+'
                                        # print(f"{line=}, {re.findall(iter_pattern, line)=}")
                                        test_acc_result["max_epoch"][plm_name][step_counter+1] = int(re.findall(iter_pattern, line)[0][5:])
                                    test_acc_result["trajectory"][plm_name][step_counter+1].append(float(matches[-2])*multiply_coefficient)                            
                            elif 'fedAVG' in line or 'all LLMs' in line:
                                fed_type = 'fedAVG_v2' if '_v2' in line else ( 'all-one-hot' if 'LLMs' in line else 'fedAVG')
                                floating_point_pattern = r'(?<=\=)[+-]?\d*\.\d+'
                                matches = re.findall(floating_point_pattern, line)
                                if len(matches) < 2:
                                    matches.append('nan')
                                # pattern = r'#iter=-?\d+'
                                # matches = re.findall(pattern, line)
                                # iter_count = int(matches[0][6:])
                                if 'Flip' in SAMPLE_SELECTION:
                                    # print(f"in flip & plm, {step_counter=}")
                                    if step_counter < 0 and ONLY_GOLD < 0 and fed_type == 'fedAVG':
                                        ONLY_GOLD = float(matches[-2])*multiply_coefficient
                                        continue
                                    if test_acc_result["first"][fed_type][step_counter] < 0: # and float(matches[2]) > 0.5000000000001
                                        test_acc_result["first"][fed_type][step_counter] = float(matches[-2])*multiply_coefficient
                                    if test_acc_result["max"][fed_type][step_counter] < float(matches[-2])*multiply_coefficient: # and float(matches[2]) > 0.5000000000001
                                        test_acc_result["max"][fed_type][step_counter] = float(matches[-2])*multiply_coefficient
                                        iter_pattern = r'iter=\d+'
                                        # print(f"{line=}, {re.findall(iter_pattern, line)=}")
                                        test_acc_result["max_epoch"][fed_type][step_counter] = int(re.findall(iter_pattern, line)[0][5:])
                                    test_acc_result["trajectory"][fed_type][step_counter].append(float(matches[-2])*multiply_coefficient)                            
                                else:
                                    print(f"in no-flip & plm, {step_counter=}")
                                    if step_counter >= 4:
                                        continue
                                    if step_counter < 0 and ONLY_GOLD < 0 and fed_type == 'fedAVG':
                                        ONLY_GOLD = float(matches[-2])*multiply_coefficient
                                        continue
                                    if test_acc_result["first"][fed_type][step_counter+1] < 0: # and float(matches[2]) > 0.5000000000001
                                        test_acc_result["first"][fed_type][step_counter+1] = float(matches[-2])*multiply_coefficient
                                    if test_acc_result["max"][fed_type][step_counter+1] < float(matches[-2])*multiply_coefficient: # and float(matches[2]) > 0.5000000000001
                                        test_acc_result["max"][fed_type][step_counter+1] = float(matches[-2])*multiply_coefficient
                                        # iter_pattern = r'iter=\d+'
                                        # # print(f"{line=}, {re.findall(iter_pattern, line)=}")
                                        # test_acc_result["max_epoch"][plm_name][step_counter+1] = int(re.findall(iter_pattern, line)[0][5:])
                                        test_acc_result["max_epoch"][plm_name][step_counter+1] = int(step_counter+1)
                                    test_acc_result["trajectory"][plm_name][step_counter+1].append(float(matches[-2])*multiply_coefficient)                            
                        #     elif '-(' in line:
                        #         type_pattern = r'\((.*?)\)'
                        #         matches = re.findall(type_pattern, line)
                        #         FUSE_NAME = matches[0]
                        #         print(f"{FUSE_NAME=}")
                        #         if 'separateod' in FUSE_NAME and not 'vote-(' in line:
                        #             continue
                        #         # print(f"1{FUSE_NAME=}")
                        #         if not FUSE_NAME in test_acc_result["max"].keys():
                        #             if FUSE_NAME != 'all-one-hot':
                        #                 # print(f"add key {FUSE_NAME}")
                        #                 test_acc_result["max"][FUSE_NAME] = -1.0
                        #                 test_acc_result["first"][FUSE_NAME] = -1.0
                        #                 test_acc_result["max_epoch"][FUSE_NAME] = -1
                        #                 test_acc_result["trajectory"][FUSE_NAME] = []
                        #                 # test_acc_result["mean"][FUSE_NAME] = -1
                        #                 FUSE_NAME_columns.append(FUSE_NAME)
                        #             else:
                        #                 # print(f"add key {FUSE_NAME}")
                        #                 test_acc_result["max"][FUSE_NAME] = [-1.0] * (STEPS_COUNT+1)
                        #                 test_acc_result["first"][FUSE_NAME] = [-1.0] * (STEPS_COUNT+1)
                        #                 test_acc_result["max_epoch"][FUSE_NAME] = [-1] * (STEPS_COUNT+1)
                        #                 test_acc_result["trajectory"][FUSE_NAME] = [[]] * (STEPS_COUNT+1)
                        #                 # test_acc_result["mean"][FUSE_NAME] = [-1] * (STEPS_COUNT+1)
                        #                 FUSE_NAME_columns.append(FUSE_NAME)

                        #         # print("no LLM# in line:", line)
                        #         floating_point_pattern = r'(?<=\=)[+-]?\d*\.\d+'
                        #         matches = re.findall(floating_point_pattern, line)
                        #         # # print(matches)
                        #         # train_loss.append(float(matches[0]))
                        #         # train_acc.append(float(matches[1]))
                        #         # test_acc.append(float(matches[2]))
                        #         # test_loss.append(float(matches[3]))
                        #         if 'vote-(' in line:
                        #             match_idx = 0
                        #         else:
                        #             match_idx = -2
                        #         # print(f"{line=}, {match_idx=}, matches={matches}")
                        #         if FUSE_NAME != 'all-one-hot':
                        #             if test_acc_result["first"][FUSE_NAME] < 0:
                        #                 test_acc_result["first"][FUSE_NAME] = float(matches[match_idx])*multiply_coefficient
                        #             if test_acc_result["max"][FUSE_NAME] < float(matches[match_idx])*multiply_coefficient:
                        #                 test_acc_result["max"][FUSE_NAME] = float(matches[match_idx])*multiply_coefficient
                        #                 iter_pattern = r'iter=\d+'
                        #                 # print(re.findall(iter_pattern, line))
                        #                 test_acc_result["max_epoch"][FUSE_NAME] = int(re.findall(iter_pattern, line)[0][5:])
                        #             test_acc_result["trajectory"][FUSE_NAME].append(float(matches[match_idx])*multiply_coefficient)
                        #         else:
                        #             # print(f"{step_counter[-1]+1=}")
                        #             # if test_acc_result["first"][FUSE_NAME][step_counter[-1]+1] < 0:
                        #             #     # print(f"{matches=}")
                        #             #     step_counter[-1] += 1
                        #             #     test_acc_result["first"][FUSE_NAME][step_counter[-1]] = float(matches[match_idx])*multiply_coefficient
                        #             # if test_acc_result["max"][FUSE_NAME][step_counter[-1]] < float(matches[match_idx])*multiply_coefficient:
                        #             #     test_acc_result["max"][FUSE_NAME][step_counter[-1]] = float(matches[match_idx])*multiply_coefficient
                        #             #     iter_pattern = r'iter=\d+'
                        #             #     # print(re.findall(iter_pattern, line))
                        #             #     test_acc_result["max_epoch"][FUSE_NAME][step_counter[-1]] = int(re.findall(iter_pattern, line)[0][5:])
                        #             # test_acc_result["trajectory"][FUSE_NAME][step_counter[-1]].append(float(matches[match_idx])*multiply_coefficient)
                        #             # print(f"{step_counter=}")
                        #             if NUM_PLM == 1 and 'Flip' not in SAMPLE_SELECTION and last_total_sample == 0:
                        #                 if test_acc_result["first"][FUSE_NAME][step_counter+1] < 0:
                        #                     # print(f"{matches=}")
                        #                     test_acc_result["first"][FUSE_NAME][step_counter+1] = float(matches[match_idx])*multiply_coefficient
                        #                 if test_acc_result["max"][FUSE_NAME][step_counter+1] < float(matches[match_idx])*multiply_coefficient:
                        #                     test_acc_result["max"][FUSE_NAME][step_counter+1] = float(matches[match_idx])*multiply_coefficient
                        #                     iter_pattern = r'iter=\d+'
                        #                     # print(re.findall(iter_pattern, line))
                        #                     test_acc_result["max_epoch"][FUSE_NAME][step_counter+1] = int(re.findall(iter_pattern, line)[0][5:])
                        #                 test_acc_result["trajectory"][FUSE_NAME][step_counter+1].append(float(matches[match_idx])*multiply_coefficient)
                        #             else:
                        #                 if test_acc_result["first"][FUSE_NAME][step_counter] < 0:
                        #                     # print(f"{matches=}")
                        #                     test_acc_result["first"][FUSE_NAME][step_counter] = float(matches[match_idx])*multiply_coefficient
                        #                 if test_acc_result["max"][FUSE_NAME][step_counter] < float(matches[match_idx])*multiply_coefficient:
                        #                     test_acc_result["max"][FUSE_NAME][step_counter] = float(matches[match_idx])*multiply_coefficient
                        #                     iter_pattern = r'iter=\d+'
                        #                     # print(re.findall(iter_pattern, line))
                        #                     test_acc_result["max_epoch"][FUSE_NAME][step_counter] = int(re.findall(iter_pattern, line)[0][5:])
                        #                 test_acc_result["trajectory"][FUSE_NAME][step_counter].append(float(matches[match_idx])*multiply_coefficient)
                        # # print(f"{FUSE_NAME=}")
                        # # # print(test_acc_result["trajectory"][FUSE_NAME])
                        # # test_acc_result["mean"][FUSE_NAME] = np.mean(test_acc_result["trajectory"][FUSE_NAME], axis=-1)
                        
                        # # Step 2: Add rows to the DataFrame
                        # # Adding a row using a dictionary
                        row_data_0 = {'STM':MODEL, 'TASK':TASK, 'NUM_PLM':NUM_PLM, 'NK':NK, 'PLM_NAME':PLM_NAME, 'TOTAL_NK':TOTAL_NK, 'GOLD_NUM':GOLD_NUM, 'GOLD_PARTY':GOLD_PARTY, 'GOLD_DISTRIBUTE':GOLD_DISTRIBUTE, 'GOLD_DIRICHLET':GOLD_DIRICHLET, 'ONLY_GOLD': ONLY_GOLD, 'INIT_SAMPLE_COUNT':INIT_SAMPLE_COUNT, 'STEPS_COUNT':STEPS_COUNT, 'SAMPLE_COUNT_PER_STEP':SAMPLE_COUNT, 'SAMPLE_SELECTION':SAMPLE_SELECTION, 'INHERIT_WEIGHT':INHERIT_WEIGHT, 'WEIGHT_DECAY':WEIGHT_DECAY, 'BETA':BETA, 'KD_TEMPERATURE':KD_TEMPERATURE, 'KD_ALPHA':KD_ALPHA, 'SYN_DATA':SYN_DATA, 'EMBEDDING_TRANSFORMER':EMBEDDING_TRANSFORMER, 'MODEL_WEIGHTING':MODEL_WEIGHTING, 'MODEL_WEIGHTING_TEMPERATURE':MODEL_WEIGHTING_TEMPERATURE, 'PROMPTING':PROMPTING, 'FEW_SHOT_K':FEW_SHOT_K, 'FEW_SHOT_SAMPLE_POOL_FOR_EACH':FEW_SHOT_SAMPLE_POOL_FOR_EACH, 'FEW_SHOT_SAMPLE_AMBIGUOUS_RATIO':FEW_SHOT_SAMPLE_AMBIGUOUS_RATIO, 'SEED':SEED, 'TRAIN_METHOD':"vanilla"}
                        row_data_1 = {'STM':MODEL, 'TASK':TASK, 'NUM_PLM':NUM_PLM, 'NK':NK, 'PLM_NAME':PLM_NAME, 'TOTAL_NK':TOTAL_NK, 'GOLD_NUM':GOLD_NUM, 'GOLD_PARTY':GOLD_PARTY, 'GOLD_DISTRIBUTE':GOLD_DISTRIBUTE, 'GOLD_DIRICHLET':GOLD_DIRICHLET, 'ONLY_GOLD': ONLY_GOLD, 'INIT_SAMPLE_COUNT':INIT_SAMPLE_COUNT, 'STEPS_COUNT':STEPS_COUNT, 'SAMPLE_COUNT_PER_STEP':SAMPLE_COUNT, 'SAMPLE_SELECTION':SAMPLE_SELECTION, 'INHERIT_WEIGHT':INHERIT_WEIGHT, 'WEIGHT_DECAY':WEIGHT_DECAY, 'BETA':BETA, 'KD_TEMPERATURE':KD_TEMPERATURE, 'KD_ALPHA':KD_ALPHA, 'SYN_DATA':SYN_DATA, 'EMBEDDING_TRANSFORMER':EMBEDDING_TRANSFORMER, 'MODEL_WEIGHTING':MODEL_WEIGHTING, 'MODEL_WEIGHTING_TEMPERATURE':MODEL_WEIGHTING_TEMPERATURE, 'PROMPTING':PROMPTING, 'FEW_SHOT_K':FEW_SHOT_K, 'FEW_SHOT_SAMPLE_POOL_FOR_EACH':FEW_SHOT_SAMPLE_POOL_FOR_EACH, 'FEW_SHOT_SAMPLE_AMBIGUOUS_RATIO':FEW_SHOT_SAMPLE_AMBIGUOUS_RATIO, 'SEED':SEED, 'TRAIN_METHOD':"WA"}
                        row_data_2 = {'STM':MODEL, 'TASK':TASK, 'NUM_PLM':NUM_PLM, 'NK':NK, 'PLM_NAME':PLM_NAME, 'TOTAL_NK':TOTAL_NK, 'GOLD_NUM':GOLD_NUM, 'GOLD_PARTY':GOLD_PARTY, 'GOLD_DISTRIBUTE':GOLD_DISTRIBUTE, 'GOLD_DIRICHLET':GOLD_DIRICHLET, 'ONLY_GOLD': ONLY_GOLD, 'INIT_SAMPLE_COUNT':INIT_SAMPLE_COUNT, 'STEPS_COUNT':STEPS_COUNT, 'SAMPLE_COUNT_PER_STEP':SAMPLE_COUNT, 'SAMPLE_SELECTION':SAMPLE_SELECTION, 'INHERIT_WEIGHT':INHERIT_WEIGHT, 'WEIGHT_DECAY':WEIGHT_DECAY, 'BETA':BETA, 'KD_TEMPERATURE':KD_TEMPERATURE, 'KD_ALPHA':KD_ALPHA, 'SYN_DATA':SYN_DATA, 'EMBEDDING_TRANSFORMER':EMBEDDING_TRANSFORMER, 'MODEL_WEIGHTING':MODEL_WEIGHTING, 'MODEL_WEIGHTING_TEMPERATURE':MODEL_WEIGHTING_TEMPERATURE, 'PROMPTING':PROMPTING, 'FEW_SHOT_K':FEW_SHOT_K, 'FEW_SHOT_SAMPLE_POOL_FOR_EACH':FEW_SHOT_SAMPLE_POOL_FOR_EACH, 'FEW_SHOT_SAMPLE_AMBIGUOUS_RATIO':FEW_SHOT_SAMPLE_AMBIGUOUS_RATIO, 'SEED':SEED, 'TRAIN_METHOD':"WA_MEAN"}
                        
                        # for plm_name in PLM_NAME:
                        #     row_data_0[plm_name] = test_acc_result["first"][plm_name]
                        #     row_data_1[plm_name] = test_acc_result["max"][plm_name]
                        #     row_data_2[plm_name] = np.mean(test_acc_result["trajectory"][plm_name], axis=-1)
                        for plm_name in ['gpt-4-turbo-preview', 'gpt-3.5-turbo-instruct', 'gpt2-xl', 'llama-2-7b-chat-hf', 'vicuna-7b-1.5v', 'opt-6.7b', 'chatglm3-6b-base', 'flan-t5-xl']:
                            if plm_name in PLM_NAME:
                                for i_step in range(STEPS_COUNT+1):
                                    row_data_0[plm_name+f'_step{i_step}'] = test_acc_result["first"][plm_name][i_step]
                                    row_data_1[plm_name+f'_step{i_step}'] = test_acc_result["max"][plm_name][i_step]
                                    row_data_2[plm_name+f'_step{i_step}'] = np.mean(test_acc_result["trajectory"][plm_name][i_step], axis=-1)
                            else:
                                row_data_0[plm_name] = -2
                                row_data_1[plm_name] = -2
                                row_data_2[plm_name] = -2
                        
                        for FUSE_NAME in FUSE_NAME_columns:
                            # if FUSE_NAME != 'all-one-hot':
                            #     row_data_0[FUSE_NAME] = test_acc_result["first"][FUSE_NAME]
                            #     row_data_1[FUSE_NAME] = test_acc_result["max"][FUSE_NAME]
                            #     row_data_2[FUSE_NAME] = np.mean(test_acc_result["trajectory"][FUSE_NAME], axis=-1)
                            # else:
                            #     for i_step in range(STEPS_COUNT+1):
                            #         row_data_0[FUSE_NAME+f'_step{i_step}'] = test_acc_result["first"][FUSE_NAME][i_step]
                            #         row_data_1[FUSE_NAME+f'_step{i_step}'] = test_acc_result["max"][FUSE_NAME][i_step]
                            #         row_data_2[FUSE_NAME+f'_step{i_step}'] = np.mean(test_acc_result["trajectory"][FUSE_NAME][i_step], axis=-1)
                            for i_step in range(STEPS_COUNT+1):
                                row_data_0[FUSE_NAME+f'_step{i_step}'] = test_acc_result["first"][FUSE_NAME][i_step]
                                row_data_1[FUSE_NAME+f'_step{i_step}'] = test_acc_result["max"][FUSE_NAME][i_step]
                                row_data_2[FUSE_NAME+f'_step{i_step}'] = np.mean(test_acc_result["trajectory"][FUSE_NAME][i_step], axis=-1)

                        if NUM_PLM == 1:
                            df_1.loc[len(df_1)] = row_data_0
                            df_1.loc[len(df_1)] = row_data_1
                            df_1.loc[len(df_1)] = row_data_2
                        elif NUM_PLM == 2:
                            df_2.loc[len(df_2)] = row_data_0
                            df_2.loc[len(df_2)] = row_data_1
                            df_2.loc[len(df_2)] = row_data_2
                        elif NUM_PLM == 3:
                            df_3.loc[len(df_3)] = row_data_0
                            df_3.loc[len(df_3)] = row_data_1
                            df_3.loc[len(df_3)] = row_data_2
                        elif NUM_PLM == 4:
                            df_4.loc[len(df_4)] = row_data_0
                            df_4.loc[len(df_4)] = row_data_1
                            df_4.loc[len(df_4)] = row_data_2
                        elif NUM_PLM == 5:
                            df_5.loc[len(df_5)] = row_data_0
                            df_5.loc[len(df_5)] = row_data_1
                            df_5.loc[len(df_5)] = row_data_2
                        elif NUM_PLM == 6:
                            df_6.loc[len(df_6)] = row_data_0
                            df_6.loc[len(df_6)] = row_data_1
                            df_6.loc[len(df_6)] = row_data_2
                        elif NUM_PLM == 7:
                            df_7.loc[len(df_7)] = row_data_0
                            df_7.loc[len(df_7)] = row_data_1
                            df_7.loc[len(df_7)] = row_data_2

                        # # Adding another row using a different method
                        # df.loc[len(df.index)] = ['Value4', 'Value5', 'Value6']

                        # # Display the DataFrame
                        # print(df)

                        # print(file_path)
                        # print(NK, WEIGHT_DECAY, INHERIT_WEIGHT, SAMPLE_SELECTION)
                        print(NK, NUM_PLM, MODEL, TASK, SEED, SAMPLE_SELECTION, WEIGHT_DECAY, INHERIT_WEIGHT,)
                        print(test_acc_result['first'])
                        print(test_acc_result['max'])
                        print()

    df_1.to_csv("./results/csv/1_plm_accumulate.csv")
    df_2.to_csv("./results/csv/2_plm_accumulate.csv")
    df_3.to_csv("./results/csv/3_plm_accumulate.csv")
    df_4.to_csv("./results/csv/4_plm_accumulate.csv")
    df_5.to_csv("./results/csv/5_plm_accumulate.csv")
    df_6.to_csv("./results/csv/6_plm_accumulate.csv")
    df_7.to_csv("./results/csv/7_plm_accumulate.csv")


def group_mean_calculation_accumulate():
    for num_plm in range(1,8):
        file_name = f'./results/csv/{num_plm}_plm_accumulate.csv'
        df = pd.read_csv(file_name, index_col=False)
        # df = df.replace(-2, np.nan) # .dropna(how='any')
        mean_result = df.groupby(['STM', 'TASK', 'NUM_PLM', 'NK', 'PLM_NAME', 'TOTAL_NK', 'SAMPLE_SELECTION', 'INHERIT_WEIGHT', 'WEIGHT_DECAY', 'BETA', 'KD_TEMPERATURE', 'KD_ALPHA', 'SYN_DATA', 'PROMPTING', 'PROMPTING', 'EMBEDDING_TRANSFORMER', 'MODEL_WEIGHTING', 'MODEL_WEIGHTING_TEMPERATURE', 'FEW_SHOT_K', 'FEW_SHOT_SAMPLE_POOL_FOR_EACH', 'FEW_SHOT_SAMPLE_AMBIGUOUS_RATIO', 'TRAIN_METHOD']).mean()
        print(mean_result)
        mean_result.to_csv(f'./results/csv/{num_plm}_plm_accumulate_mean.csv')


def plot_increase_curve():
    for num_plm in range(1,8):
        file_name = f'./results/csv/{num_plm}_plm_accumulate_mean.csv'
        df = pd.read_csv(file_name, index_col=False)
        label_list = ['gpt-4-turbo-preview', 'gpt-3.5-turbo-instruct', 'gpt2-xl', 'llama-2-7b-chat-hf', 'vicuna-7b-1.5v', 'opt-6.7b', 'chatglm3-6b-base', 'flan-t5-xl', 'all-one-hot']
        linestyle_list = ['-'] * (len(label_list)-1) + [':']
        for index, row in df.iterrows():
            data_list = []
            for plm_name in label_list:
                _data_list = [float(row[plm_name+f'_step{i_step}']) for i_step in range(5)]
                data_list.append(_data_list)
            plt.figure(figsize=(8,6))
            plot_x = [i for i in range(5)]
            for _i, (_data, _label) in enumerate(zip(data_list, label_list)):
                plt.plot(plot_x, _data, label=_label, linestyle=linestyle_list[_i])
            plt.title(f'change with accumulation')
            plt.legend()
            plt.tight_layout()
            save_dir = f"./figure/accumulated/{row['STM']}/{row['TASK']}/{row['NK']}_{row['PLM_NAME']}/{row['SAMPLE_SELECTION']}_{row['INHERIT_WEIGHT']}_{row['WEIGHT_DECAY']}/{row['BETA']}_{row['KD_TEMPERATURE']}_{row['KD_ALPHA']}_{row['SYN_DATA']}/{row['FEW_SHOT_K']}_{row['FEW_SHOT_SAMPLE_POOL_FOR_EACH']}_{row['FEW_SHOT_SAMPLE_AMBIGUOUS_RATIO']}/"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(f"{save_dir}{row['TRAIN_METHOD']}_accumulate.png", dpi=300)

if __name__ == "__main__":
    # get_max_in_iter()

    # get_mean_in_iter()

    # get_flip_results_in_iter()
    # group_mean_calculation()

    get_accumualte_results_in_iter()
    # group_mean_calculation_accumulate()
    # plot_increase_curve()