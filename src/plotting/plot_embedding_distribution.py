import os, sys
import argparse
import json
import random
import re
import copy
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertForSequenceClassification, BertTokenizer
from sentence_transformers import SentenceTransformer

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
# from scipy.linalg import sqrtm
from scipy.stats import entropy, gaussian_kde
from scipy.spatial.distance import cdist
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from plot_utils import *
from plot_dynamics_v2 import plot_data_map
# from src.utils.basic_utils import merge_all_dataset

EPSILON = 1e-30


# Load and preprocess data from the jsonl file
class TokenizedDataset(Dataset):
    def __init__(self, file_path='', text_column='text', label_column='label', index_column='idx', tokenizer=None, max_length=512, device='cpu', max_sample=-1, small_dataset_shuffle=False):
        self.text = []
        self.ids = []
        self.attention_mask = []
        self.label = []
        self.idx = []
        if file_path == '':
            self.ids = torch.tensor([self.ids],dtype=torch.int64).to(device)
            self.attention_mask = torch.tensor([self.attention_mask],dtype=torch.int64).to(device)
            self.label = torch.tensor(self.label,dtype=torch.int64).to(device)
            self.idx = torch.tensor(self.idx,dtype=torch.int64).to(device)
        else:
            with open(file_path, 'r') as file:
                counter = 0
                lines = []
                # with open(file_path, 'r') as file:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                    for line in file:
                        lines.append(line)
                if small_dataset_shuffle == True:
                    random.shuffle(lines)
                for line in lines:
                    item = json.loads(line.strip())
                    text = item[text_column]
                    label = item[label_column]  # Assuming your jsonl file contains a 'label' field
                    idx = item[index_column]
                    tokenized = tokenizer.encode_plus(
                        text,
                        add_special_tokens=True,
                        max_length=max_length,
                        padding='max_length',
                        truncation=True,
                        return_attention_mask=True,
                        return_tensors='pt',
                    )
                    self.text.append(text)
                    self.ids.append(tokenized['input_ids'])
                    self.attention_mask.append(tokenized['attention_mask'])
                    self.label.append(label)
                    self.idx.append(idx)
                    counter += 1
                    if max_sample > 0 and counter == max_sample:
                        break
            # print("in TokenizedDataset init", self.text[0], self.ids[0], self.attention_mask[0], self.label[0], self.idx[0])
            # print("in TokenizedDataset init", self.text[-1], self.ids[-1], self.attention_mask[-1], self.label[-1], self.idx[-1])
            # print(self.ids)
            # print(self.label)
            # print(self.ids[-1].dtype)
            # self.ids = torch.stack(self.ids).squeeze().to(device)
            # self.attention_mask = torch.stack(self.attention_mask).squeeze().to(device)
            # self.label = torch.tensor(self.label).long().to(device)
            # self.idx = torch.tensor(self.idx).long().to(device)
            self.ids = torch.stack(self.ids).squeeze()
            self.attention_mask = torch.stack(self.attention_mask).squeeze()
            self.label = torch.tensor(self.label).long()
            self.idx = torch.tensor(self.idx).long()
        # print(self.ids.shape, self.attention_mask.shape, self.label.shape, self.idx.shape)
        # print(self.ids.dtype, self.attention_mask.dtype, self.label.dtype, self.idx.dtype)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        return self.ids[index], self.attention_mask[index], self.label[index], self.idx[index]

SYN_DATA_PATH = 'data_new/'
MODEL_PATH = {
    'bert-base-uncased': "../../../.cache/huggingface/hub/models--bert-base-uncased/snapshots/1dbc166cf8765166998eff31ade2eb64c8a40076/",
    'distilbert-base-uncased': "../../../.cache/huggingface/hub/models--distilbert-base-uncased/snapshots/6cdc0aad91f5ae2e6712e91bc7b65d1cf5c05411/",
    'sentence-t5-base': "../../../.cache/huggingface/hub/sentence-transformers--sentence-t5-base/",
}

def file_choose(num_samples):
    bins = [0,10,1000,10000,20000,50000,100000,200000,500000,1000000]
    bins = [0,10000,20000,50000]
    file_samples = -1
    for j in range(1,len(bins)):
        if bins[j-1] < num_samples <= bins[j]:
            file_samples = bins[j]
    assert file_samples > 0, "too many samples, haven't generated enough"
    print(f"require #{num_samples}, use file under #{file_samples}")
    return file_samples 


def set_seed(seed = 42) -> None:
    """Set RNG seeds for python's `random` module, numpy and torch"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(args, batch_size=32, backward_batch_size=1000, device="cpu", gold_data_path='data/', syn_data_path='data_new/', vectors=None, use_tree=False, num_use_samples_inner=100, num_use_samples_outer=100, shuffle_train=True):
    if args.small_model_name.upper() == "LSTM":
        return load_iters_lstm(args, batch_size, backward_batch_size, device, gold_data_path, syn_data_path, vectors, use_tree, num_use_samples_inner, num_use_samples_outer, shuffle_train)
    elif 'bert' in args.small_model_name.lower():
        return load_iters_bert(args, batch_size, backward_batch_size, device, gold_data_path, syn_data_path, vectors, use_tree, num_use_samples_inner, num_use_samples_outer, shuffle_train)


def load_iters_lstm(args, batch_size=32, backward_batch_size=1000, device="cpu", gold_data_path='data', syn_data_path='data', vectors=None, use_tree=False, num_use_samples_inner=100, num_use_samples_outer=100, shuffle_train=True):
    TEXT = data.Field(batch_first=True, include_lengths=True, lower=True, unk_token='<unk>')
    LABEL = data.LabelField(batch_first=True, use_vocab=False) # , use_vocab=True
    INDEX = data.RawField()
    fields = {'C': ('text', TEXT),
              'Y': ('label', LABEL),
              'idx': ('idx', INDEX)}

    # if args.query_input_file == None:
    #     args.query_input_file = []
    #     for i in range(args.len_LLM):
    #         args.query_input_file.append((f'{SYN_DATA_PATH}{args.task_name}/mix/{args.llms[i]}/{file_choose(args.separate_num_use_samples_inner[i])}/train.jsonl') if args.mix else (f'{SYN_DATA_PATH}{args.task_name}/{args.llms[i]}/{file_choose(args.num_use_samples_inner[i])}/train.jsonl'))
    #         print(f"args.query_input_file[-1]={args.query_input_file[-1]}")
 
    train_data_list = []
    small_train_data_list = []
    small_valid_data_list = []
    all_data_examples = []
    for i in range(args.len_LLM):
        if args.steps == 0:
            train_data_path = (f'{SYN_DATA_PATH}{args.task_name}/mix/{args.llms[i]}/{file_choose(args.separate_num_use_samples_inner[i])}/') if args.mix else (f'{SYN_DATA_PATH}{args.task_name}/{args.llms[i]}/{file_choose(args.num_use_samples_inner[i])}/')
        else:
            assert args.mix == False, "Setting error, --mix should be False with --steps > 0, but now --mix is True"
            train_data_path = f'{SYN_DATA_PATH}{args.model_name_sample}/{args.small_model_name}/{args.fuse_dataset_sample_selection}_KD{args.kd_slm}_FuseDataset{args.fuse_dataset}/fewshotK{args.gen_few_shot_k}_{args.gen_few_shot_pool_size}_{args.gen_few_shot_ambiguous_ratio}/{args.seed}/{args.llms[i]}/{args.num_use_samples_inner[i]}_{args.num_use_samples_init[i]}_{args.num_use_samples_each_step_extend[i]}/'
        train_data, _ = data.TabularDataset.splits(
            path=train_data_path,
            train='train.jsonl',
            test='train.jsonl',
            # train='test.jsonl',
            # test='test.jsonl',
            format='json',
            fields=fields,
            filter_pred=lambda ex: ex.label != '-'  # filter the example which label is '-'(means unlabeled)
        )
        traindataset = train_data.examples[:args.num_use_samples_inner[i]]
        for _i, data_item in enumerate(traindataset):
            data_item.idx = _i
        # for data_item in traindataset:
        #     print("in construct, total data", data_item.idx, data_item.text, data_item.label)
        # small_traindataset, small_validationdataset = copy.deepcopy(traindataset[:int(args.num_use_samples_inner[i]*args.train_ratio)]), copy.deepcopy(traindataset[int(args.num_use_samples_inner[i]*args.train_ratio):])
        all_data_examples += traindataset
        shuffled_traindataset = copy.deepcopy(traindataset)
        random.shuffle(shuffled_traindataset)
        # train-valid split
        small_traindataset, small_validationdataset = copy.deepcopy(shuffled_traindataset[:int(args.num_use_samples_inner[i]*args.train_ratio)]), copy.deepcopy(shuffled_traindataset[int(args.num_use_samples_inner[i]*args.train_ratio):])
        small_traindataset_idx = []
        for _i, data_item in enumerate(small_traindataset):
            small_traindataset_idx.append(data_item.idx)
            data_item.idx = _i
        # for data_item in small_traindataset:
        #     print("in construct, small data", data_item.idx, data_item.text, data_item.label)
        # random.shuffle(small_traindataset)
        for _i, data_item in enumerate(small_validationdataset):
            data_item.idx = _i
        # ############## construct all data and separate as train and test ##############
        train_data = data.Dataset(traindataset, train_data.fields)
        train_data_list.append(train_data)
        small_train_data = data.Dataset(small_traindataset, train_data.fields)
        small_train_data_list.append(small_train_data)
        small_valid_data = data.Dataset(small_validationdataset, train_data.fields)
        small_valid_data_list.append(small_valid_data)
        # ############## construct all data and separate as train and test ##############

        # save original text of small train dataset
        args.samples_text[i] = []
        total_sample_text = []
        with jsonlines.open(f'{train_data_path}train.jsonl', 'r') as reader:
            for json_obj in reader:
                total_sample_text.append(json_obj['C'])
        for _idx in small_traindataset_idx:
            args.samples_text[i].append(total_sample_text[_idx])
        print(f"[debug] sample_text has length {len(args.samples_text[i])}")

    fields_dev = {'text': ('text', TEXT),
                  'label': ('label', LABEL),
                  'idx': ('idx', INDEX)}
    dev_data, test_data = data.TabularDataset.splits(
        path=gold_data_path,
        validation='train.jsonl',
        test='test.jsonl',
        # test='test_small.jsonl',
        format='json',
        fields=fields_dev,
        # fields=fields,
        filter_pred=lambda ex: ex.label != '-'  # filter the example which label is '-'(means unlabeled)
    )

    print(f"[debug] args.use_dev_outer={args.use_dev_outer}, args.subset_outer={args.subset_outer}")
    if args.use_dev_outer:
        dev_data_all = data.Dataset(dev_data.examples, dev_data.fields)
        dev_data = data.Dataset(dev_data.examples[:num_use_samples_outer], dev_data.fields)
    else:
        if args.subset_outer: # currently use this one
            indices = np.random.choice(list(range(args.sample_each_llm[-1])), int(num_use_samples_outer//args.len_LLM), replace=False)
            print(f"[debug] len(train_data.examples)={len(train_data.examples)}")
            data_sample_list = []
            for i in range(args.len_LLM):
                data_sample_list = data_sample_list + [train_data_list[i].examples[ix] for ix in indices]
            dev_data = data.Dataset(data_sample_list, train_data.fields)
        else:
            dev_data=train_data
        dev_data_all=train_data

    print(f'[debug] len(all_data_examples) for train data is {len(all_data_examples)}')
    all_data_examples = all_data_examples + dev_data.examples + test_data.examples
    print(f'[debug] len(all_data_examples) for all data is {len(all_data_examples)}')
    all_data = data.Dataset(all_data_examples, train_data.fields)
    if vectors is not None:
        TEXT.build_vocab(all_data, vectors=vectors, unk_init=torch.Tensor.normal_)
    else:
        TEXT.build_vocab(all_data, max_size=500000)
    # print(f"[debug] see TEXT after build_vocab {TEXT}")
    LABEL.build_vocab(all_data)
    print(f"[debug] see LABEL after build_vocab {LABEL}")

    concat_of_data = train_data_list + train_data_list + small_train_data_list + small_valid_data_list + [dev_data]
    concat_of_data = tuple(concat_of_data)
    concat_of_batch_size = [batch_size]*args.len_LLM + [backward_batch_size]*args.len_LLM + [batch_size]*args.len_LLM + [batch_size]*args.len_LLM + [batch_size]
    concat_of_batch_size = tuple(concat_of_batch_size)

    iters = BucketIterator.splits(
        concat_of_data,
        batch_sizes=concat_of_batch_size,
        device=device,
        sort_key=lambda x: len(x.text),
        sort_within_batch=True,
        repeat=False,
        shuffle=shuffle_train,
    )
    iters = list(iters)
    train_iter_list = iters[0:args.len_LLM]
    train_iter_backward_list = iters[args.len_LLM:2*args.len_LLM]
    small_train_iter_list = iters[2*args.len_LLM:3*args.len_LLM]
    small_valid_iter_list = iters[3*args.len_LLM:4*args.len_LLM]
    dev_iter = iters[-1]

    test_iter = Iterator(test_data,
                         batch_size=batch_size,
                         device=device,
                         sort=False,
                         sort_within_batch=False,
                         repeat=False,
                         shuffle=False)
    
    print(f'[debug] before exiting load iter: len(train_iter_list)={len(train_iter_list)}, len(train_data_list)={len(train_data_list)}')
    # return train_iter_list, small_train_iter_list, small_valid_iter_list, train_iter_backward_list, dev_iter, test_iter, TEXT, LABEL, train_data_list, small_train_data_list, small_valid_data_list, dev_data_all
    return train_data_list, train_iter_list


def load_iters_bert(args, batch_size=32, backward_batch_size=1000, device="cpu", gold_data_path='data', syn_data_path='data', vectors=None, use_tree=False, num_use_samples_inner=100, num_use_samples_outer=100, shuffle_train=True):
    init_token = args.tokenizer.cls_token # [CLS]   
    eos_token = args.tokenizer.sep_token # [SEP]
    pad_token = args.tokenizer.pad_token # [PAD]
    unk_token = args.tokenizer.unk_token # [UNK]
    init_token_idx = args.tokenizer.convert_tokens_to_ids(init_token) # 101
    eos_token_idx = args.tokenizer.convert_tokens_to_ids(eos_token) # 102
    pad_token_idx = args.tokenizer.convert_tokens_to_ids(pad_token) # 0
    unk_token_idx = args.tokenizer.convert_tokens_to_ids(unk_token) # 100


    train_data_list = []
    small_train_data_list = []
    small_valid_data_list = []
    all_data_examples = []
    for i in range(args.len_LLM):
        print("train_dataset for", args.llms[i])
        print(f"{SYN_DATA_PATH=}")
        # print("train_dataset for", args.llms[i], "file for data is", (file_choose(args.separate_num_use_samples_inner[i] if args.mix else file_choose(args.num_use_samples_inner[i]))))
        if SYN_DATA_PATH == 'data_new/':
            train_data_path = f'{SYN_DATA_PATH}{args.task_name}/{args.llms[i]}/{file_choose(args.num_use_samples_inner[i])}/train.jsonl'
        else:
            train_data_path = f'{SYN_DATA_PATH}{args.llms[i]}/1000_200_4_unbalance_temp1.0/train.jsonl' # accumulate-adjust-2-2
            # train_data_path = f'{SYN_DATA_PATH}{args.llms[i]}/6000_1200_4_unbalance_temp1.0/train.jsonl' # accumulate-adjust-2-2
            # train_data_path = f'{SYN_DATA_PATH}{args.llms[i]}/6000_1200_4_unbalance_temp3/train.jsonl' # accumulate-adjust-2-2
            # train_data_path = f'{SYN_DATA_PATH}{args.llms[i]}/1000_200_200/train.jsonl' # accumulate-adjust-2-2
            # train_data_path = f'{SYN_DATA_PATH}{args.llms[i]}/100_20_20/train.jsonl' # accumulate-adjust-2-2
        print(f"{train_data_path=}")
        train_data = TokenizedDataset(
            file_path=train_data_path,
            text_column='C',
            label_column='Y',
            index_column='idx',
            tokenizer=args.tokenizer,
            max_length=512,
            # device=args.device,
            max_sample=-1,
        )
        # train_data.idx = torch.tensor([_i for _i in range(args.num_use_samples_inner[i])]).long().to(args.device)
        train_data.idx = torch.tensor([_i for _i in range(len(train_data))]).long()
        print(f"loading {len(train_data)} samples ...")
        
        # indices = list(range(args.num_use_samples_inner[i]))
        # random.shuffle(indices)
        # train_valid_pivot_point = int(args.num_use_samples_inner[i]*args.train_ratio)
        
        # # ############## separate as train and test ##############
        # # train-valid split
        # small_train_data = TokenizedDataset(
        #     file_path=(''),
        # )
        # small_train_data.text = [copy.deepcopy(train_data.text[ix]) for ix in indices[:train_valid_pivot_point]]
        # small_train_data.ids = copy.deepcopy(train_data.ids[indices[:train_valid_pivot_point]])
        # small_train_data.attention_mask = copy.deepcopy(train_data.attention_mask[indices[:train_valid_pivot_point]])
        # small_train_data.label = copy.deepcopy(train_data.label[indices[:train_valid_pivot_point]])
        # # small_train_data.idx = torch.tensor([_i for _i in range(train_valid_pivot_point)]).long().to(args.device)
        # small_train_data.idx = torch.tensor([_i for _i in range(train_valid_pivot_point)]).long()
        
        # small_valid_data = TokenizedDataset(
        #     file_path=(''),
        # )
        # small_valid_data.text = [copy.deepcopy(train_data.text[ix]) for ix in indices[train_valid_pivot_point:]]
        # small_valid_data.ids = copy.deepcopy(train_data.ids[indices[train_valid_pivot_point:]])
        # small_valid_data.attention_mask = copy.deepcopy(train_data.attention_mask[indices[train_valid_pivot_point:]])
        # small_valid_data.label = copy.deepcopy(train_data.label[indices[train_valid_pivot_point:]])
        # # small_valid_data.idx = torch.tensor([_i for _i in range(args.num_use_samples_inner[i]-train_valid_pivot_point)]).long().to(args.device)
        # small_valid_data.idx = torch.tensor([_i for _i in range(args.num_use_samples_inner[i]-train_valid_pivot_point)]).long()

        train_data_list.append(train_data)
        # small_train_data_list.append(small_train_data)
        # small_valid_data_list.append(small_valid_data)
        # # ############## separate as train and test ##############
        
        # # save original text of small train dataset
        # args.samples_text[i] = [copy.deepcopy(text) for text in small_train_data.text]
        # print(f"[debug] sample_text has length {len(args.samples_text[i])}")

    print("test dataset")
    if os.path.exists(f'{SYN_DATA_PATH}gold/'):
        test_data_list = []
        for root, dirs, files in os.walk(f'{SYN_DATA_PATH}gold/'):
            for file in files:
                print(f"{file=}")
                if file.endswith('train.jsonl'):
                    gold_file_path = os.path.join(root, file)
                    _test_data = TokenizedDataset(
                        file_path=gold_file_path,
                        text_column='C',
                        label_column='Y',
                        index_column='idx',
                        tokenizer=args.tokenizer,
                        device=args.device,
                        max_length=512,
                        max_sample=-1, # use all that is provided in the dataset file
                        # max_sample=-1 # use all that is provided in the dataset file
                        small_dataset_shuffle=True,
                    )
                    test_data_list.append(_test_data)
        test_data = merge_all_dataset(args, test_data_list, max_sample_count_for_total=-1)
    else:
        test_data = TokenizedDataset(
            file_path=(gold_data_path+'test.jsonl'),
            # file_path=(gold_data_path+'test_small.jsonl'),
            text_column='text',
            label_column='label',
            index_column='idx',
            tokenizer=args.tokenizer,
            device=args.device,
            max_length=512,
            max_sample=args.gold_data_num, # use all that is provided in the dataset file
            # max_sample=-1 # use all that is provided in the dataset file
            small_dataset_shuffle=True,
        )

    if args.consider_real:
        train_data_list.append(test_data)

    train_iter_list = [DataLoader(dataset, batch_size=batch_size, shuffle=False) for dataset in train_data_list]
    # test_iter = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    print(f'[debug] before exiting load iter: len(train_iter_list)={len(train_iter_list)}, len(train_data_list)={len(train_data_list)}')
    # return train_iter_list, small_train_iter_list, small_valid_iter_list, train_iter_backward_list, dev_iter, test_iter, train_data_list, small_train_data_list, small_valid_data_list, dev_data_all
    return train_data_list, train_iter_list


def get_embedding(args, model, train_iter):
    if 'sentence' in args.small_model_name:
        model = model.to(args.device)
        # # sentences = ["This is an example sentence", "Each sentence is converted"]
        # model = SentenceTransformer('sentence-transformers/stsb-roberta-base-v2')
        embedding_list = model.encode(train_iter.dataset.text)
        # # print(embedding_list)
        print(f"{type(embedding_list)=}")
        # embedding_list = np.asarray(embedding_list.detach().cpu())
        label_list = np.asarray(train_iter.dataset.label)
    else:
        model_copy = copy.deepcopy(model)
        model_copy.to(args.device)
        # print(f'a model on gpu, {torch.cuda.memory_reserved()/1024/1024=}M, {torch.cuda.memory_allocated()/1024/1024=}M')
        # print(f"{theta.shape=}, {type(theta)=}")
        model_copy.train()
        embedding_list = []
        label_list = []
        for batch in tqdm(train_iter):
            if args.small_model_name.upper() == 'LSTM':
                (inputs, lens), labels = batch.text, batch.label
                idx = batch.idx
            elif 'bert' in args.small_model_name.lower():
                inputs, attention_mask, labels, idx = batch
                inputs = inputs.to(args.device)
                attention_mask = attention_mask.to(args.device)
                labels = labels.to(args.device)
                idx = idx.to(args.device)

            if args.small_model_name.upper() == 'LSTM':
                output = model_copy(inputs, lens)
            elif 'bert' in args.small_model_name.lower():
                model_output = model_copy(inputs, attention_mask=attention_mask)
                # output = model_output.logits
                embeddings = model_output.last_hidden_state
                embeddings = torch.mean(embeddings, dim=1)  # mean_pooling_embeddings, Shape: [batch_size, hidden_size]
            embedding_list.append(embeddings.detach().cpu().numpy())
            label_list.append(labels.cpu().numpy())
        model_copy.to("cpu")
        embedding_list = np.concatenate(embedding_list, axis=0)
        label_list = np.concatenate(label_list, axis=0)
    # return embedding_list
    return embedding_list, label_list


def plot_labeled_distribution(args, embeddings_2d, embeddings, labels, embeddings_label, label_unique_values, counts):
    # Define colors and markers for each class
    # colors = ['r', 'g', 'b']  # Red, Green, Blue
    # markers = ['o', '^', 's']  # Circle, Triangle, Square
    colors = [
        (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),  # Blue
        (1.0, 0.4980392156862745, 0.054901960784313725),  # Orange
        (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),  # Green
        (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),  # Red
        (0.5803921568627451, 0.403921568627451, 0.7411764705882353),  # Purple
        (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),  # Brown
        (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),  # Pink
        (0.4980392156862745, 0.4980392156862745, 0.4980392156862745),  # Grey
        (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),  # Yellow
        (0.09019607843137255, 0.7450980392156863, 0.8117647058823529),  # Cyan
        (0.9843137254901961, 0.8666666666666667, 0.4941176470588235),  # Wheat
        (0.4627450980392157, 0.911764705882353, 0.4823529411764706),  # Lightgreen
    ]
    markers = ['o', '^', 's', 'v', 'D', 'H', '*', '+', 'x', '1', '2', '3', '4']

    # Plot the 2D scatter plot, with/without test dataset, consider label, all label in 1 figure
    # fig = plt.figure(figsize=(8, 6))
    LLM_names_mapping_dict = {'gpt2-xl':'GPT2', 'llama-2-7b-chat-hf':'Llama2', 'vicuna-7b-1.5v':'Vicuna', 'opt-6.7b':'OPT', 'chatglm3-6b-base':'ChatGLM3', 'flan-t5-xl':'Flan-T5', 'gpt-3.5-turbo-instruct':'GPT-3.5', 'gpt-4-turbo-preview':'GPT-4', 'real':'real'}
    LLM_names = [LLM_names_mapping_dict[llm] for llm in args.llms] + ['real']
    nrow = 1
    ncolumn = (args.len_LLM+1 if args.consider_real else args.len_LLM)
    fig, axs = plt.subplots(nrow, ncolumn, figsize=(18 if args.consider_real else 16, 3), sharex=True, sharey=True)
    for i in range((args.len_LLM+1 if args.consider_real else args.len_LLM)):
        # for ir, label in enumerate(label_unique_values):
        # temp_embeddings_2d = np.array(embeddings_2d[args.accumulate_sampels[i]:args.accumulate_sampels[i]+1000, :])
        # temp_labels = labels[args.accumulate_sampels[i]:args.accumulate_sampels[i]+1000]
        temp_embeddings_2d = embeddings_2d[args.accumulate_sampels[-1][i]:args.accumulate_sampels[-1][i+1], :]
        temp_labels = labels[args.accumulate_sampels[-1][i]:args.accumulate_sampels[-1][i+1]]
        # class_mask = (temp_labels==0).astype(np.int64)
        # class_mask = np.nonzero(class_mask)[0]
        # for ic in class_mask[110:120]:
        #     print(int(ic))
        color_list = [colors[1-int(ic)] for ic in temp_labels]
        marker_list = [markers[1-int(ic)] for ic in temp_labels]
        # print(type(class_mask), class_mask.shape, class_mask.dtype)
        # print(temp_embeddings_2d[class_mask, 0])
        axs[i].scatter(temp_embeddings_2d[:, 0], temp_embeddings_2d[:, 1], color=color_list, marker='o',
                    # label=(f'{LLM_names[i]} if i<args.len_LLM else 'Golden Data'), 
                    alpha=0.3) #+f'\nclass {ir}'
        axs[i].set_title(f'{LLM_names[i]}', fontsize=14)
        # axs[i].legend(loc='upper center')
    # fig.suptitle('t-SNE Visualization of Embeddings for syndataset generated by different LLMs')
    # plt.xlabel('Dimension 1')
    # plt.ylabel('Dimension 2')
    # plt.legend()
    plt.tight_layout()
    if not os.path.exists(f'./figure/distribution/2D/{args.folder_name}/'):
        os.makedirs(f'./figure/distribution/2D/{args.folder_name}/')
    print(f'./figure/distribution/2D/{args.folder_name}/{save_type}_{f"with{args.gold_data_num}" if args.consider_real else "without"}test_withLabel.png')
    plt.savefig(f'./figure/distribution/2D/{args.folder_name}/{save_type}_{"with" if args.consider_real else "without"}test_withLabel.png',dpi=200)

    for i_step in range(1,args.steps+1):
        plt.clf()
        fig, axs = plt.subplots(nrow, ncolumn, figsize=(18 if args.consider_real else 16, 3), sharex=True, sharey=True)
        for i in range((args.len_LLM+1 if args.consider_real else args.len_LLM)):
            # for ir, label in enumerate(label_unique_values):
            if i < args.len_LLM:
                # temp_embeddings_2d = np.array(embeddings_2d[args.accumulate_sampels[i]:args.accumulate_sampels[i]+int(args.num_use_samples_inner[i]*(i_step/(args.steps+1))), :])
                # temp_labels = labels[args.accumulate_sampels[i]:args.accumulate_sampels[i]+int(args.num_use_samples_inner[i]*(i_step/(args.steps+1)))]
                temp_embeddings_2d = np.array(embeddings_2d[args.accumulate_sampels[-1][i]:args.accumulate_sampels[-1][i]+args.step_sample_count[i_step-1][i], :])
                temp_labels = labels[args.accumulate_sampels[-1][i]:args.accumulate_sampels[-1][i]+args.step_sample_count[i_step-1][i]]
            else:
                temp_embeddings_2d = np.array(embeddings_2d[args.accumulate_sampels[-1][i]:, :])
                temp_labels = labels[args.accumulate_sampels[-1][i]:]
            # temp_embeddings_2d = embeddings_2d[args.accumulate_sampels[i]:args.accumulate_sampels[i+1], :]
            # temp_labels = labels[args.accumulate_sampels[i]:args.accumulate_sampels[i+1]]
            # class_mask = (temp_labels==0).astype(np.int64)
            # class_mask = np.nonzero(class_mask)[0]
            # for ic in class_mask[110:120]:
            #     print(int(ic))
            color_list = [colors[1-int(ic)] for ic in temp_labels]
            marker_list = [markers[1-int(ic)] for ic in temp_labels]
            # print(type(class_mask), class_mask.shape, class_mask.dtype)
            # print(temp_embeddings_2d[class_mask, 0])
            axs[i].scatter(temp_embeddings_2d[:, 0], temp_embeddings_2d[:, 1], color=color_list, marker='o',
                        # label=(f'{LLM_names[i]} if i<args.len_LLM else 'Golden Data'), 
                        alpha=0.3) #+f'\nclass {ir}'
            axs[i].set_title(f'{LLM_names[i]}', fontsize=14)
            # axs[i].legend(loc='upper center')
        # fig.suptitle('t-SNE Visualization of Embeddings for syndataset generated by different LLMs')
        # plt.xlabel('Dimension 1')
        # plt.ylabel('Dimension 2')
        # plt.legend()
        plt.tight_layout()
        if not os.path.exists(f'./figure/distribution/2D/{args.folder_name}/'):
            os.makedirs(f'./figure/distribution/2D/{args.folder_name}/')
        print(f'./figure/distribution/2D/{args.folder_name}/step{i_step}_{save_type}_{f"with{args.gold_data_num}" if args.consider_real else "without"}test_withLabel.png')
        plt.savefig(f'./figure/distribution/2D/{args.folder_name}/step{i_step}_{save_type}_{"with" if args.consider_real else "without"}test_withLabel.png',dpi=200)

    plt.clf()
    fig, axs = plt.subplots(args.steps+1, ncolumn, figsize=(18 if args.consider_real else 16, 3*(args.steps+1+0.2)), sharex=True, sharey=True)
    for i_step in range(1,args.steps+2):
        for i in range((args.len_LLM+1 if args.consider_real else args.len_LLM)):
            # for ir, label in enumerate(label_unique_values):
            if i < args.len_LLM:
                # temp_embeddings_2d = np.array(embeddings_2d[args.accumulate_sampels[i]:args.accumulate_sampels[i]+int(args.num_use_samples_inner[i]*(i_step/(args.steps+1))), :])
                # temp_labels = labels[args.accumulate_sampels[i]:args.accumulate_sampels[i]+int(args.num_use_samples_inner[i]*(i_step/(args.steps+1)))]
                _previous_step_obtain = args.step_sample_count[i_step-2][i] if i_step > 1 else 0
                temp_embeddings_2d = np.array(embeddings_2d[args.accumulate_sampels[-1][i]+_previous_step_obtain:args.accumulate_sampels[-1][i]+args.step_sample_count[i_step-1][i], :])
                temp_labels = labels[args.accumulate_sampels[-1][i]+_previous_step_obtain:args.accumulate_sampels[-1][i]+args.step_sample_count[i_step-1][i]]
            else:
                temp_embeddings_2d = np.array(embeddings_2d[args.accumulate_sampels[-1][i]:, :])
                temp_labels = labels[args.accumulate_sampels[-1][i]:]
            # temp_embeddings_2d = embeddings_2d[args.accumulate_sampels[i]:args.accumulate_sampels[i+1], :]
            # temp_labels = labels[args.accumulate_sampels[i]:args.accumulate_sampels[i+1]]
            # class_mask = (temp_labels==0).astype(np.int64)
            # class_mask = np.nonzero(class_mask)[0]
            # for ic in class_mask[110:120]:
            #     print(int(ic))
            color_list = [colors[1-int(ic)] for ic in temp_labels]
            marker_list = [markers[1-int(ic)] for ic in temp_labels]
            # print(type(class_mask), class_mask.shape, class_mask.dtype)
            # print(temp_embeddings_2d[class_mask, 0])
            axs[i_step-1][i].scatter(temp_embeddings_2d[:, 0], temp_embeddings_2d[:, 1], color=color_list, marker='o',
                        # label=(f'{LLM_names[i]} if i<args.len_LLM else 'Golden Data'), 
                        alpha=0.3) #+f'\nclass {ir}'
            axs[i_step-1][i].set_title(f'{LLM_names[i]}', fontsize=14)
            # axs[i].legend(loc='upper center')
        # fig.suptitle('t-SNE Visualization of Embeddings for syndataset generated by different LLMs')
        # plt.xlabel('Dimension 1')
        # plt.ylabel('Dimension 2')
        # plt.legend()
        plt.tight_layout()
        if not os.path.exists(f'./figure/distribution/2D/{args.folder_name}/'):
            os.makedirs(f'./figure/distribution/2D/{args.folder_name}/')
        print(f'./figure/distribution/2D/{args.folder_name}/bystep_{save_type}_{f"with{args.gold_data_num}" if args.consider_real else "without"}test_withLabel.png')
        plt.savefig(f'./figure/distribution/2D/{args.folder_name}/bystep_{save_type}_{"with" if args.consider_real else "without"}test_withLabel.png',dpi=200)


    # # Plot the 2D scatter plot, with test dataset, consider label
    # # fig = plt.figure(figsize=(8, 6))
    # nrow = len(label_unique_values)
    # ncolumn = args.len_LLM+1
    # fig, axs = plt.subplots(nrow, ncolumn, figsize=(18, 6), sharex=True, sharey=True)
    # for i in range(args.len_LLM+1):
    #     for ir, label in enumerate(label_unique_values):
    #         temp_embeddings_2d = np.array(embeddings_2d[args.accumulate_sampels[i]:args.accumulate_sampels[i]+1000, :])
    #         temp_labels = labels[args.accumulate_sampels[i]:args.accumulate_sampels[i]+1000]
    #         # temp_embeddings_2d = embeddings_2d[args.accumulate_sampels[i]:args.accumulate_sampels[i+1], :]
    #         # temp_labels = labels[args.accumulate_sampels[i]:args.accumulate_sampels[i+1]]
    #         class_mask = (temp_labels==label).astype(np.int64)
    #         class_mask = np.nonzero(class_mask)[0]
    #         # print(type(class_mask), class_mask.shape, class_mask.dtype)
    #         # print(temp_embeddings_2d[class_mask, 0])
    #         axs[ir][i].scatter(temp_embeddings_2d[class_mask, 0], temp_embeddings_2d[class_mask, 1], c=colors[ir], marker=markers[ir],
    #                     label=(f'LLM {args.llms[i]}' if i<args.len_LLM else 'Golden Data')+f'\nclass {ir}', alpha=0.3)
    #         # axs[ir][i].set_title()
    #         axs[ir][i].legend(loc='upper center')
    # fig.suptitle('t-SNE Visualization of Embeddings for syndataset generated by different LLMs')
    # # plt.xlabel('Dimension 1')
    # # plt.ylabel('Dimension 2')
    # # plt.legend()
    # plt.tight_layout()
    # plt.savefig(f'./figure/distribution/2D/withTest_withLabel_{args.model_name_sample}.png',dpi=200)

    # # Plot the 3D scatter plot
    # fig = plt.figure(figsize=(8, 6))
    # ax = fig.add_subplot(111, projection='3d')
    # for i in range(args.len_LLM):
    #     # mask = np.range(args.accumulate_sampels[i], args.accumulate_sampels[i+1])
    #     # plt.scatter(embeddings_2d[args.accumulate_sampels[i]:args.accumulate_sampels[i+1], 0], embeddings_2d[args.accumulate_sampels[i]:args.accumulate_sampels[i+1], 1], [i]*args.num_use_samples_inner[i], c=colors[i], marker=markers[i],
    #     #             label=f'LLM {args.llms[i]}', alpha=0.3)

    #     # ax.scatter(embeddings_2d[args.accumulate_sampels[i]:args.accumulate_sampels[i+1], 0], embeddings_2d[args.accumulate_sampels[i]:args.accumulate_sampels[i+1], 1], np.array([i for _ in range(args.num_use_samples_inner[i])]), c=colors[i], marker=markers[i],
    #     #             label=f'LLM {args.llms[i]}', alpha=0.3)
    #     ax.scatter(embeddings_2d[args.accumulate_sampels[i]:args.accumulate_sampels[i]+1000, 0], embeddings_2d[args.accumulate_sampels[i]:args.accumulate_sampels[i]+1000, 1], np.array([i for _ in range(1000)]), c=colors[i], marker=markers[i],
    #                 label=f'LLM {args.llms[i]}', alpha=0.3)
    # plt.title('t-SNE Visualization of Embeddings for different LLMs')
    # plt.xlabel('Dimension 1')
    # plt.ylabel('Dimension 2')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(f'./figure/distribution/3D/{args.model_name_sample}.png',dpi=200)


def calculate_embedding_distance(args, embeddings_2d, embeddings, labels, embeddings_label, label_unique_values, count=8):

    # Plot the 2D scatter plot, with/without test dataset, consider label, all label in 1 figure
    # fig = plt.figure(figsize=(8, 6))
    LLM_names_mapping_dict = {'gpt2-xl':'GPT2', 'llama-2-7b-chat-hf':'Llama2', 'vicuna-7b-1.5v':'Vicuna', 'opt-6.7b':'OPT', 'chatglm3-6b-base':'ChatGLM3', 'flan-t5-xl':'Flan-T5', 'gpt-3.5-turbo-instruct':'GPT-3.5', 'gpt-4-turbo-preview':'GPT-4', 'real':'real'}
    LLM_names = [LLM_names_mapping_dict[llm] for llm in args.llms] + ['real']


    gold_embedding, gold_label = embeddings[args.accumulate_sampels[-1][-2]:args.accumulate_sampels[-1][-1], :], labels[args.accumulate_sampels[-1][-2]:args.accumulate_sampels[-1][-1]]
    gold_embedding = torch.tensor(gold_embedding)
    gold_embedding_list = [gold_embedding]
    gold_label_list = [gold_label]
    print(f"[debug] {gold_embedding.shape=}, {len(gold_label)=}")
    # for i in range((args.len_LLM+1 if args.consider_real else args.len_LLM)):
    for i in range((args.len_LLM)):
        syn_embedding = embeddings[args.accumulate_sampels[-1][i]:args.accumulate_sampels[-1][i+1], :]
        syn_embedding = torch.tensor(syn_embedding)
        syn_label = labels[args.accumulate_sampels[-1][i]:args.accumulate_sampels[-1][i+1]]
        print(f"[debug] {syn_embedding.shape=}, {len(syn_label)=}")

        furthest_sample_voting = [0.0]*len(syn_label)
        furthest_sample_voting_per_party = [[0.0]*len(syn_label) for _ in range(1)]

        # ########## each data party vote using local private data ##########
        args.real_voting_votes = 8
        for i_party, (gold_embedding, gold_label) in enumerate(zip(gold_embedding_list, gold_label_list)): # nearest_sample_voting_per_party      
            unique_gold_label = np.unique(gold_label)
            syn_embedding_each_class = {}
            for _u_label in unique_gold_label:
                syn_embedding_each_class[_u_label] = (syn_embedding[syn_label==_u_label], np.where(syn_label==_u_label)[0])
            # print(f"{syn_embedding_each_class=}")
            for _gold, _gold_label in zip(gold_embedding, gold_label):
                distances = torch.sqrt(((syn_embedding_each_class[_gold_label][0] - _gold)**2).sum(dim=-1))
                # print(f"{distances.shape}")
                # _, top_k_indices = torch.topk(distances, k=min(args.real_voting_votes,len(syn_embedding_each_class[_gold_label][1])), largest=False)
                # for _i, _indice in enumerate(top_k_indices):
                #     print(f"{_indice=} in class#{_gold_label}, which should be mapped to {syn_embedding_each_class[_gold_label][1][_indice]} in the total dataset")
                #     nearest_sample_voting_per_party[i_party][syn_embedding_each_class[_gold_label][1][_indice]] += 1/(2**_i)
                _, top_k_indices = torch.topk(distances, k=min(args.real_voting_votes,len(syn_embedding_each_class[_gold_label][1])), largest=True)
                for _i, _indice in enumerate(top_k_indices):
                    # print(f"{_indice=} in class#{_gold_label}, which should be mapped to {syn_embedding_each_class[_gold_label][1][_indice]} in the total dataset")
                    furthest_sample_voting_per_party[i_party][syn_embedding_each_class[_gold_label][1][_indice]] += 1/(2**_i)
        # ########## each data party vote using local private data ##########


        # # ########## aggregation of nearest_sample_voting_per_party to nearest_sample_voting ##########
        # SIGMA = args.voting_dp_sigma
        # # ### #
        # nearest_sample_voting_per_party = np.asarray(nearest_sample_voting_per_party)
        # for i_party in range(len(nearest_sample_voting_per_party)):
        #     nearest_sample_voting_per_party[i_party] += np.random.standard_normal(size=nearest_sample_voting_per_party[i_party].shape)*SIGMA
        # nearest_sample_voting = np.sum(nearest_sample_voting_per_party, axis=0)
        # # ### #
        # furthest_sample_voting_per_party = np.asarray(furthest_sample_voting_per_party)
        # for i_party in range(len(furthest_sample_voting_per_party)):
        #     furthest_sample_voting_per_party[i_party] += np.random.standard_normal(size=furthest_sample_voting_per_party[i_party].shape)*SIGMA
        furthest_sample_voting = np.sum(furthest_sample_voting_per_party, axis=0)
        # # ########## aggregation of nearest_sample_voting_per_party to nearest_sample_voting ##########

        furthest_sample_voting = np.asarray(furthest_sample_voting)

        total_num_classes = 2 if args.task_name=='imdb' else 5
        selected_sample_model_position_list = {"nearest": [[] for _ in range(total_num_classes)], "furthest": [[] for _ in range(total_num_classes)]}
        for i_class in range(total_num_classes):
            # ################## furthest bad samples ##################
            furthest_sample_voting_for_class = [0.0]*len(syn_label)
            furthest_sample_voting_for_class = [furthest_sample_voting[_i]*(1.0 if syn_label[_i]==i_class else 0.0) for _i in range(len(syn_label))]
            # print(f"[debug] after class masking, {furthest_sample_voting_for_class=}")
            furthest_sample_voting_for_class = np.asarray(furthest_sample_voting_for_class)
            # furthest_sample_voting_for_class += SMALL_EPSILON
            furthest_sample_voting_for_class =  furthest_sample_voting_for_class / np.sum(furthest_sample_voting_for_class)

            # ########### sample the top-<#count> samples with the highest probability value (furthest_sample_voting value) ###########
            if np.count_nonzero(furthest_sample_voting_for_class) < count:
                print(f"None zero values in furthest_sample_voting_for_class is {np.count_nonzero(furthest_sample_voting_for_class)}, < number of required in-context sample {count}")
                top_k_syn_samples_indices = np.random.choice([i for i in range(len(furthest_sample_voting_for_class))], size=count, p=furthest_sample_voting_for_class, replace=True)
            else:
                value_index_pairs = [(furthest_sample_voting_for_class[i], i) for i in range(len(furthest_sample_voting_for_class))]  
                sorted_pairs = sorted(value_index_pairs, key=lambda x: x[0], reverse=True)  
                # print(f"{count=}")
                top_k_syn_samples_indices = [pair[1] for pair in sorted_pairs[:count]] 
            print(f"{top_k_syn_samples_indices=}")
            # ########### sample the top-<#count> samples with the highest probability value (furthest_sample_voting_for_class value) ###########
            # change into [(im, ic), (im, ic), ..., (im, ic)] format
            for ic in range(len(top_k_syn_samples_indices)):
                model_idx = -1
                for im in range(args.len_LLM):
                    if args.accumulate_sampels[-1][im] <= top_k_syn_samples_indices[ic] < args.accumulate_sampels[-1][im+1]:
                        model_idx = im
                        break
                assert model_idx != -1, f"[ERROR] sample #{top_k_syn_samples_indices[ic]} not mapped into {args.accumulate_sampels[-1]}"
                selected_sample_model_position_list["furthest"][i_class].append((model_idx,(top_k_syn_samples_indices[ic]-args.accumulate_sampels[-1][model_idx]).item()))
            # ################## furthest bad samples ##################
            print(f'{i_class=}, {selected_sample_model_position_list["furthest"][i_class]=}')


def calculate_KL(args, embeddings_2d, embeddings, labels, embeddings_label, label_unique_values):
    if args.consider_real:
        # as the real sample size is small, decrease the embedding dimention
        pca = PCA(n_components=4, random_state=42)
        embeddings_40d = pca.fit_transform(embeddings)
        if not os.path.exists(f'./figure/distribution/embeddings/4d/{args.model_name_sample}/'):
            os.makedirs(f'./figure/distribution/embeddings/4d/{args.model_name_sample}/')
        # np.savez(f'./figure/distribution/embeddings/{args.model_name_sample}.npz', embeddings_2d=embeddings_2d, embeddings=embeddings, labels=labels)
        np.savez(f'./figure/distribution/embeddings/4d/{args.model_name_sample}/{save_type}_{f"with{args.gold_data_num}" if args.consider_real else "without"}test.npz', embeddings_40d=embeddings_40d, embeddings=embeddings, labels=labels) # currently without test
        embeddings = embeddings_40d
        
        embeddings = np.asanyarray(embeddings_40d)
        # embeddings = np.asanyarray(embeddings_2d)

        total_kl, within_class_kl = {}, {}
        for i_step in range(1,args.steps+2):
            # current_step_embeddings = []
            # current_step_labels = []
            # for i in range(args.len_LLM+1):
            #     if i < args.len_LLM:
            #         current_step_embeddings.append(np.array(embeddings[args.accumulate_sampels[i]:args.accumulate_sampels[i]+int(args.num_use_samples_inner[i]*(i_step/(args.steps+1))), :]))
            #         current_step_labels.append(labels[args.accumulate_sampels[i]:args.accumulate_sampels[i]+int(args.num_use_samples_inner[i]*(i_step/(args.steps+1)))])
            #     else:
            #         current_step_embeddings.append(np.array(embeddings[args.accumulate_sampels[i]:args.accumulate_sampels[i+1], :]))
            #         current_step_labels.append(labels[args.accumulate_sampels[i]:args.accumulate_sampels[i+1]])
            # current_step_embeddings = np.concatenate(current_step_embeddings, axis=0)
            # current_step_labels = np.concatenate(current_step_labels, axis=0)

            embeddings_label = [{} for _ in range((args.len_LLM+1 if args.consider_real else args.len_LLM))] # key=label, value=list of embeddings belonging to this label from different LLMs
            for ir, label in enumerate(label_unique_values):
                # print(f"{ir=}, {label=}")
                for _embeddings_label_for_each_model in embeddings_label:
                    _embeddings_label_for_each_model[label] = []
            for i in range((args.len_LLM+1 if args.consider_real else args.len_LLM)):
                # samples in the range: args.accumulate_sampels[i1]:args.accumulate_sampels[i1+1]
                # but should be the end of the list with i==args.len_LLM
                for ir, label in enumerate(label_unique_values):
                    if i < args.len_LLM:
                        # _index = (labels[args.accumulate_sampels[i]:args.accumulate_sampels[i]+int(args.num_use_samples_inner[i]*(i_step/(args.steps+1)))]==label)
                        # _embeddings_of_this_label = embeddings[args.accumulate_sampels[i]:args.accumulate_sampels[i]+int(args.num_use_samples_inner[i]*(i_step/(args.steps+1)))][_index]
                        _index = (labels[args.accumulate_sampels[-1][i]:args.accumulate_sampels[-1][i]+args.step_sample_count[i_step-1][i]]==label)
                        _embeddings_of_this_label = embeddings[args.accumulate_sampels[-1][i]:args.accumulate_sampels[-1][i]+args.step_sample_count[i_step-1][i]][_index]
                    else:
                        # _index = (labels[args.accumulate_sampels[i]:]==label)
                        # _embeddings_of_this_label = embeddings[args.accumulate_sampels[i]:][_index]
                        _index = (labels[args.accumulate_sampels[-1][i]:]==label)
                        _embeddings_of_this_label = embeddings[args.accumulate_sampels[-1][i]:][_index]
                    # print(_index, type(_index), len(_index))
                    # print(f"{ir=}, {label=}", _embeddings_of_this_label, type(_embeddings_of_this_label), len(_embeddings_of_this_label))
                    embeddings_label[i][label] = copy.deepcopy(_embeddings_of_this_label)
                # print(embeddings_label[i])

            total_kl[int(i_step-1)], within_class_kl[int(i_step-1)] = [float('inf')]*args.len_LLM, [0.0]*args.len_LLM
            # real_embeddings = embeddings[args.accumulate_sampels[-2]:args.accumulate_sampels[-1], :]
            # real_labels = labels[args.accumulate_sampels[-2]:args.accumulate_sampels[-1]]
            real_embeddings = embeddings[args.accumulate_sampels[-1][-2]:args.accumulate_sampels[-1][-1], :]
            real_labels = labels[args.accumulate_sampels[-1][-2]:args.accumulate_sampels[-1][-1]]
            for i in range(args.len_LLM):
                # ############# total_kl calculation #############
                # temp_embeddings = embeddings[args.accumulate_sampels[i]:args.accumulate_sampels[i]+int(args.num_use_samples_inner[i]*(i_step/(args.steps+1))), :]
                # temp_labels = labels[args.accumulate_sampels[i]:args.accumulate_sampels[i]+int(args.num_use_samples_inner[i]*(i_step/(args.steps+1)))]
                temp_embeddings = embeddings[args.accumulate_sampels[-1][i]:args.accumulate_sampels[-1][i]+args.step_sample_count[i_step-1][i], :]
                temp_labels = labels[args.accumulate_sampels[-1][i]:args.accumulate_sampels[-1][i]+args.step_sample_count[i_step-1][i]]
                print(f"{real_embeddings.shape=}, {temp_embeddings.shape=}")
                min_values = np.minimum(temp_embeddings.min(axis=0), real_embeddings.min(axis=0))
                max_values = np.maximum(temp_embeddings.max(axis=0), real_embeddings.max(axis=0))
                print(f"Min values: {min_values}, Max values: {max_values}")
                
                # Define the number of grid points for each dimension
                num_points = 32  # You can adjust this to control the resolution of the grid
                # # Create a grid that spans from min_values to max_values
                # x = np.linspace(min_values[0], max_values[0], num_points)
                # y = np.linspace(min_values[1], max_values[1], num_points)
                # # Create 2D grid
                # X, Y = np.meshgrid(x, y)
                # # Stack the grid points into a 2D array of shape (2, num_points * num_points)
                # grid_points = np.vstack([X.ravel(), Y.ravel()])

                # Create a grid that spans from min_values to max_values
                axis_grid = []
                for i_axis in range(len(max_values)):
                    axis_grid.append(np.linspace(min_values[i_axis], max_values[i_axis], num_points))
                # Create the grid of required dimension, here 40
                Axis = np.meshgrid(*axis_grid)
                # Stack the grid points into a 40D array of shape (40, num_points * num_points)
                grid_points = np.vstack([axis.ravel() for axis in Axis])
                print(f"grid_points preparation done")

                # KDE for each set
                kde1 = gaussian_kde(temp_embeddings.T)  # KDE for set 1
                kde2 = gaussian_kde(real_embeddings.T)  # KDE for set 2
                # Evaluate the KDEs on the shared grid
                p = kde1(grid_points)  # Evaluate KDE for set 1 on the grid
                q = kde2(grid_points)  # Evaluate KDE for set 2 on the grid
                p += EPSILON # avoid devision of zero
                q += EPSILON # avoid devision of zero
                # Normalize the KDE outputs to make them proper probability distributions
                p /= p.sum()  # Normalize p
                q /= q.sum()  # Normalize q
                print(f"{type(p)=}, {type(q)=}")

                # Compute the pointwise average distribution M
                m = 0.5 * (p + q)
                # Compute KL divergence for JSD
                kl_p_m = entropy(p, m)  # KL(P || M)
                kl_q_m = entropy(q, m)  # KL(Q || M)
                # Jensen-Shannon Divergence
                jsd = 0.5 * (kl_p_m + kl_q_m)
                print(f'Jensen-Shannon Divergence: {jsd}')

                total_kl[int(i_step-1)][i] = jsd
                # ############# total_kl calculation #############
                
                # ############# within_class_kl calculation #############
                unique_label_counter = 0
                for unique_label in embeddings_label[-1]:
                    print(f"{unique_label=}, {len(embeddings_label[i][unique_label])=}, {len(embeddings_label[-1][unique_label])=}")
                    if len(embeddings_label[-1][unique_label]) > 0:
                        if len(embeddings_label[i][unique_label]) == 0:
                            within_class_kl[int(i_step-1)][i] = float('inf')
                            break
                        else:
                            unique_label_counter += 1
                            min_values = np.minimum(embeddings_label[i][unique_label].min(axis=0), embeddings_label[-1][unique_label].min(axis=0))
                            max_values = np.maximum(embeddings_label[i][unique_label].max(axis=0), embeddings_label[-1][unique_label].max(axis=0))
                            print(f"Min values: {min_values}, Max values: {max_values}")
                            
                            # Define the number of grid points for each dimension
                            num_points = 32  # You can adjust this to control the resolution of the grid
                            # # Create a grid that spans from min_values to max_values
                            # x = np.linspace(min_values[0], max_values[0], num_points)
                            # y = np.linspace(min_values[1], max_values[1], num_points)
                            # # Create 2D grid
                            # X, Y = np.meshgrid(x, y)
                            # # Stack the grid points into a 2D array of shape (2, num_points * num_points)
                            # grid_points = np.vstack([X.ravel(), Y.ravel()])

                            # Create a grid that spans from min_values to max_values
                            axis_grid = []
                            for i_axis in range(len(max_values)):
                                axis_grid.append(np.linspace(min_values[i_axis], max_values[i_axis], num_points))
                            # Create the grid of required dimension, here 40
                            Axis = np.meshgrid(*axis_grid)
                            # Stack the grid points into a 40D array of shape (40, num_points * num_points)
                            grid_points = np.vstack([axis.ravel() for axis in Axis])

                            # KDE for each set
                            kde1 = gaussian_kde(embeddings_label[i][unique_label].T)  # KDE for set 1
                            kde2 = gaussian_kde(embeddings_label[-1][unique_label].T)  # KDE for set 2
                            # Evaluate the KDEs on the shared grid
                            p = kde1(grid_points)  # Evaluate KDE for set 1 on the grid
                            q = kde2(grid_points)  # Evaluate KDE for set 2 on the grid
                            p += EPSILON # avoid devision of zero
                            q += EPSILON # avoid devision of zero
                            # Normalize the KDE outputs to make them proper probability distributions
                            p /= p.sum()  # Normalize p
                            q /= q.sum()  # Normalize q
                            print(f"{type(p)=}, {type(q)=}")

                            # Compute the pointwise average distribution M
                            m = 0.5 * (p + q)
                            # Compute KL divergence for JSD
                            kl_p_m = entropy(p, m)  # KL(P || M)
                            kl_q_m = entropy(q, m)  # KL(Q || M)
                            # Jensen-Shannon Divergence
                            jsd = 0.5 * (kl_p_m + kl_q_m)
                            print(f'Jensen-Shannon Divergence for label#{unique_label} is: {jsd}')

                            within_class_kl[int(i_step-1)][i] += jsd
                    if within_class_kl[int(i_step-1)][i] != float('inf'):
                        within_class_kl[int(i_step-1)][i] = within_class_kl[int(i_step-1)][i] / unique_label_counter
                # ############# within_class_kl calculation #############
        else:
            print(f"[WARNING] Real samples not considered, no KL can be calculated")
        
    return total_kl, within_class_kl


def calculate_distance(args, embeddings_2d, embeddings, labels, embeddings_label, label_unique_values):
    if args.consider_real:

        total_l2, within_class_l2 = {}, {}
        total_cos, within_class_cos = {}, {}
        for i_step in range(1,args.steps+2):
            embeddings_label = [{} for _ in range((args.len_LLM+1 if args.consider_real else args.len_LLM))] # key=label, value=list of embeddings belonging to this label from different LLMs
            for ir, label in enumerate(label_unique_values):
                # print(f"{ir=}, {label=}")
                for _embeddings_label_for_each_model in embeddings_label:
                    _embeddings_label_for_each_model[label] = []
            for i in range((args.len_LLM+1 if args.consider_real else args.len_LLM)):
                # samples in the range: args.accumulate_sampels[i1]:args.accumulate_sampels[i1+1]
                # but should be the end of the list with i==args.len_LLM
                for ir, label in enumerate(label_unique_values):
                    if i < args.len_LLM:
                        # _index = (labels[args.accumulate_sampels[i]:args.accumulate_sampels[i]+int(args.num_use_samples_inner[i]*(i_step/(args.steps+1)))]==label)
                        # _embeddings_of_this_label = embeddings[args.accumulate_sampels[i]:args.accumulate_sampels[i]+int(args.num_use_samples_inner[i]*(i_step/(args.steps+1)))][_index]
                        _index = (labels[args.accumulate_sampels[-1][i]:args.accumulate_sampels[-1][i]+args.step_sample_count[i_step-1][i]]==label)
                        _embeddings_of_this_label = embeddings[args.accumulate_sampels[-1][i]:args.accumulate_sampels[-1][i]+args.step_sample_count[i_step-1][i]][_index]
                    else:
                        # _index = (labels[args.accumulate_sampels[i]:]==label)
                        # _embeddings_of_this_label = embeddings[args.accumulate_sampels[i]:][_index]
                        _index = (labels[args.accumulate_sampels[-1][i]:]==label)
                        _embeddings_of_this_label = embeddings[args.accumulate_sampels[-1][i]:][_index]
                        print(f'{i=}, {i_step=}, {label=}, {args.accumulate_sampels[i_step-1][i]=}, {_embeddings_of_this_label.shape=}')
                    # print(_index, type(_index), len(_index))
                    # print(f"{ir=}, {label=}", _embeddings_of_this_label, type(_embeddings_of_this_label), len(_embeddings_of_this_label))
                    embeddings_label[i][label] = copy.deepcopy(_embeddings_of_this_label)
                # print(embeddings_label[i])

            total_l2[int(i_step-1)], within_class_l2[int(i_step-1)] = [float('inf')]*args.len_LLM, [0.0]*args.len_LLM
            total_cos[int(i_step-1)], within_class_cos[int(i_step-1)] = [float('inf')]*args.len_LLM, [0.0]*args.len_LLM
            # real_embeddings = embeddings[args.accumulate_sampels[-2]:args.accumulate_sampels[-1], :]
            # real_labels = labels[args.accumulate_sampels[-2]:args.accumulate_sampels[-1]]
            real_embeddings = embeddings[args.accumulate_sampels[-1][-2]:args.accumulate_sampels[-1][-1], :]
            real_labels = labels[args.accumulate_sampels[-1][-2]:args.accumulate_sampels[-1][-1]]
            for i in range(args.len_LLM):
                # ############# total_l2 and total_cos calculation #############
                # temp_embeddings = embeddings[args.accumulate_sampels[i]:args.accumulate_sampels[i]+int(args.num_use_samples_inner[i]*(i_step/(args.steps+1))), :]
                # temp_labels = labels[args.accumulate_sampels[i]:args.accumulate_sampels[i]+int(args.num_use_samples_inner[i]*(i_step/(args.steps+1)))]
                temp_embeddings = embeddings[args.accumulate_sampels[-1][i]:args.accumulate_sampels[-1][i]+args.step_sample_count[i_step-1][i], :]
                temp_labels = labels[args.accumulate_sampels[-1][i]:args.accumulate_sampels[-1][i]+args.step_sample_count[i_step-1][i]]
                # print(f"{real_embeddings.shape=}, {temp_embeddings.shape=}")
                # print(type(temp_embeddings), type(real_embeddings))
                l2_distances = cdist(temp_embeddings, real_embeddings, metric='euclidean')
                cos_distances = cdist(temp_embeddings, real_embeddings, metric='cosine')
                total_l2[int(i_step-1)][i] = l2_distances.sum()/l2_distances.size
                total_cos[int(i_step-1)][i] = cos_distances.sum()/cos_distances.size
                # print(f"{l2_distances=}, {cos_distances=}, {total_l2[int(i_step-1)][i]=}, {total_cos[int(i_step-1)][i]=}")
                # ############# total_kl calculation #############
                
                # ############# within_class_kl calculation #############
                unique_label_counter = 0
                for unique_label in embeddings_label[-1]:
                    print(f"{unique_label=}, {len(embeddings_label[i][unique_label])=}, {len(embeddings_label[-1][unique_label])=}")
                    if len(embeddings_label[-1][unique_label]) > 0:
                        if len(embeddings_label[i][unique_label]) == 0:
                            within_class_l2[int(i_step-1)][i] = float('inf')
                            within_class_cos[int(i_step-1)][i] = float('inf')
                            break
                        else:
                            unique_label_counter += 1
                            l2_distances = cdist(embeddings_label[i][unique_label], embeddings_label[-1][unique_label], metric='euclidean')
                            cos_distances = cdist(embeddings_label[i][unique_label], embeddings_label[-1][unique_label], metric='cosine')
                            within_class_l2[int(i_step-1)][i] += l2_distances.sum()/l2_distances.size
                            within_class_cos[int(i_step-1)][i] += cos_distances.sum()/cos_distances.size
                            # print(f"{l2_distances=}, {cos_distances=}, {within_class_l2[int(i_step-1)][i]=}, {within_class_cos[int(i_step-1)][i]=}")
                    else:
                        within_class_l2[int(i_step-1)][i] = float('inf')
                        within_class_cos[int(i_step-1)][i] = float('inf')
                    if within_class_l2[int(i_step-1)][i] != float('inf'):
                        within_class_l2[int(i_step-1)][i] = within_class_l2[int(i_step-1)][i] / unique_label_counter
                    if within_class_cos[int(i_step-1)][i] != float('inf'):
                        within_class_cos[int(i_step-1)][i] = within_class_cos[int(i_step-1)][i] / unique_label_counter
                # ############# within_class_kl calculation #############
        else:
            print(f"[WARNING] Real samples not considered, no KL can be calculated")
        
    # assert 1 == 0
    return total_l2, within_class_l2, total_cos, within_class_cos


def calculate_fid_metrics(args, embeddings_2d, embeddings, labels, embeddings_label, label_unique_values):
    total_fid = {}
    within_class_fid = {}
    if args.consider_real:
        for i_step in range(1,args.steps+2):
            embeddings_label = [{} for _ in range((args.len_LLM+1 if args.consider_real else args.len_LLM))] # key=label, value=list of embeddings belonging to this label from different LLMs
            for ir, label in enumerate(label_unique_values):
                # print(f"{ir=}, {label=}")
                for _embeddings_label_for_each_model in embeddings_label:
                    _embeddings_label_for_each_model[label] = []
            for i in range((args.len_LLM+1 if args.consider_real else args.len_LLM)):
                # samples in the range: args.accumulate_sampels[i1]:args.accumulate_sampels[i1+1]
                # but should be the end of the list with i==args.len_LLM
                for ir, label in enumerate(label_unique_values):
                    if i < args.len_LLM:
                        # _index = (labels[args.accumulate_sampels[i]:args.accumulate_sampels[i]+int(args.num_use_samples_inner[i]*(i_step/(args.steps+1)))]==label)
                        # _embeddings_of_this_label = embeddings[args.accumulate_sampels[i]:args.accumulate_sampels[i]+int(args.num_use_samples_inner[i]*(i_step/(args.steps+1)))][_index]
                        _index = (labels[args.accumulate_sampels[-1][i]:args.accumulate_sampels[-1][i]+args.step_sample_count[i_step-1][i]]==label)
                        _embeddings_of_this_label = embeddings[args.accumulate_sampels[-1][i]:args.accumulate_sampels[-1][i]+args.step_sample_count[i_step-1][i]][_index]
                    else:
                        # _index = (labels[args.accumulate_sampels[i]:]==label)
                        # _embeddings_of_this_label = embeddings[args.accumulate_sampels[i]:][_index]
                        _index = (labels[args.accumulate_sampels[-1][i]:]==label)
                        _embeddings_of_this_label = embeddings[args.accumulate_sampels[-1][i]:][_index]
                    # print(_index, type(_index), len(_index))
                    # print(f"{ir=}, {label=}", _embeddings_of_this_label, type(_embeddings_of_this_label), len(_embeddings_of_this_label))
                    embeddings_label[i][label] = copy.deepcopy(_embeddings_of_this_label)
                # print(embeddings_label[i])

            # ############################### calculate FID per plm ###############################
            # total_fid[int(i_step-1)], within_class_fid[int(i_step-1)] = [float('inf')]*args.len_LLM, [0.0]*args.len_LLM
            # # real_embeddings = embeddings[args.accumulate_sampels[-2]:args.accumulate_sampels[-1], :]
            # # real_labels = labels[args.accumulate_sampels[-2]:args.accumulate_sampels[-1]]
            # real_embeddings = embeddings[args.accumulate_sampels[-1][-2]:args.accumulate_sampels[-1][-1], :]
            # real_labels = labels[args.accumulate_sampels[-1][-2]:args.accumulate_sampels[-1][-1]]
            # for i in range(args.len_LLM):
            #     # ############# total_fid calculation #############
            #     # temp_embeddings = embeddings[args.accumulate_sampels[i]:args.accumulate_sampels[i]+int(args.num_use_samples_inner[i]*(i_step/(args.steps+1))), :]
            #     # temp_labels = labels[args.accumulate_sampels[i]:args.accumulate_sampels[i]+int(args.num_use_samples_inner[i]*(i_step/(args.steps+1)))]
            #     temp_embeddings = embeddings[args.accumulate_sampels[-1][i]:args.accumulate_sampels[-1][i]+args.step_sample_count[i_step-1][i], :]
            #     temp_labels = labels[args.accumulate_sampels[-1][i]:args.accumulate_sampels[-1][i]+args.step_sample_count[i_step-1][i]]
            #     # print(f"{real_embeddings.shape=}, {temp_embeddings.shape=}")
            #     # print(type(temp_embeddings), type(real_embeddings))
            #     fid_value = calculate_fid(temp_embeddings, real_embeddings)
            #     total_fid[int(i_step-1)][i] = fid_value.sum()/fid_value.size
            #     # print(f"{fid_value=}, {total_fid[int(i_step-1)][i]=}")
            #     # ############# total_fid calculation #############
                
            #     # ############# within_class_fid calculation #############
            #     unique_label_counter = 0
            #     for unique_label in embeddings_label[-1]:
            #         print(f"{unique_label=}, {len(embeddings_label[i][unique_label])=}, {len(embeddings_label[-1][unique_label])=}")
            #         if len(embeddings_label[-1][unique_label]) > 0:
            #             if len(embeddings_label[i][unique_label]) == 0:
            #                 within_class_fid[int(i_step-1)][i] = float('inf')
            #                 break
            #             else:
            #                 unique_label_counter += 1
            #                 fid_value = calculate_fid(embeddings_label[i][unique_label], embeddings_label[-1][unique_label])
            #                 within_class_fid[int(i_step-1)][i] += fid_value.sum()/fid_value.size
            #                 print(f"{fid_value=}, {within_class_fid[int(i_step-1)][i]=}")
            #         else:
            #             within_class_fid[int(i_step-1)][i] = float('inf')
            #         if within_class_fid[int(i_step-1)][i] != float('inf'):
            #             within_class_fid[int(i_step-1)][i] = within_class_fid[int(i_step-1)][i] / unique_label_counter
            #     # ############# within_class_kl calculation #############
            # ############################### calculate FID per plm ###############################
            # ############################### calculate FID with total data from PLMs ###############################
            total_fid[int(i_step-1)], within_class_fid[int(i_step-1)] = float('inf'), 0.0
            # real_embeddings = embeddings[args.accumulate_sampels[-2]:args.accumulate_sampels[-1], :]
            # real_labels = labels[args.accumulate_sampels[-2]:args.accumulate_sampels[-1]]
            real_embeddings = embeddings[args.accumulate_sampels[-1][-2]:args.accumulate_sampels[-1][-1], :]
            real_labels = labels[args.accumulate_sampels[-1][-2]:args.accumulate_sampels[-1][-1]]
            temp_embeddings = []
            temp_labels = []
            for i in range(args.len_LLM):
                # ############# total_fid calculation #############
                # temp_embeddings = embeddings[args.accumulate_sampels[i]:args.accumulate_sampels[i]+int(args.num_use_samples_inner[i]*(i_step/(args.steps+1))), :]
                # temp_labels = labels[args.accumulate_sampels[i]:args.accumulate_sampels[i]+int(args.num_use_samples_inner[i]*(i_step/(args.steps+1)))]
                temp_embeddings.append(embeddings[args.accumulate_sampels[-1][i]:args.accumulate_sampels[-1][i]+args.step_sample_count[i_step-1][i], :])
                temp_labels.append(labels[args.accumulate_sampels[-1][i]:args.accumulate_sampels[-1][i]+args.step_sample_count[i_step-1][i]])
                # print(f"{real_embeddings.shape=}, {temp_embeddings.shape=}")
                # print(type(temp_embeddings), type(real_embeddings))
            temp_embeddings = np.concatenate(temp_embeddings, axis=0)
            fid_value = calculate_fid(temp_embeddings, real_embeddings)
            total_fid[int(i_step-1)] = fid_value.sum()/fid_value.size
            # print(f"{fid_value=}, {total_fid[int(i_step-1)][i]=}")
            # ############# total_fid calculation #############
            
            # ############# within_class_fid calculation #############
            unique_label_counter = 0
            for unique_label in embeddings_label[-1]:
                print(f"{unique_label=}, {len(embeddings_label[i][unique_label])=}, {len(embeddings_label[-1][unique_label])=}")
                if len(embeddings_label[-1][unique_label]) > 0:
                    if len(embeddings_label[i][unique_label]) == 0:
                        within_class_fid[int(i_step-1)] = float('inf')
                        break
                    else:
                        unique_label_counter += 1
                        fid_value = calculate_fid(embeddings_label[i][unique_label], embeddings_label[-1][unique_label])
                        within_class_fid[int(i_step-1)] += fid_value.sum()/fid_value.size
                        print(f"{fid_value=}, {within_class_fid[int(i_step-1)]=}")
                else:
                    within_class_fid[int(i_step-1)] = float('inf')
                if within_class_fid[int(i_step-1)] != float('inf'):
                    within_class_fid[int(i_step-1)] = within_class_fid[int(i_step-1)] / unique_label_counter
            # ############# within_class_kl calculation #############
            # ############################### calculate FID with all data from PLMs ###############################
    else:
        print(f"[WARNING] Real samples not considered, no FID can be calculated")

    return total_fid, within_class_fid


def calculate_fid_metrics_sample_delta(args, embeddings_2d, embeddings, labels, embeddings_label, label_unique_values):
    total_fid = {}
    within_class_fid = {}
    if args.consider_real:
        for i_step in range(1,args.steps+2):
            embeddings_label = [{} for _ in range((args.len_LLM+1 if args.consider_real else args.len_LLM))] # key=label, value=list of embeddings belonging to this label from different LLMs
            for ir, label in enumerate(label_unique_values):
                # print(f"{ir=}, {label=}")
                for _embeddings_label_for_each_model in embeddings_label:
                    _embeddings_label_for_each_model[label] = []
            for i in range((args.len_LLM+1 if args.consider_real else args.len_LLM)):
                # samples in the range: args.accumulate_sampels[i1]:args.accumulate_sampels[i1+1]
                # but should be the end of the list with i==args.len_LLM
                for ir, label in enumerate(label_unique_values):
                    if i < args.len_LLM:
                        # _index = (labels[args.accumulate_sampels[i]+int(args.num_use_samples_inner[i]*((i_step-1)/(args.steps+1))):args.accumulate_sampels[i]+int(args.num_use_samples_inner[i]*(i_step/(args.steps+1)))]==label)
                        # _embeddings_of_this_label = embeddings[args.accumulate_sampels[i]+int(args.num_use_samples_inner[i]*((i_step-1)/(args.steps+1))):args.accumulate_sampels[i]+int(args.num_use_samples_inner[i]*(i_step/(args.steps+1)))][_index]
                        # _index = (labels[args.accumulate_sampels[i]:args.accumulate_sampels[i]+int(args.num_use_samples_inner[i]*(i_step/(args.steps+1)))]==label)
                        # _embeddings_of_this_label = embeddings[args.accumulate_sampels[i]:args.accumulate_sampels[i]+int(args.num_use_samples_inner[i]*(i_step/(args.steps+1)))][_index]
                        _previous_step_obtain = args.step_sample_count[i_step-2][i] if i_step > 1 else 0
                        # print(f"LLM#{i}, step={i_step}, start={args.accumulate_sampels[-1][i]+_previous_step_obtain}, end={args.accumulate_sampels[-1][i]+args.step_sample_count[i_step-1][i]}, {args.accumulate_sampels[-1][i]=}, {_previous_step_obtain=}, {args.step_sample_count[i_step-1][i]=}")
                        _index = (labels[args.accumulate_sampels[-1][i]+_previous_step_obtain:args.accumulate_sampels[-1][i]+args.step_sample_count[i_step-1][i]]==label)
                        _embeddings_of_this_label = embeddings[args.accumulate_sampels[-1][i]+_previous_step_obtain:args.accumulate_sampels[-1][i]+args.step_sample_count[i_step-1][i]][_index]
                    else:
                        # _index = (labels[args.accumulate_sampels[i]:]==label)
                        # _embeddings_of_this_label = embeddings[args.accumulate_sampels[i]:][_index]
                        _index = (labels[args.accumulate_sampels[-1][i]:]==label)
                        _embeddings_of_this_label = embeddings[args.accumulate_sampels[-1][i]:][_index]
                    # print(_index, type(_index), len(_index))
                    # print(f"{ir=}, {label=}", _embeddings_of_this_label, type(_embeddings_of_this_label), len(_embeddings_of_this_label))
                    embeddings_label[i][label] = copy.deepcopy(_embeddings_of_this_label)
                # print(embeddings_label[i])

            total_fid[int(i_step-1)], within_class_fid[int(i_step-1)] = [float('inf')]*args.len_LLM, [0.0]*args.len_LLM
            # real_embeddings = embeddings[args.accumulate_sampels[-2]:args.accumulate_sampels[-1], :]
            # real_labels = labels[args.accumulate_sampels[-2]:args.accumulate_sampels[-1]]
            real_embeddings = embeddings[args.accumulate_sampels[-1][-2]:args.accumulate_sampels[-1][-1], :]
            real_labels = labels[args.accumulate_sampels[-1][-2]:args.accumulate_sampels[-1][-1]]
            for i in range(args.len_LLM):
                # ############# total_fid calculation #############
                # temp_embeddings = embeddings[args.accumulate_sampels[i]+int(args.num_use_samples_inner[i]*((i_step-1)/(args.steps+1))):args.accumulate_sampels[i]+int(args.num_use_samples_inner[i]*(i_step/(args.steps+1))), :]
                # temp_labels = labels[args.accumulate_sampels[i]+int(args.num_use_samples_inner[i]*((i_step-1)/(args.steps+1))):args.accumulate_sampels[i]+int(args.num_use_samples_inner[i]*(i_step/(args.steps+1)))]
                _previous_step_obtain = args.step_sample_count[i_step-2][i] if i_step > 1 else 0
                temp_embeddings = embeddings[args.accumulate_sampels[i_step-1][i]+_previous_step_obtain:args.accumulate_sampels[-1][i]+args.step_sample_count[i_step-1][i], :]
                temp_labels = labels[args.accumulate_sampels[i_step-1][i]+_previous_step_obtain:args.accumulate_sampels[-1][i]+args.step_sample_count[i_step-1][i]]
                # print(f"{real_embeddings.shape=}, {temp_embeddings.shape=}")
                # print(type(temp_embeddings), type(real_embeddings))
                fid_value = calculate_fid(temp_embeddings, real_embeddings)
                total_fid[int(i_step-1)][i] = fid_value.sum()/fid_value.size
                # print(f"{fid_value=}, {total_fid[int(i_step-1)][i]=}")
                # ############# total_fid calculation #############
                
                # ############# within_class_fid calculation #############
                unique_label_counter = 0
                for unique_label in embeddings_label[-1]:
                    print(f"{unique_label=}, {len(embeddings_label[i][unique_label])=}, {len(embeddings_label[-1][unique_label])=}")
                    if len(embeddings_label[-1][unique_label]) > 0:
                        if len(embeddings_label[i][unique_label]) == 0:
                            within_class_fid[int(i_step-1)][i] = float('inf')
                            break
                        else:
                            unique_label_counter += 1
                            fid_value = calculate_fid(embeddings_label[i][unique_label], embeddings_label[-1][unique_label])
                            within_class_fid[int(i_step-1)][i] += fid_value.sum()/fid_value.size
                            # print(f"{fid_value=}, {within_class_fid[int(i_step-1)][i]=}")
                    if within_class_fid[int(i_step-1)][i] != float('inf'):
                        within_class_fid[int(i_step-1)][i] = within_class_fid[int(i_step-1)][i] / unique_label_counter
                # ############# within_class_kl calculation #############
    else:
        print(f"[WARNING] Real samples not considered, no FID can be calculated")

    return total_fid, within_class_fid


def calculate_and_save_tsne(args):
    ############################### caldulate and save ################################
    if 'bert' in args.small_model_name:
        init_model = BertModel.from_pretrained(MODEL_PATH[args.small_model_name])
        args.tokenizer = BertTokenizer.from_pretrained(MODEL_PATH[args.small_model_name])
    elif 'sentence' in args.small_model_name:
        embedding_model = SentenceTransformer(MODEL_PATH[args.small_model_name])

    train_data_list, train_iter_list = load_data(args, batch_size=8, backward_batch_size=1000, device=args.device, gold_data_path=f'./data/{args.task_name}/std/', syn_data_path=SYN_DATA_PATH, vectors=None, use_tree=False, num_use_samples_inner=args.num_use_samples_inner, num_use_samples_outer=100, shuffle_train=True)
    # args.train_data_list, args.train_iter_list = train_data_list, train_iter_list
    print(f"{len(train_data_list)=}, {len(train_iter_list)=}")
    embedding_list = []
    label_list = []
    for i in range(len(train_data_list)):
        _embeddings, _labels = get_embedding(args, init_model, train_iter_list[i])
        embedding_list.append(_embeddings)
        label_list.append(_labels)
    # embeddings = torch.cat(embedding_list,dim=0).cpu().numpy()
    embeddings = np.concatenate(embedding_list, axis=0)
    labels = np.concatenate(label_list,axis=0)
    print(f"{embeddings.shape=}, {labels.shape=}")
    embeddings = embeddings.reshape(embeddings.shape[0],-1)
    
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    if not os.path.exists(f'./figure/distribution/embeddings/{args.folder_name}/'):
        os.makedirs(f'./figure/distribution/embeddings/{args.folder_name}/')
    # np.savez(f'./figure/distribution/embeddings/{args.folder_name}/{args.model_name_sample}.npz', embeddings_2d=embeddings_2d, embeddings=embeddings, labels=labels)
    np.savez(f'./figure/distribution/embeddings/{args.folder_name}/{save_type}_{f"with{args.gold_data_num}" if args.consider_real else "without"}test.npz', embeddings_2d=embeddings_2d, embeddings=embeddings, labels=labels) # currently without test
    
    # # data = np.load(f'./figure/distribution/embeddings/{args.model_name_sample}.npz')
    # data = np.load(f'./figure/distribution/embeddings/{save_type}_{args.model_name_sample}.npz')
    # embeddings_2d = data['embeddings_2d']
    # embeddings = data['embeddings']
    # labels = data['labels']
    # tsne = TSNE(n_components=2, random_state=42)
    # embeddings_2d = tsne.fit_transform(embeddings[:int(args.accumulate_sampels[-1])])
    # # np.savez(f'./figure/distribution/embeddings/withoutest_{args.model_name_sample}.npz', embeddings_2d=embeddings_2d, embeddings=embeddings[:int(args.accumulate_sampels[-1])], labels=labels[:int(args.accumulate_sampels[-1])])
    # np.savez(f'./figure/distribution/embeddings/{save_type}_withoutest_{args.model_name_sample}.npz', embeddings_2d=embeddings_2d, embeddings=embeddings[:int(args.accumulate_sampels[-1])], labels=labels[:int(args.accumulate_sampels[-1])])
    ############################### caldulate and save ################################


def read_training_dynamics_together(metric_file):

    '''
        ./
        results/
        multiGold_eval_on_real/
        with_real_few_shot_accumulate_votingCLASS_promptCLASS_8_0.0_randomContrast_top/
        gold_100_1_0.0_OODgold/bert-base-uncased/
        sentence-t5-base/
        0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD1_FuseDataset1/
        0_1_init200_steps4_unbalance_temp1.0/
        fewshotK8_15_0.5/
        imdb/
        gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__chatglm3-6b-base_1000__flan-t5-xl_1000/
        12345/
        correctness_prediction_logits_confidence_variability_for_dynamic_{}.pth
    '''

    train_dynamics = {}

    task_name = metric_file.split('/')[-4]
    models_samples = metric_file.split('/')[-3]
    # assert '__' in models_samples, f"{models_samples=}"
    models_samples = models_samples.split('__')
    models_samples = [item.split('_') for item in models_samples]
    models = [item[0] for item in models_samples]
    samples = [int(item[1]) for item in models_samples]
    accumulated_sample_count = [0]
    for sample_num in samples:
        accumulated_sample_count.append(accumulated_sample_count[-1]+sample_num)
    NUM_MODELS = len(models)

    for im, model in enumerate(models):
        train_dynamics[model] = {}
    
        _correctness, _prediction, _logits, _confidence, _variability = torch.load(metric_file.format(model))
        print(f"{_correctness.shape=}")
        _correctness = torch.sum(_correctness, dim=0)
        print(f"{_correctness.shape=}")
        print(f"{_correctness=}")

        correctness =  _correctness.cpu().numpy()
        confidence =  _confidence.cpu().numpy()
        variability =  _variability.cpu().numpy()

        print(f"{correctness.shape=}, {confidence.shape=}, {variability.shape=}")

        dataframe_dict = {"variability":variability, "confidence":confidence, "correctness": correctness, "guid":np.arange(0,len(confidence),1)}
        # train_dynamics[model] = pd.DataFrame(dataframe_dict)
        train_dynamics[model] = copy.deepcopy(dataframe_dict)

    return train_dynamics, models, samples, accumulated_sample_count, NUM_MODELS, task_name



def plot_dynamics(args, labels, embeddings_label, label_unique_values):
    PLM_names = {'gpt2-xl':'GPT2', 'llama-2-7b-chat-hf':'Llama2', 'vicuna-7b-1.5v':'Vicuna', 'opt-6.7b':'OPT', 'chatglm3-6b-base':'ChatGLM3', 'flan-t5-xl':'Flan-T5', 'gpt-3.5-turbo-instruct':'GPT3.5', 'gpt-4-turbo-preview':'GPT4'}
    if args.results_path == None:
        print(f"[WARNING], args.results_path==None, expected to be specified")
        return
    
    _file_path = args.results_path + "correctness_prediction_logits_confidence_variability_for_dynamic_{}.pth"

    total_train_dy_metrics, plms, samples, accumulated_sample_count, NUM_MODELS, task_name = read_training_dynamics_together(metric_file=_file_path)
    for i_step in range(1, args.steps+2):
        train_dy_metrics = copy.deepcopy(total_train_dy_metrics)
        for im in range(len(plms)):
            train_sample_count = int(args.step_sample_count[i_step-1][im]*0.9)
            train_dy_metrics[args.llms[im]] = {"variability":train_dy_metrics[args.llms[im]]["variability"][:train_sample_count], 
                                               "confidence":train_dy_metrics[args.llms[im]]["confidence"][:train_sample_count], 
                                               "correctness": train_dy_metrics[args.llms[im]]["correctness"][:train_sample_count], 
                                               "guid":np.arange(0,train_sample_count,1)}
            train_dy_metrics[args.llms[im]] = pd.DataFrame(train_dy_metrics[args.llms[im]])
        for plm, sample in zip(plms, args.step_sample_count[i_step-1]):                
            if 'flip' in _file_path:
                dy_metrics_save_dir = f'./figure/dynamics/record/{args.folder_name}/flip/'
                plot_dy_save_dir = f'./figure/dynamics/{args.folder_name}/flip/'
            else:
                folder_name = 'original' if 'data_new' in _file_path else ('single_progen' if 'single' in _file_path else 'fuse')
                num_epochs = 0 # 3 if '3epochs' in _file_path else (6 if '6epochs' in _file_path else (10 if '10epochs' in _file_path else 0))
                dy_metrics_save_dir = f'./figure/dynamics/record/{args.folder_name}/{folder_name}/'
                plot_dy_save_dir = f'./figure/dynamics/{args.folder_name}/{folder_name}/'
            if not os.path.exists(dy_metrics_save_dir):
                os.makedirs(dy_metrics_save_dir)
            if not os.path.exists(plot_dy_save_dir):
                os.makedirs(plot_dy_save_dir)
            
            train_dy_metrics[plm].to_json(f'{dy_metrics_save_dir}/{plm}.jsonl', orient='records', lines=True)
            _mean = np.mean(train_dy_metrics[plm]['variability'])
            _max = np.max(train_dy_metrics[plm]['variability'])
            _min = np.min(train_dy_metrics[plm]['variability'])
            _std = np.std(train_dy_metrics[plm]['variability'])
            print(f"model {plm} with {_mean=}, {_max=}, {_min=}, {_std=}")
            # plot_data_map(train_dy_metrics[plm], f'{plot_dy_save_dir}/{plm}_graphOnly.pdf', title=f'{PLM_names[plm]}', show_hist=False, model='bert-base-uncased')
            plot_data_map(train_dy_metrics[plm], f'{plot_dy_save_dir}/step{i_step}_{plm}_graphOnly.png', title=f'{PLM_names[plm]}', show_hist=False, model='bert-base-uncased')
            # plot_data_map(train_dy_metrics[plm], f'{plot_dy_save_dir}/{plm}.pdf', title=f'{PLM_names[plm]}', show_hist=True, model='bert-base-uncased')
            plot_data_map(train_dy_metrics[plm], f'{plot_dy_save_dir}/step{i_step}_{plm}.png', title=f'{PLM_names[plm]}', show_hist=True, model='bert-base-uncased')


def plot_embedding_length_distributions(args, embeddings):
    PLM_names = {'gpt2-xl':'GPT2', 'llama-2-7b-chat-hf':'Llama2', 'vicuna-7b-1.5v':'Vicuna', 'opt-6.7b':'OPT', 'chatglm3-6b-base':'ChatGLM3', 'flan-t5-xl':'Flan-T5', 'gpt-3.5-turbo-instruct':'GPT3.5', 'gpt-4-turbo-preview':'GPT4'}

    if 'bert' in args.small_model_name:
        init_model = BertModel.from_pretrained(MODEL_PATH[args.small_model_name])
        args.tokenizer = BertTokenizer.from_pretrained(MODEL_PATH[args.small_model_name])
    elif 'sentence' in args.small_model_name:
        embedding_model = SentenceTransformer(MODEL_PATH[args.small_model_name])
    train_data_list, train_iter_list = load_data(args, batch_size=8, backward_batch_size=1000, device=args.device, gold_data_path=f'./data/{args.task_name}/std/', syn_data_path=SYN_DATA_PATH, vectors=None, use_tree=False, num_use_samples_inner=args.num_use_samples_inner, num_use_samples_outer=100, shuffle_train=True)

    fig, axs = plt.subplots(nrows=1, ncols=args.steps+1, figsize=(17, 3), sharex=True, sharey=True)
    # args.train_data_list, args.train_iter_list
    if args.consider_real:
        real_embeddings = train_data_list[-1].text
        # print(f"{real_embeddings.shape}")
        # real_length = np.asarray([len(_emb) for _emb in real_embeddings])
        real_length = np.asarray([len(_emb.split()) for _emb in real_embeddings])
        print(f"{real_length=}")
        for i_step in range(1,args.steps+2):
            syn_embeddings = []
            for i in range(args.len_LLM):
                start_idx = 0 if i_step==1 else args.step_sample_count[i_step-2][i]
                start_idx = 0
                syn_embeddings = syn_embeddings + train_data_list[i].text[start_idx:args.step_sample_count[i_step-1][i]]
                # syn_embeddings.append(embeddings[args.accumulate_sampels[-1][i]:args.accumulate_sampels[-1][i]+args.step_sample_count[i_step-1][i]])
            # syn_length = np.asarray([len(_emb) for _emb in syn_embeddings])
            syn_length = np.asarray([len(_emb.split()) for _emb in syn_embeddings])
            print(f"{syn_length=}")
            axs[i_step-1].hist(real_length, density=True, bins=30, color='skyblue', edgecolor='skyblue', alpha=0.7, label='real(private)')
            axs[i_step-1].hist(syn_length, density=True, bins=30, color='yellowgreen', edgecolor='yellowgreen', alpha=0.3, label='synthetic')
            axs[i_step-1].set_title(f'PE iteration {i_step-1}', fontsize=21)
            if args.task_name == 'yelpRating':
                axs[i_step-1].set_ylim([0.0,0.065])
            if i_step == 1:
                axs[0].set_ylabel('Density', fontsize=20)
            if i_step == args.steps+1:
                axs[i_step-1].legend(fontsize=13)

    fig.tight_layout()
    plt.tight_layout()
    if not os.path.exists(f'./figure/introduction/text_converge/{args.folder_name}/'):
        os.makedirs(f'./figure/introduction/text_converge/{args.folder_name}/')
    print(f'./figure/introduction/text_converge/{args.folder_name}/{args.llms}.png')
    plt.savefig(f'./figure/introduction/text_converge/{args.folder_name}/{args.llms}.png',dpi=200)

    # only start and end
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(6.8, 2), sharex=True, sharey=True)
    # args.train_data_list, args.train_iter_list
    if args.consider_real:
        real_embeddings = train_data_list[-1].text
        # print(f"{real_embeddings.shape}")
        # real_length = np.asarray([len(_emb) for _emb in real_embeddings])
        real_length = np.asarray([len(_emb.split()) for _emb in real_embeddings])
        print(f"{real_length=}")
        for i_step in [1,args.steps+1]:
            axs_idx = 0 if i_step==1 else 1
            syn_embeddings = []
            for i in range(args.len_LLM):
                start_idx = 0 if i_step==1 else args.step_sample_count[i_step-2][i]
                start_idx = 0
                syn_embeddings = syn_embeddings + train_data_list[i].text[start_idx:args.step_sample_count[i_step-1][i]]
                # syn_embeddings.append(embeddings[args.accumulate_sampels[-1][i]:args.accumulate_sampels[-1][i]+args.step_sample_count[i_step-1][i]])
            # syn_length = np.asarray([len(_emb) for _emb in syn_embeddings])
            syn_length = np.asarray([len(_emb.split()) for _emb in syn_embeddings])
            print(f"{syn_length=}")
            axs[axs_idx].hist(real_length, density=True, bins=30, color='skyblue', edgecolor='skyblue', alpha=0.7, label='real(private)')
            axs[axs_idx].hist(syn_length, density=True, bins=30, color='yellowgreen', edgecolor='yellowgreen', alpha=0.5, label='synthetic')
            axs[axs_idx].set_title(f'PE iteration {i_step-1}', fontsize=21)
            if args.task_name == 'yelpRating':
                axs[axs_idx].set_ylim([0.0,0.05])
            if i_step == 1:
                axs[0].set_ylabel('Density', fontsize=20)
            if i_step == args.steps+1:
                axs[axs_idx].legend(fontsize=13)

    fig.tight_layout()
    plt.tight_layout()
    if not os.path.exists(f'./figure/introduction/text_converge/start_end/{args.folder_name}/'):
        os.makedirs(f'./figure/introduction/text_converge/start_end/{args.folder_name}/')
    print(f'./figure/introduction/text_converge/start_end/{args.folder_name}/{args.llms}.png')
    plt.savefig(f'./figure/introduction/text_converge/start_end/{args.folder_name}/{args.llms}.png',dpi=200)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='summary generator')
    parser.add_argument("--small_model_name", type=str, default='bert-base-uncased', help="The small Transformer language model to use.")
    parser.add_argument('--llms', default=['gpt2-xl','llama-2-7b-chat-hf'], nargs='+', type=str)
    parser.add_argument('--num_use_samples_inner', default=[200000,200000], nargs='+', type=int)
    parser.add_argument('--syn_data_path', default=None, type=str)
    parser.add_argument('--logging_path', default=None, type=str)
    parser.add_argument('--results_path', default=None, type=str)
    parser.add_argument('--task_name', default="rte", type=str)
    parser.add_argument('--gpu', default=0, type=int, help='gpu device id')
    parser.add_argument('--seed', default=12345, type=int, help='random seed')
    parser.add_argument('--consider_real', default=False, type=bool, help='whether considers real test data in distribution visalization')
    parser.add_argument('--gold_data_num', default=1000, type=int, help='how much real data to consider')
    parser.add_argument('--steps', default=4, type=int, help='how many accumulation iterations are done')
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(args.gpu)  
    args.device = device
    torch.cuda.empty_cache()

    args.len_LLM = len(args.llms)
    assert args.len_LLM == len(args.num_use_samples_inner), "Must specify the number of inner samples used for every LLM's generated data"

    ##########################################
    # find each step accumulated samples
    args.start_sample_count = [int(args.num_use_samples_inner[im]/(args.steps+1)) for im in range(args.len_LLM)]
    SAMPLE_COUNT = [[]] * (args.steps+1)
    with open(args.logging_path, 'r') as file:
        step_counter = -1
        for line in file:
            line = line.strip()
            test_acc_result = {"max":{}, "max_epoch":{}, "first":{}, "trajectory":{}, "mean":{}}
            if 'use [' in line and '] train data...' in line:
                print(f'[debug] {line=}')
                start_idx = line.rfind('[')
                end_idx = line.rfind(']')
                str_of_list = line[start_idx+1:end_idx]
                print(f'[debug] {str_of_list=}')
                list_of_str = str_of_list.replace(' ', '').split(',')
                step_sample = [int(_i) for _i in list_of_str]
                _starting = True
                for im in range(args.len_LLM):
                    if step_sample[im] != args.start_sample_count[im]:
                        step_counter += 1
                        _starting = False
                        break
                if _starting:
                    step_counter = -1
                SAMPLE_COUNT[step_counter+1] = step_sample
                print(f'[debug] {SAMPLE_COUNT[step_counter+1]=}')
    args.start_sample_count = torch.tensor(args.start_sample_count, dtype=torch.long).to(args.device)
    args.step_sample_count = torch.tensor(SAMPLE_COUNT, dtype=torch.long).to(args.device)
    print(f'[INFO] {args.step_sample_count=}')
    args.accumulate_sampels = []
    for i_step in range(args.steps+1):
        args.accumulate_sampels.append([0])
        for _i in range(args.len_LLM):
            args.accumulate_sampels[i_step].append(args.accumulate_sampels[i_step][-1]+args.step_sample_count[i_step][_i])
        if args.consider_real:
            args.accumulate_sampels[i_step].append(args.accumulate_sampels[i_step][-1]+args.gold_data_num)
    args.accumulate_sampels = torch.tensor(args.accumulate_sampels, dtype=torch.long).to(args.device)
    print(f"[INFO] {args.accumulate_sampels=}")
    ##########################################


    args.model_name_sample = f'{args.task_name}___{args.llms[0]}_{args.num_use_samples_inner[0]}'
    for _model, num_samples_inner in zip(args.llms[1:], args.num_use_samples_inner[1:]):
        args.model_name_sample += f'__{_model}_{num_samples_inner}'
    args.folder_name = f"{args.syn_data_path.split('/')[2]}/{args.syn_data_path.split('/')[3]}/{args.model_name_sample}"

    print(f"{args.syn_data_path=}")
    if args.syn_data_path != None:
        SYN_DATA_PATH = args.syn_data_path
        print(f"{SYN_DATA_PATH}")

    save_type = 'origianl' if 'data_new' in SYN_DATA_PATH else ('singleProgen' if 'single' in SYN_DATA_PATH else 'accumulate')

    # ############## calculate and save tsne ##############
    # calculate_and_save_tsne(args)
    # ############## calculate and save tsne ##############

    # assert 1 == 0

    data = np.load(f'./figure/distribution/embeddings/{args.folder_name}/{save_type}_{f"with{args.gold_data_num}" if args.consider_real else "without"}test.npz')
    # data = np.load(f'./figure/distribution/embeddings/withoutest_{args.model_name_sample}.npz')
    # data = np.load(f'./figure/distribution/temp.npz')
    embeddings_2d = data['embeddings_2d']
    embeddings = data['embeddings']
    labels = data['labels']
    label_unique_values, counts = np.unique(labels, return_counts=True)
    # print(f"{label_unique_values=}")
    print(embeddings.shape, type(embeddings))
    print(labels.shape, type(labels))

    embeddings_label = [{} for _ in range((args.len_LLM+1 if args.consider_real else args.len_LLM))] # key=label, value=list of embeddings belonging to this label from different LLMs
    for ir, label in enumerate(label_unique_values):
        # print(f"{ir=}, {label=}")
        for _embeddings_label_for_each_model in embeddings_label:
            _embeddings_label_for_each_model[label] = []
    for i in range((args.len_LLM+1 if args.consider_real else args.len_LLM)):
        # samples in the range: args.accumulate_sampels[i1]:args.accumulate_sampels[i1+1]
        # but should be the end of the list with i==args.len_LLM
        for ir, label in enumerate(label_unique_values):
            if i < args.len_LLM:
                # _index = (labels[args.accumulate_sampels[i]:args.accumulate_sampels[i+1]]==label)
                # _embeddings_of_this_label = embeddings[args.accumulate_sampels[i]:args.accumulate_sampels[i+1]][_index]
                _index = (labels[args.accumulate_sampels[-1][i]:args.accumulate_sampels[-1][i+1]]==label)
                _embeddings_of_this_label = embeddings[args.accumulate_sampels[-1][i]:args.accumulate_sampels[-1][i+1]][_index]
            else:
                # _index = (labels[args.accumulate_sampels[i]:]==label)
                # _embeddings_of_this_label = embeddings[args.accumulate_sampels[i]:][_index]
                _index = (labels[args.accumulate_sampels[-1][i]:]==label)
                _embeddings_of_this_label = embeddings[args.accumulate_sampels[-1][i]:][_index]
            # print(_index, type(_index), len(_index))
            # print(f"{ir=}, {label=}", _embeddings_of_this_label, type(_embeddings_of_this_label), len(_embeddings_of_this_label))
            embeddings_label[i][label].append(_embeddings_of_this_label)

    # for i1 in range(args.len_LLM):
    #     for i2 in range(i1, args.len_LLM):
    #         mutual_info = mutual_info_regression(embeddings[args.accumulate_sampels[i1]:args.accumulate_sampels[i1]+1000], embeddings[args.accumulate_sampels[i2]:args.accumulate_sampels[i2]+1000], discrete_features=[False])
    #         # mutual_info = mutual_info_regression(embeddings[args.accumulate_sampels[i1]:args.accumulate_sampels[i1+1]], embeddings[args.accumulate_sampels[i2]:args.accumulate_sampels[i2+1]], discrete_features=[False])
    #         print(f"syn #{args.llms[i1]}, syn #{args.llms[i2]}, MI={mutual_info}")
    # for i1 in range(args.len_LLM):
    #     mutual_info = mutual_info_regression(embeddings[args.accumulate_sampels[i1]:args.accumulate_sampels[i1]+1000], embeddings[args.accumulate_sampels[-1]:args.accumulate_sampels[-1]+1000], discrete_features=[False])
    #     # mutual_info = mutual_info_regression(embeddings[args.accumulate_sampels[i1]:args.accumulate_sampels[i1+1]], embeddings[args.accumulate_sampels[-1]:], discrete_features=[False])
    #     print(f"syn #{args.llms[i1]}, golden dataset, MI={mutual_info}")

    # assert 1 == 0
    

    # total_kl, within_class_kl = calculate_KL(args, embeddings_2d, embeddings, labels, embeddings_label, label_unique_values)
    # print(f"KL results: {total_kl=}, {within_class_kl=}")
    
    # total_l2, within_class_l2, total_cos, within_class_cos = calculate_distance(args, embeddings_2d, embeddings, labels, embeddings_label, label_unique_values)
    # print(f"L2 & cosine-similarity results: {total_l2=}, {within_class_l2=}, {total_cos=}, {within_class_cos=}")

    total_fid, within_class_fid = calculate_fid_metrics(args, embeddings_2d, embeddings, labels, embeddings_label, label_unique_values)
    print(f"FID results: {total_fid=}, {within_class_fid=}")
    total_fid, within_class_fid = calculate_fid_metrics_sample_delta(args, embeddings_2d, embeddings, labels, embeddings_label, label_unique_values)
    print(f"FID for sample delta results: {total_fid=}, {within_class_fid=}")

    # total_fid, within_class_fid = calculate_fid_metrics_sample_delta(args, embeddings_2d, embeddings_2d, labels, embeddings_label, label_unique_values)
    # print(f"FID for 2-major components, sample delta results: {total_fid=}, {within_class_fid=}")
    
    # plot_labeled_distribution(args, embeddings_2d, embeddings, labels, embeddings_label, label_unique_values, counts)
    
    # calculate_embedding_distance(args, embeddings_2d, embeddings, labels, embeddings_label, label_unique_values, count=8)

    # plot_dynamics(args, labels, embeddings_label, label_unique_values)

    # plot_embedding_length_distributions(args, embeddings)
