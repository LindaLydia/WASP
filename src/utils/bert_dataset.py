import torch
# from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
import json
import random
import copy

# Load and preprocess data from the jsonl file
class TokenizedDataset(Dataset):
    def __init__(self, file_path='', text_column='text', label_column='label', index_column='idx', is_syn_column='is_syn', tokenizer=None, max_length=512, device='cpu', max_sample=-1, small_dataset_shuffle=False):
        self.text = []
        self.ids = []
        self.attention_mask = []
        self.label = []
        self.idx = []
        self.is_syn = []
        if file_path == '' or file_path == ['', '']:
            self.ids = torch.tensor([self.ids], dtype=torch.int64).to(device)
            self.attention_mask = torch.tensor([self.attention_mask], dtype=torch.int64).to(device)
            self.label = torch.tensor(self.label, dtype=torch.int64).to(device)
            self.idx = torch.tensor(self.idx, dtype=torch.int64).to(device)
            self.is_syn = torch.tensor(self.is_syn, dtype=torch.bool).to(device)
        elif type(file_path) == type(['list']):
            syn_file_path = file_path[0]
            real_file_path = file_path[1]
            print(f"[info] synthetic data file path is {syn_file_path}")
            print(f"[info] real data file path is {real_file_path}")
            for _file_path, _is_syn in zip(file_path, [1,0]):
                if _file_path == '':
                    continue
                with open(_file_path, 'r') as file:
                    counter = 0
                    for line in file:
                        item = json.loads(line.strip())
                        text = item[text_column]
                        label = int(item[label_column])  # Assuming your jsonl file contains a 'label' field
                        # # idx = int(item[index_column])
                        syn_indicator = _is_syn
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
                        self.idx.append(counter)
                        self.is_syn.append(syn_indicator)
                        counter += 1
                        if max_sample > 0 and counter == max_sample:
                            break
        else:
            if ('imdb' in file_path and (0 < max_sample < 1000)) or small_dataset_shuffle==True:
                lines = []
                with open(file_path, 'r') as file:
                    for line in file:
                        lines.append(line)
                random.shuffle(lines)
                counter = 0
                for line in lines:
                    item = json.loads(line.strip())
                    text = item[text_column]
                    label = item[label_column]  # Assuming your jsonl file contains a 'label' field
                    # idx = item[index_column]
                    if is_syn_column != None:
                        syn_indicator = bool(item[is_syn_column])
                    else:
                        syn_indicator = False
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
                    self.idx.append(counter) # append counter this time, not the original idx
                    self.is_syn.append(syn_indicator)
                    counter += 1
                    if counter == max_sample:
                        break
            else:
                with open(file_path, 'r') as file:
                    counter = 0
                    for line in file:
                        item = json.loads(line.strip())
                        text = item[text_column]
                        label = int(item[label_column])  # Assuming your jsonl file contains a 'label' field
                        # idx = int(item[index_column])
                        if is_syn_column != None: # only for training data, test data should always be real data
                            syn_indicator = bool(item[is_syn_column])
                            # if syn_indicator == False: # only use synthetic data
                            #     co
                            # ntinue
                            # if syn_indicator == True: # only use real data
                            #     continue
                        else:
                            syn_indicator = False
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
                        self.idx.append(counter)
                        self.is_syn.append(syn_indicator)
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
            self.is_syn = torch.tensor(self.is_syn).bool()
        # print(self.ids.shape, self.attention_mask.shape, self.label.shape, self.idx.shape, self.is_syn.shape)
        # print(self.ids.dtype, self.attention_mask.dtype, self.label.dtype, self.idx.dtype)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        return self.ids[index], self.attention_mask[index], self.label[index], self.idx[index] #, self.is_syn[index]

    def clear_and_copy_dataset(self, source_dataset, indices, len_LLM, new_idx=True):
        self.text = [] # clear all the samples
        self.ids = [] # clear all the samples
        self.attention_mask = [] # clear all the samples
        self.label = [] # clear all the samples
        self.idx = [] # clear all the samples
        self.is_syn = [] # clear all the samples
        for i in range(len_LLM):
            if len(indices) == 0:
                local_indices = [ix for ix in range(len(source_dataset[i].text))]
            else:
                local_indices = indices
            self.text += [source_dataset[i].text[ix] for ix in local_indices]
            self.ids += [source_dataset[i].ids[ix] for ix in local_indices]
            self.attention_mask += [source_dataset[i].attention_mask[ix] for ix in local_indices]
            self.label += [source_dataset[i].label[ix] for ix in local_indices]
            self.idx += [source_dataset[i].idx[ix] for ix in local_indices]
            self.is_syn += [source_dataset[i].is_syn[ix] for ix in local_indices]
        # self.ids = torch.stack(self.ids).squeeze().to(args.device)
        # self.attention_mask = torch.stack(self.attention_mask).squeeze().to(args.device)
        # self.label = torch.tensor(self.label).long().to(args.device)
        # self.idx = torch.tensor(self.idx).long().to(args.device)
        if new_idx == True:
            self.idx = [ix for ix in range(len(self.label))]
        self.ids = torch.stack(self.ids).squeeze()
        self.attention_mask = torch.stack(self.attention_mask).squeeze()
        self.label = torch.tensor(self.label).long()
        self.idx = torch.tensor(self.idx).long()
        self.is_syn = torch.tensor(self.is_syn).bool()
    
    def copy_dataset(self, source_dataset, indices, new_idx=True):
        self.text = [copy.deepcopy(source_dataset.text[ix]) for ix in indices]
        self.ids = copy.deepcopy(source_dataset.ids[indices])
        self.attention_mask = copy.deepcopy(source_dataset.attention_mask[indices])
        self.label = copy.deepcopy(source_dataset.label[indices])
        if new_idx:
            self.idx = torch.tensor([_i for _i in range(len(indices))]).long()
        else:
            self.idx = copy.deepcopy(source_dataset.idx[indices])
        self.is_syn = copy.deepcopy(source_dataset.is_syn[indices])
        
    def copy_selected_dataset(self, source_dataset, selected_sample_rows, selected_sample_columns):
        self.text = [] # clear all the samples
        self.ids = [] # clear all the samples
        self.attention_mask = [] # clear all the samples
        self.label = [] # clear all the samples
        self.idx = [] # clear all the samples
        self.is_syn = [] # clear all the samples
        _id = 0
        for row, column in zip(selected_sample_rows,selected_sample_columns):
            self.text += [source_dataset[row].text[column]]
            self.ids += [source_dataset[row].ids[column]]
            self.attention_mask += [source_dataset[row].attention_mask[column]]
            self.label += [source_dataset[row].label[column] ]
            self.idx += [_id]
            self.is_syn += [source_dataset[row].is_syn[column]]
            _id += 1
        self.ids = torch.stack(self.ids).squeeze()
        self.attention_mask = torch.stack(self.attention_mask).squeeze()
        self.label = torch.tensor(self.label).long()
        self.idx = torch.tensor(self.idx).long()
        self.is_syn = torch.tensor(self.is_syn).bool()


# Load and preprocess data from the jsonl file
class TokenizedQADataset(Dataset):
    # {"context": "He survived this on Day 47, but was evicted on Day 52 through the backdoor after receiving only 1.52% of the overall final vote to win.", "question": "How many days did joshuah alagappan survive in big brother?", "answers": {"answer_start": [20], "text": ["Day 47"]}, "id": "ID_1982-0-0"}
    def __init__(self, file_path='', context_column='context', question_column='question', label_column='answers', index_column='idx', tokenizer=None, max_length=512, device='cpu', max_sample=-1, small_dataset_shuffle=False):
        self.context = []
        self.question = []
        self.ids = []
        self.attention_mask = []
        self.offset_mapping = []
        self.sample_mapping = []
        self.label = []
        self.idx = []
        if file_path == '':
            self.ids = torch.tensor([self.ids],dtype=torch.int64).to(device)
            self.attention_mask = torch.tensor([self.attention_mask],dtype=torch.int64).to(device)
            self.offset_mapping = torch.tensor([self.offset_mapping],dtype=torch.int64).to(device)
            self.sample_mapping = torch.tensor([self.sample_mapping],dtype=torch.int64).to(device)
            # self.label = torch.tensor(self.label,dtype=torch.int64).to(device)
            self.idx = torch.tensor(self.idx,dtype=torch.int64).to(device)
        else: 
            with open(file_path, 'r') as file:
                counter = 0
                for line in file:
                    item = json.loads(line.strip())
                    context = item[context_column]
                    question = item[question_column]
                    label = item[label_column]  # Assuming your jsonl file contains a 'label' field
                    idx = item[index_column]
                    tokenized = tokenizer.encode_plus(
                        question,
                        context,
                        add_special_tokens=True,
                        max_length=max_length,
                        padding='max_length',
                        truncation="only_second",
                        return_attention_mask=True,
                        return_tensors='pt',
                        return_overflowing_tokens=True,
                        return_offsets_mapping=True,
                    )
                    self.context.append(context)
                    self.question.append(question)
                    self.ids.append(tokenized['input_ids'])
                    # self.ids.append(tokenized)
                    self.attention_mask.append(tokenized['attention_mask'])
                    self.offset_mapping.append(tokenized["offset_mapping"])
                    self.sample_mapping.append(tokenized["overflow_to_sample_mapping"])
                    self.label.append(label)
                    self.idx.append(idx)
                    counter += 1
                    if max_sample > 0 and counter == max_sample:
                        break
            # self.ids = torch.stack(self.ids).squeeze()
            # self.attention_mask = torch.stack(self.attention_mask).squeeze()
            # self.offset_mapping = torch.stack(self.offset_mapping).squeeze()
            # self.sample_mapping = torch.stack(self.sample_mapping).squeeze()
            # # self.label = torch.tensor(self.label).long()
            self.idx = torch.tensor(self.idx).long()
        # print(self.ids.shape, self.attention_mask.shape, self.label.shape, self.idx.shape)
        # print(self.ids.dtype, self.attention_mask.dtype, self.label.dtype, self.idx.dtype)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        return self.context[index], self.ids[index], self.attention_mask[index], self.offset_mapping[index], self.sample_mapping[index], self.label[index], self.idx[index]

