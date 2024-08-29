import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import copy

from tqdm import tqdm

from transformers import BertModel, BertTokenizer

from utils.constant import MODEL_PATH
from utils.basic_utils import merge_all_dataset


def find_nearest_syn_samples(args, train_data, gold_loader, count):
    gold_data = gold_loader.dataset
    total_data = merge_all_dataset(args, train_data)
    print(f"{args.accumulate_sampels=}")
    nearest_sample_voting = [0.0]*len(total_data)
    print(f"{len(nearest_sample_voting)=}")
    if args.small_model_name.upper() == 'LSTM':
        pass
    elif 'bert' in args.small_model_name.lower():
        embedding_model = BertModel.from_pretrained(MODEL_PATH['bert-base-uncased'])

    gold_embedding = get_embedding(args, embedding_model, gold_loader)
    total_syn_loader = DataLoader(total_data, batch_size=args.train_batch_size, shuffle=False)
    syn_embedding = get_embedding(args, embedding_model, total_syn_loader)

    for _gold in gold_embedding:
        # nearest_distance = 10000
        # nearest_id = -1
        # for _id, _syn in enumerate(syn_embedding):
        #     _distance = torch.norm(_gold-_syn)
        #     print(f"{_distance=} between {_gold}, {_syn} with {_id=}")
        #     if _distance < nearest_distance:
        #         nearest_distance = _distance
        #         nearest_id = _id
        # nearest_sample_voting[nearest_id] += 1
        distances = torch.sqrt(((syn_embedding - _gold[:, None])**2).sum(dim=2))
        print(f"{distances.shape}")
        _, top_k_indices = torch.topk(distances, k=1, largest=False)
        top_k_vectors = syn_embedding[top_k_indices]
        print(f"{top_k_vectors=}")
        for _i, _indice in enumerate(top_k_indices):
            nearest_sample_voting[_indice] += 1/(2**_i)
    
    value_index_pairs = [(nearest_sample_voting[i], i) for i in range(len(nearest_sample_voting))]  
    sorted_pairs = sorted(value_index_pairs, key=lambda x: x[0], reverse=True)  
    top_k_syn_samples_indices = [pair[1] for pair in sorted_pairs[:args.count]] 
    print(f"{top_k_syn_samples_indices=}")
    
    # change into [(im, ic), (im, ic), ..., (im, ic)] format
    selected_sample_model_position_list = []
    for ic in range(len(top_k_syn_samples_indices)):
        model_idx = -1
        for im in range(args.len_LLM):
            if args.accumulate_sampels[im] <= top_k_syn_samples_indices[ic] < args.accumulate_sampels[im+1]:
                model_idx = im
                break
        assert model_idx != -1, f"[ERROR] sample #{top_k_syn_samples_indices[ic]} not mapped into {args.accumulate_sampels}"
        selected_sample_model_position_list.append((model_idx,top_k_syn_samples_indices[ic]-args.accumulate_sampels[model_idx]))
    return selected_sample_model_position_list


def get_embedding(args, model, train_iter):
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
