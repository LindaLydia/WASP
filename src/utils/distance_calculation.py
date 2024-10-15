import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import copy

from tqdm import tqdm

from transformers import BertModel, BertTokenizer
from sentence_transformers import SentenceTransformer

from utils.constant import MODEL_PATH, SENTENCE_TRANSFORMERS_PATH
from utils.basic_utils import merge_all_dataset



def find_nearest_syn_samples(args, train_data, gold_loader, count, sample_limitation=None):
    gold_data = gold_loader.dataset
    print(f"{len(train_data)=}, {[len(train_data[i].idx) for i in range(len(train_data))]}")
    print(f"{args.len_LLM}")
    total_data = merge_all_dataset(args, train_data, max_sample_count_for_total=-1)
    local_accumulate_samples = [0]
    for im in range(len(train_data)):
        local_accumulate_samples.append(local_accumulate_samples[-1]+len(train_data[im].idx))
    local_accumulate_samples = torch.tensor(local_accumulate_samples, dtype=torch.long).to(args.device)
    print(f"{len(total_data.idx)=},{len(total_data)=}")
    print(f"{local_accumulate_samples=}")
    nearest_sample_voting = [0.0]*len(total_data)
    print(f"{len(nearest_sample_voting)=}")
    if args.sentence_transformer.upper() != 'NONE':
        # use specified sentence transformer
        embedding_model = SentenceTransformer(SENTENCE_TRANSFORMERS_PATH[args.sentence_transformer])
    else:
        # use the small local model as sentence embedding model
        if args.small_model_name.upper() == 'LSTM':
            pass
        elif 'bert' in args.small_model_name.lower():
            embedding_model = BertModel.from_pretrained(MODEL_PATH['bert-base-uncased'])

    gold_embedding, gold_label = get_embedding(args, embedding_model, gold_loader)
    gold_embedding = torch.tensor(gold_embedding)
    # print(f"{gold_embedding=}")
    total_syn_loader = DataLoader(total_data, batch_size=args.train_batch_size, shuffle=False)
    syn_embedding, syn_label = get_embedding(args, embedding_model, total_syn_loader)
    syn_embedding = torch.tensor(syn_embedding)
    # print(f"{syn_embedding=}")
    print(f"{gold_embedding.shape=}, {syn_embedding.shape=}")

    if args.voting_range.lower() == 'all':
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
            distances = torch.sqrt(((syn_embedding - _gold)**2).sum(dim=-1))
            print(f"{distances.shape}")
            _, top_k_indices = torch.topk(distances, k=args.real_voting_votes, largest=False)
            # top_k_vectors = syn_embedding[top_k_indices]
            # print(f"{top_k_vectors=}")
            for _i, _indice in enumerate(top_k_indices):
                nearest_sample_voting[_indice] += 1/(2**_i)
    elif args.voting_range.lower() == 'class':
        unique_gold_label = np.unique(gold_label)
        syn_embedding_each_class = {}
        for _u_label in unique_gold_label:
            syn_embedding_each_class[_u_label] = (syn_embedding[syn_label==_u_label], np.where(syn_label==_u_label)[0])
        print(f"{syn_embedding_each_class=}")
        for _gold, _gold_label in zip(gold_embedding, gold_label):
            distances = torch.sqrt(((syn_embedding_each_class[_gold_label][0] - _gold)**2).sum(dim=-1))
            print(f"{distances.shape}")
            _, top_k_indices = torch.topk(distances, k=args.real_voting_votes, largest=False)
            # top_k_vectors = syn_embedding_each_class[_gold_label][0][top_k_indices]
            # print(f"{top_k_vectors=}")
            for _i, _indice in enumerate(top_k_indices):
                print(f"{_indice=} in class#{_gold_label}, which should be mapped to {syn_embedding_each_class[_gold_label][1][_indice]} in the total dataset")
                nearest_sample_voting[syn_embedding_each_class[_gold_label][1][_indice]] += 1/(2**_i)

    # ###### calculate model importance before eliminating some of the samples from being selected ######
    # ############# calculate model importance based on voting result #############
    nearest_sample_voting = np.asarray(nearest_sample_voting)
    nearest_sample_voting = nearest_sample_voting / np.sum(nearest_sample_voting)
    model_voting_score = np.asarray([0.0]*args.len_LLM)
    for im in range(args.len_LLM):
        print(f"LLM #{im} has #sample={len(train_data[im].idx)}, ranging from [{local_accumulate_samples[im]},{local_accumulate_samples[im+1]}]")
        voting_score_sum = np.sum(nearest_sample_voting[local_accumulate_samples[im]:local_accumulate_samples[im+1]])
        model_voting_score[im] = (voting_score_sum * len(total_data.idx)) / (len(train_data[im].idx))
    torch.save((torch.tensor(nearest_sample_voting),torch.tensor(model_voting_score),local_accumulate_samples), f"{args.result_file_path}/real_vote_for_syn_{local_accumulate_samples}.pth")
    print(f"{[i for i in range(len(nearest_sample_voting))]=}, {count=}, {len(nearest_sample_voting)=}, {nearest_sample_voting.shape=}")
    # ############# calculate model importance based on voting result #############
    
    # ############# eliminate samples that are not in sample_limitation if sample_limitation!=None #############
    if sample_limitation != None:
        sample_mask = [0.0]*len(total_data)
        for _idx in sample_limitation:
            sample_mask[_idx] = 1.0
        nearest_sample_voting = [nearest_sample_voting[_i]*sample_mask[_i] for _i in range(len(total_data))]
        nearest_sample_voting = np.asarray(nearest_sample_voting)
        nearest_sample_voting = nearest_sample_voting / np.sum(nearest_sample_voting)
    # ############# eliminate samples that are not in sample_limitation if sample_limitation!=None #############
    
    if args.voted_sample_select == 'sampling':
        # ########### nearest_sample_voting is the probability for sampling ###########
        # ########### random sample <#count> samples with the highest probability value (nearest_sample_voting value) ###########
        if np.count_nonzero(nearest_sample_voting) < count:
            print(f"None zero values in nearest_sample_voting is {np.count_nonzero(nearest_sample_voting)}, < number of required in-context sample {count}")
            top_k_syn_samples_indices = np.random.choice([i for i in range(len(nearest_sample_voting))], size=count, p=nearest_sample_voting, replace=True)
        else:
            print(f"None zero values in nearest_sample_voting is {np.count_nonzero(nearest_sample_voting)}, >= number of required in-context sample {count}")
            top_k_syn_samples_indices = np.random.choice([i for i in range(len(nearest_sample_voting))], size=count, p=nearest_sample_voting, replace=False)
        print(f"{top_k_syn_samples_indices=}")
        # ########### random sample <#count> samples with the highest probability value (nearest_sample_voting value) ###########
    elif args.voted_sample_select == 'top':
        # ########### sample the top-<#count> samples with the highest probability value (nearest_sample_voting value) ###########
        if np.count_nonzero(nearest_sample_voting) < count:
            print(f"None zero values in nearest_sample_voting is {np.count_nonzero(nearest_sample_voting)}, < number of required in-context sample {count}")
            top_k_syn_samples_indices = np.random.choice([i for i in range(len(nearest_sample_voting))], size=count, p=nearest_sample_voting, replace=True)
        else:
            value_index_pairs = [(nearest_sample_voting[i], i) for i in range(len(nearest_sample_voting))]  
            sorted_pairs = sorted(value_index_pairs, key=lambda x: x[0], reverse=True)  
            # print(f"{count=}")
            top_k_syn_samples_indices = [pair[1] for pair in sorted_pairs[:count]] 
        print(f"{top_k_syn_samples_indices=}")
        # ########### sample the top-<#count> samples with the highest probability value (nearest_sample_voting value) ###########
    
    # change into [(im, ic), (im, ic), ..., (im, ic)] format
    selected_sample_model_position_list = []
    for ic in range(len(top_k_syn_samples_indices)):
        model_idx = -1
        for im in range(args.len_LLM):
            if local_accumulate_samples[im] <= top_k_syn_samples_indices[ic] < local_accumulate_samples[im+1]:
                model_idx = im
                break
        assert model_idx != -1, f"[ERROR] sample #{top_k_syn_samples_indices[ic]} not mapped into {local_accumulate_samples}"
        selected_sample_model_position_list.append((model_idx,(top_k_syn_samples_indices[ic]-local_accumulate_samples[model_idx]).item()))
    return selected_sample_model_position_list, nearest_sample_voting, model_voting_score


def get_embedding(args, model, train_iter):
    if args.sentence_transformer.upper() != 'NONE':
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
