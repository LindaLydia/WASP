import numpy as np
from scipy.linalg import sqrtm
import torch
import torch.nn as nn
import random
import copy

from src.utils.bert_dataset import TokenizedDataset
from src.utils.constant import SMALL_MODEL_WITH_TOKENIZER

class RBF(nn.Module):

    def __init__(self, device, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels).to(device) - n_kernels // 2)
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)


class MMDLoss(nn.Module):

    def __init__(self, device):
        super().__init__()
        self.kernel = RBF(device=device)

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))

        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY


def calculate_fid(act1, act2):
    # calculate mean and covariance statistics
    # print(act1.shape, type(act1))
    # print(act2.shape, type(act2))
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    sigma1[np.isnan(sigma1)] = 0.0
    sigma1[np.isinf(sigma1)] = 1.0E30
    sigma2[np.isnan(sigma2)] = 0.0
    sigma2[np.isinf(sigma2)] = 1.0E30
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def merge_all_dataset(args, datasets, max_sample_count_for_total=100):
    if max_sample_count_for_total != -1:
        max_sample_count_for_each = max_sample_count_for_total // len(datasets)
    else:
        max_sample_count_for_each = -1
    # ############### prepare total_data ###############
    # accumulate_sampels = [0]
    if args.small_model_name.upper() == 'LSTM':
        total_data = []
        for _dataset in datasets:
            _dataset_examples = _dataset.examples[:-1] + [_dataset.examples[-1]]
            random.shuffle(_dataset_examples)
            if max_sample_count_for_each != -1:
                print(f"{len(_dataset_examples)=}, use {min(len(_dataset_examples),max_sample_count_for_each)} samples")
                total_data += copy.deepcopy(_dataset_examples[:min(len(_dataset_examples),max_sample_count_for_each)])
            else:
                print(f"{len(_dataset_examples)=}, use {len(_dataset_examples)} samples")
                total_data += copy.deepcopy(_dataset_examples[:])
            # accumulate_sampels.append(accumulate_sampels[-1]+len(_dataset_examples)) 
        for _i in range(len(total_data)):
            total_data[_i].idx = _i
        total_dataset = data.Dataset(total_data, datasets[0].fields)
    elif any(substring in args.small_model_name.lower() for substring in SMALL_MODEL_WITH_TOKENIZER):
        _id = 0
        total_dataset = TokenizedDataset(
            file_path=(''),
        )        
        total_dataset.text = [] # clear all the samples
        total_dataset.ids = [] # clear all the samples
        total_dataset.attention_mask = [] # clear all the samples
        total_dataset.label = [] # clear all the samples
        total_dataset.idx = [] # clear all the samples
        # total_dataset.is_syn = [] # clear all the samples
        for row in range(len(datasets)):
            # accumulate_sampels.append(accumulate_sampels[-1]+len(datasets[row].idx))
            idx_list = [_i for _i in range(len(datasets[row].idx))]
            if max_sample_count_for_each != -1 and max_sample_count_for_each < len(datasets[row].idx):
                random.shuffle(idx_list)
                idx_list = idx_list[:min(len(datasets[row].idx),max_sample_count_for_each)]
                print(f"{len(datasets[row].idx)=}, use {min(len(datasets[row].idx),max_sample_count_for_each)} samples")
            else:
                idx_list = idx_list[:]
                print(f"{len(datasets[row].idx)=}, use {len(datasets[row].idx)} samples")
            for column in idx_list:
                total_dataset.text += [datasets[row].text[column]]
                total_dataset.ids += [datasets[row].ids[column]]
                total_dataset.attention_mask += [datasets[row].attention_mask[column]]
                total_dataset.label += [datasets[row].label[column]]
                total_dataset.idx += [_id]
                # total_dataset.is_syn += [datasets[row].is_syn[column]]
                _id += 1
    # accumulate_sampels = torch.tensor(accumulate_sampels, dtype=torch.long).to(args.device)
    # ############### prepare total_data ###############
    return total_dataset
