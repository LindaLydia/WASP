import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from collections import OrderedDict


def FedAVG(args, model_list, aggregation_weights=None):
    assert len(model_list) == args.gold_party_num, f"[ERROR] {len(model_list)} model(s) for FedAVG, but have {args.gold_party_num} private data parties"
    if aggregation_weights == None:
        aggregation_weights = [(1/len(model_list))]*len(model_list)
    
    fedAVG_model = copy.deepcopy(args.fused_model)
    update_state = OrderedDict()

    for k, model in enumerate(model_list):
        local_state = model.state_dict()
        for key in fedAVG_model.state_dict().keys():
            if k == 0:
                update_state[key] = local_state[key] * aggregation_weights[k]
            else:
                update_state[key] += local_state[key] * aggregation_weights[k]

    fedAVG_model.load_state_dict(update_state)
    return fedAVG_model