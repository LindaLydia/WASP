import os, sys
import torch
import numpy as np

# torch_file_name = "/home/DAIR/zouty/ModelFederation/PrivateGenerateEnhancement/src/results/eval_on_real/with_real_few_shot_accumulate_voting_1/bert-base-uncased/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD1_FuseDataset1/0_1_init20_steps4/fewshotK8_5_0.5/imdb/gpt2-xl_100__llama-2-7b-chat-hf_100__vicuna-7b-1.5v_100__opt-6.7b_100__chatglm3-6b-base_100__flan-t5-xl_100/12345/real_vote_for_syn_tensor([  0,  18,  36,  54,  72,  90, 108, 117], device='cuda:0').pth"
# torch_file_name = "/home/DAIR/zouty/ModelFederation/PrivateGenerateEnhancement/src/results/eval_on_real/with_real_few_shot_accumulate_voting_1/bert-base-uncased/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD1_FuseDataset1/0_1_init20_steps4/fewshotK8_5_0.5/imdb/gpt2-xl_100__llama-2-7b-chat-hf_100__vicuna-7b-1.5v_100__opt-6.7b_100__chatglm3-6b-base_100__flan-t5-xl_100/12345/real_vote_for_syn_tensor([  0,  36,  72, 108, 144, 180, 216, 225], device='cuda:0').pth"
# voting_result, model_score, model_sample_counts = torch.load(torch_file_name)
# print(voting_result, model_score, model_sample_counts)

torch_file_dir = "/home/DAIR/zouty/ModelFederation/PrivateGenerateEnhancement/src/results/eval_on_real/with_real_few_shot_accumulate_voting_8/bert-base-uncased/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD1_FuseDataset1/0_1_init20_steps4/fewshotK8_5_0.5/imdb/gpt2-xl_100__llama-2-7b-chat-hf_100__vicuna-7b-1.5v_100__opt-6.7b_100__chatglm3-6b-base_100__flan-t5-xl_100/12345/real_vote_for_syn_tensor([  0,  18,  36,  54,  72,  90, 108, 117], device='cuda:0').pth"
torch_file_dir = "/home/DAIR/zouty/ModelFederation/PrivateGenerateEnhancement/src/results/eval_on_real/with_real_few_shot_accumulate_voting_8/bert-base-uncased/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD1_FuseDataset1/0_1_init20_steps4/fewshotK8_5_0.5/imdb/gpt2-xl_100__llama-2-7b-chat-hf_100__vicuna-7b-1.5v_100__opt-6.7b_100__chatglm3-6b-base_100__flan-t5-xl_100/12345/"
for dirpath, dirnames, filenames in os.walk(torch_file_dir):
    # print(f"Checking directory: {dirpath}")
    # # Loop through all files in the current directory
    for filename in filenames:
        if 'real_vote_for_syn_tensor' in filename:
            torch_file_name = os.path.join(dirpath, filename)
            voting_result, model_score, model_sample_counts = torch.load(torch_file_name)
            # print(voting_result, model_score, model_sample_counts)
            # print(f"{model_score=}\n{model_sample_counts=}")
            print(f"{model_score=}")

