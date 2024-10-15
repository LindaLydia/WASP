import sys, os
import matplotlib.pyplot as plt
import numpy as np
import torch
import re



voting_file_path = "/home/DAIR/zouty/ModelFederation/PrivateGenerateEnhancement/src/results/eval_on_real/with_real_few_shot_accumulate_votingCLASS_8_Cartography_sampling/bert-base-uncased/sentence-t5-base/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD1_FuseDataset1/0_1_init200_steps4_unbalance_temp3/fewshotK8_5_0.5/imdb/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__chatglm3-6b-base_1000__flan-t5-xl_1000/12345/real_vote_for_syn_tensor([   0,  180,  360,  540,  720,  900, 1080, 1170], device='cuda:0').pth"
# voting_file_path = "/home/DAIR/zouty/ModelFederation/PrivateGenerateEnhancement/src/results/eval_on_real/with_real_few_shot_accumulate_votingCLASS_8_Cartography_sampling/bert-base-uncased/sentence-t5-base/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD1_FuseDataset1/0_1_init200_steps4_unbalance_temp3/fewshotK8_5_0.5/imdb/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__chatglm3-6b-base_1000__flan-t5-xl_1000/12345/real_vote_for_syn_tensor([   0,  292,  852, 1279, 1572, 1864, 2157, 2247], device='cuda:0').pth"

voting_file_path = "/home/DAIR/zouty/ModelFederation/PrivateGenerateEnhancement/src/results/eval_on_real/with_real_few_shot_accumulate_votingALL_8_Cartography_sampling/bert-base-uncased/sentence-t5-base/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD1_FuseDataset1/0_1_init200_steps4_unbalance_temp3/fewshotK8_5_0.5/imdb/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__chatglm3-6b-base_1000__flan-t5-xl_1000/12345/real_vote_for_syn_tensor([   0,  180,  360,  540,  720,  900, 1080, 1170], device='cuda:0').pth"

samples = re.findall(r'\[(.*?)\]', voting_file_path)[0]
samples = samples.replace(' ','')
print(f"{samples=}")
accumulated_samples = samples.split(',')
accumulated_samples = [int(a) for a in accumulated_samples]
samples_per_llm = [accumulated_samples[i]-accumulated_samples[i-1] for i in range(1,len(accumulated_samples))]
print(f"{samples_per_llm=}")

nearest_sample_voting, model_voting_score, local_accumulate_samples = torch.load(voting_file_path, map_location=torch.device('cpu'))
print(f"{nearest_sample_voting.shape=}")

nrow = len(samples_per_llm)-1
ncolumn = 1
fig, axs = plt.subplots(nrows=nrow, ncols=ncolumn, figsize=(8, 6), sharex=True, sharey=True)

for i, ax in enumerate(axs):
    voting_values = nearest_sample_voting[accumulated_samples[i]:accumulated_samples[i+1]].numpy()
    print(f"{voting_values=}")
    idxs = [j for j in range(accumulated_samples[i+1]-accumulated_samples[i])]
    ax.set_yscale('log')
    for x, y in zip(idxs, voting_values):
        ax.bar(x,y,width=1.,color='blue',alpha=0.5)

plt.tight_layout()
if not os.path.exists('./figure/voting/'):
    os.makedirs('./figure/voting/')
print(f'./figure/voting/{re.findall(r"with_real_few_shot_accumulate_(.*?)/bert-base-uncased/", voting_file_path)[0]}_{accumulated_samples[-2]}.png')
plt.savefig(f'./figure/voting/{re.findall(r"with_real_few_shot_accumulate_(.*?)/bert-base-uncased/", voting_file_path)[0]}_{accumulated_samples[-2]}.png',dpi=200)
