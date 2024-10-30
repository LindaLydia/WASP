import torch
import numpy as np
import matplotlib.pyplot as plt
import jsonlines


# # votes = "/shared/project/PrivateGenerateEnhancement/src/results/multiGold_eval_on_real/with_real_few_shot_accumulate_votingCLASS_8_0.0_Cartography_sampling/gold_100_-1_0.0/bert-base-uncased/sentence-t5-base/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD1_FuseDataset1/0_1_init200_steps4_unbalance_temp3/fewshotK8_15_0.5/imdb/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__chatglm3-6b-base_1000__flan-t5-xl_1000/12345/real_vote_for_syn_tensor([   0,  180,  360,  540,  720,  900, 1080], device='cuda:0').pth"
# votes_within_and_others = "/shared/project/PrivateGenerateEnhancement/src/results/multiGold_eval_on_real/with_real_few_shot_accumulate_votingCLASS_8_0.0_Cartography_sampling/gold_100_-1_0.0/bert-base-uncased/sentence-t5-base/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD1_FuseDataset1/0_1_init200_steps4_unbalance_temp3/fewshotK8_15_0.5/imdb/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__chatglm3-6b-base_1000__flan-t5-xl_1000/12345/real_vote_for_syn_within_and_outside_tensor([   0,  180,  360,  540,  720,  900, 1080], device='cuda:0').pth"
# confidence_and_variability = "/shared/project/PrivateGenerateEnhancement/src/results/multiGold_eval_on_real/with_real_few_shot_accumulate_votingCLASS_8_0.0_Cartography_sampling/gold_100_-1_0.0/bert-base-uncased/sentence-t5-base/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD1_FuseDataset1/0_1_init200_steps4_unbalance_temp3/fewshotK8_15_0.5/imdb/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__chatglm3-6b-base_1000__flan-t5-xl_1000/12345/cross_model_confidence_and_variability_tensor([   0,  200,  400,  600,  800, 1000, 1200, 1300], device='cuda:0').pth"
# text_and_label = "/shared/project/PrivateGenerateEnhancement/src/results/multiGold_eval_on_real/with_real_few_shot_accumulate_votingCLASS_8_0.0_Cartography_sampling/gold_100_-1_0.0/bert-base-uncased/sentence-t5-base/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD1_FuseDataset1/0_1_init200_steps4_unbalance_temp3/fewshotK8_15_0.5/imdb/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__chatglm3-6b-base_1000__flan-t5-xl_1000/12345/sample_text_label.pth"

# iter2
confidence_and_variability = "/shared/project/PrivateGenerateEnhancement/src/results/multiGold_eval_on_real/with_real_few_shot_accumulate_votingCLASS_8_0.0_Cartography_sampling/gold_100_2_0.0/bert-base-uncased/sentence-t5-base/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD1_FuseDataset1/0_1_init200_steps4_unbalance_temp3/fewshotK8_15_0.5/imdb/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__chatglm3-6b-base_1000__flan-t5-xl_1000/12345/cross_model_confidence_and_variability_tensor([   0,  375,  772, 1147, 1495, 1863, 2400, 2500], device='cuda:0').pth"
votes_within_and_others = "/shared/project/PrivateGenerateEnhancement/src/results/multiGold_eval_on_real/with_real_few_shot_accumulate_votingCLASS_8_0.0_Cartography_sampling/gold_100_2_0.0/bert-base-uncased/sentence-t5-base/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD1_FuseDataset1/0_1_init200_steps4_unbalance_temp3/fewshotK8_15_0.5/imdb/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__chatglm3-6b-base_1000__flan-t5-xl_1000/12345/real_vote_for_syn_within_and_outside_tensor([   0,  337,  694, 1031, 1344, 1675, 2158], device='cuda:0').pth"
text_and_label = "/shared/project/PrivateGenerateEnhancement/src/results/multiGold_eval_on_real/with_real_few_shot_accumulate_votingCLASS_8_0.0_Cartography_sampling/gold_100_2_0.0/bert-base-uncased/sentence-t5-base/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD1_FuseDataset1/0_1_init200_steps4_unbalance_temp3/fewshotK8_15_0.5/imdb/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__chatglm3-6b-base_1000__flan-t5-xl_1000/12345/sample_text_label_iter1.pth"

# [[votes for label0], [votes for label1], ...], [[votes for label other than 0], [votes for label other than 1], ...]
nearest_sample_voting_per_party, nearest_other_class_sample_voting_per_party = torch.load(votes_within_and_others)
nearest_sample_voting_per_party = nearest_sample_voting_per_party.cpu().numpy()
nearest_other_class_sample_voting_per_party = nearest_other_class_sample_voting_per_party.cpu().numpy() 

# [confidence for all samples], [variability for all samples]
_confidence, _variability = torch.load(confidence_and_variability)
confidence, variability = [], []
for _c in _confidence:
    confidence += _c
for _v in _variability:
    variability += _v
confidence = np.asarray(confidence)
variability = np.asarray(variability)

text, label = torch.load(text_and_label)
print(len(text), len(label))
# print(label)

num_classes = nearest_sample_voting_per_party.shape[0]
print(f"{num_classes=}, {(num_classes<<1)=}")


with jsonlines.open("./analysis/votes_and_variability/high_scoring_sample.jsonl", 'w') as writer:
    for i_class in range(num_classes):
        sorted_indices = np.argsort(nearest_sample_voting_per_party[i_class])[::-1]
        for i in range(20):
            writer.write({
                "confidence": float(confidence[sorted_indices[i]]), "variability": float(variability[sorted_indices[i]]), "voting": float(nearest_sample_voting_per_party[i_class][sorted_indices[i]]),
                "type": "within-class", "real_sample_class": i_class,
                "text": text[sorted_indices[i]], "label": int(label[sorted_indices[i]])
            })
        writer.write({})
    for i_class in range(num_classes):
        sorted_indices = np.argsort(nearest_other_class_sample_voting_per_party[i_class])[::-1]
        for i in range(20):
            writer.write({
                "confidence": float(confidence[sorted_indices[i]]), "variability": float(variability[sorted_indices[i]]), "voting": float(nearest_other_class_sample_voting_per_party[i_class][sorted_indices[i]]),
                "type": "other-class", "real_sample_class": i_class,
                "text": text[sorted_indices[i]], "label": int(label[sorted_indices[i]])
            })
        writer.write({})


fig, axs = plt.subplots(nrows=2, ncols=(num_classes<<1), figsize=(10.5, 6), sharex=False, sharey=False)
for i_class in range(num_classes):
    axs[0][i_class+num_classes*0].scatter(confidence[nearest_sample_voting_per_party[i_class]!=0.0], nearest_sample_voting_per_party[i_class][nearest_sample_voting_per_party[i_class]!=0.0], alpha=0.5)
    axs[0][i_class+num_classes*0].set_xlabel("confidence")
    axs[0][i_class+num_classes*0].set_ylabel(f"within-class vote for class {i_class}")
    # axs[0][i_class+num_classes*0].set_title("confidence, within-class vote")
    axs[0][i_class+num_classes*1].scatter(confidence[nearest_other_class_sample_voting_per_party[i_class]!=0.0], nearest_other_class_sample_voting_per_party[i_class][nearest_other_class_sample_voting_per_party[i_class]!=0.0], alpha=0.5)
    axs[0][i_class+num_classes*1].set_xlabel("confidence")
    axs[0][i_class+num_classes*1].set_ylabel(f"other-class vote for class {i_class}")
    # axs[0][i_class+num_classes*1].set_title("confidence, other-class vote")
    axs[1][i_class+num_classes*0].scatter(variability[nearest_sample_voting_per_party[i_class]!=0.0], nearest_sample_voting_per_party[i_class][nearest_sample_voting_per_party[i_class]!=0.0], alpha=0.5)
    axs[1][i_class+num_classes*0].set_xlabel("variability")
    axs[1][i_class+num_classes*0].set_ylabel(f"within-class vote for class {i_class}")
    # axs[1][i_class+num_classes*0].set_title("variability, within-class vote")
    axs[1][i_class+num_classes*1].scatter(variability[nearest_other_class_sample_voting_per_party[i_class]!=0.0], nearest_other_class_sample_voting_per_party[i_class][nearest_other_class_sample_voting_per_party[i_class]!=0.0], alpha=0.5)
    axs[1][i_class+num_classes*1].set_xlabel("variability")
    axs[1][i_class+num_classes*1].set_ylabel(f"other-class vote for class {i_class}")
    # axs[1][i_class+num_classes*1].set_title("variability, other-class vote")
    # axs[ir][i].legend(loc='upper center')
plt.tight_layout()
plt.savefig(f'./analysis/votes_and_variability/temp.png',dpi=200)


fig, axs = plt.subplots(nrows=1, ncols=(num_classes<<1), figsize=(10.5, 4), sharex=False, sharey=False)
for i_class in range(num_classes):
    # axs[i_class+num_classes*0].scatter(variability, confidence, c=nearest_sample_voting_per_party[i_class], cmap='cool', alpha=0.2, s=10)
    axs[i_class+num_classes*0].scatter(variability[nearest_sample_voting_per_party[i_class]!=0.0], confidence[nearest_sample_voting_per_party[i_class]!=0.0], c=nearest_sample_voting_per_party[i_class][nearest_sample_voting_per_party[i_class]!=0.0], cmap='Wistia', alpha=0.5, s=30)
    # axs[i_class+num_classes*0].scatter(variability, confidence, c=nearest_sample_voting_per_party[i_class], cmap='YlGn', alpha=0.9, s=10)
    axs[i_class+num_classes*0].set_xlabel("variability")
    axs[i_class+num_classes*0].set_ylabel("confidence")
    axs[i_class+num_classes*0].set_title(f"within-class vote for class {i_class}")
    # axs[i_class+num_classes*1].scatter(variability, confidence, c=nearest_other_class_sample_voting_per_party[i_class], cmap='cool', alpha=0.2, s=10)
    axs[i_class+num_classes*1].scatter(variability[nearest_other_class_sample_voting_per_party[i_class]!=0.0], confidence[nearest_other_class_sample_voting_per_party[i_class]!=0.0], c=nearest_other_class_sample_voting_per_party[i_class][nearest_other_class_sample_voting_per_party[i_class]!=0.0], cmap='Wistia', alpha=0.5, s=30)
    # axs[i_class+num_classes*1].scatter(variability, confidence, c=nearest_other_class_sample_voting_per_party[i_class], cmap='YlGn', alpha=0.9, s=10)
    axs[i_class+num_classes*1].set_xlabel("variability")
    axs[i_class+num_classes*1].set_ylabel("confidence")
    axs[i_class+num_classes*1].set_title(f"other-class vote for class {i_class}")
    # axs[i_class+num_classes*1].set_title("confidence, other-class vote")
    # axs[ir][i].legend(loc='upper center')
plt.tight_layout()
plt.savefig(f'./analysis/votes_and_variability/temp2.png',dpi=200)
