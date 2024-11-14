


import torch
import numpy as np
import matplotlib.pyplot as plt
import jsonlines
# from scipy.stats import entropy
import re

# 6plm, gold=[76,24], iter1, 1080
confidence_and_variability_file = "/shared/project/PrivateGenerateEnhancement/src/results/multiGold_eval_on_real/with_real_few_shot_accumulate_votingCLASS_8_0.0_GoldChange_top/gold_100_1_0.0_OODgold/bert-base-uncased/sentence-t5-base/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD1_FuseDataset1/0_1_init200_steps4_unbalance_temp3/fewshotK8_15_0.5/imdb/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__chatglm3-6b-base_1000__flan-t5-xl_1000/0/iter0_self_confidence_variability.pth"
votes_within_and_others = "/shared/project/PrivateGenerateEnhancement/src/results/multiGold_eval_on_real/with_real_few_shot_accumulate_votingCLASS_8_0.0_GoldChange_top/gold_100_1_0.0_OODgold/bert-base-uncased/sentence-t5-base/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD1_FuseDataset1/0_1_init200_steps4_unbalance_temp3/fewshotK8_15_0.5/imdb/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__chatglm3-6b-base_1000__flan-t5-xl_1000/0/real_vote_for_syn_tensor([   0,  180,  360,  540,  720,  900, 1080], device='cuda:0').pth"
pred_change_file = "/shared/project/PrivateGenerateEnhancement/src/results/multiGold_eval_on_real/with_real_few_shot_accumulate_votingCLASS_8_0.0_GoldChange_top/gold_100_1_0.0_OODgold/bert-base-uncased/sentence-t5-base/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD1_FuseDataset1/0_1_init200_steps4_unbalance_temp3/fewshotK8_15_0.5/imdb/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__chatglm3-6b-base_1000__flan-t5-xl_1000/0/iter0_sample_pred_change.pth"
pred_file = "/shared/project/PrivateGenerateEnhancement/src/results/multiGold_eval_on_real/with_real_few_shot_accumulate_votingCLASS_8_0.0_GoldChange_top/gold_100_1_0.0_OODgold/bert-base-uncased/sentence-t5-base/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD1_FuseDataset1/0_1_init200_steps4_unbalance_temp3/fewshotK8_15_0.5/imdb/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__chatglm3-6b-base_1000__flan-t5-xl_1000/0/iter0_sample_pred.pth"

# 6plm, gold=[50,50], iter1, 1080
confidence_and_variability_file = "/shared/project/PrivateGenerateEnhancement/src/results/multiGold_eval_on_real/with_real_few_shot_accumulate_votingCLASS_8_0.0_GoldChange_top/gold_100_1_0.0_IIDgold/bert-base-uncased/sentence-t5-base/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD1_FuseDataset1/0_1_init200_steps4_unbalance_temp3/fewshotK8_15_0.5/imdb/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__chatglm3-6b-base_1000__flan-t5-xl_1000/0/iter0_self_confidence_variability.pth"
votes_within_and_others = "/shared/project/PrivateGenerateEnhancement/src/results/multiGold_eval_on_real/with_real_few_shot_accumulate_votingCLASS_8_0.0_GoldChange_top/gold_100_1_0.0_IIDgold/bert-base-uncased/sentence-t5-base/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD1_FuseDataset1/0_1_init200_steps4_unbalance_temp3/fewshotK8_15_0.5/imdb/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__chatglm3-6b-base_1000__flan-t5-xl_1000/0/real_vote_for_syn_tensor([   0,  180,  360,  540,  720,  900, 1080], device='cuda:0').pth"
pred_change_file = "/shared/project/PrivateGenerateEnhancement/src/results/multiGold_eval_on_real/with_real_few_shot_accumulate_votingCLASS_8_0.0_GoldChange_top/gold_100_1_0.0_IIDgold/bert-base-uncased/sentence-t5-base/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD1_FuseDataset1/0_1_init200_steps4_unbalance_temp3/fewshotK8_15_0.5/imdb/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__chatglm3-6b-base_1000__flan-t5-xl_1000/0/iter0_sample_pred_change.pth"
pred_file = "/shared/project/PrivateGenerateEnhancement/src/results/multiGold_eval_on_real/with_real_few_shot_accumulate_votingCLASS_8_0.0_GoldChange_top/gold_100_1_0.0_IIDgold/bert-base-uncased/sentence-t5-base/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD1_FuseDataset1/0_1_init200_steps4_unbalance_temp3/fewshotK8_15_0.5/imdb/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__chatglm3-6b-base_1000__flan-t5-xl_1000/0/iter0_sample_pred.pth"


gold_distribution = re.search(r'_(.*?)Dgold/', confidence_and_variability_file).group(1).split('_')[-1]
gold_distribution += 'D'


_confidence, _variability = torch.load(confidence_and_variability_file)
confidence, variability = [], []
for _c in _confidence:
    confidence += _c
for _v in _variability:
    variability += _v
confidence = np.asarray(confidence)
variability = np.asarray(variability)
print(confidence.shape, variability.shape)

# [[votes for label0], [votes for label1], ...], [[votes for label other than 0], [votes for label other than 1], ...]
nearest_sample_voting, model_voting_score, local_accumulate_samples = torch.load(votes_within_and_others)
print(type(nearest_sample_voting), type(model_voting_score), type(local_accumulate_samples))
print(nearest_sample_voting.shape, model_voting_score.shape, local_accumulate_samples.shape)
print(nearest_sample_voting, model_voting_score, local_accumulate_samples)

# [confidence for all samples], [variability for all samples]
text, label, loss_per_sample_change, error_per_sample_change, correctness_per_sample_change, prediction_per_sample_change, norm_logits_per_sample_change = torch.load(pred_change_file)
text, label, loss_per_sample_change, error_per_sample_change, correctness_per_sample_change, prediction_per_sample_change, norm_logits_per_sample_change = text, torch.tensor(label[0]).cpu(), torch.cat(loss_per_sample_change,axis=0).cpu(), torch.cat(error_per_sample_change,axis=0).cpu(), torch.cat(correctness_per_sample_change,axis=0).cpu(), torch.cat(prediction_per_sample_change,axis=0).cpu(), torch.cat(norm_logits_per_sample_change,axis=0).cpu()
print(type(text))
print(type(label), label.shape)
print(type(loss_per_sample_change), loss_per_sample_change.shape)
print(type(error_per_sample_change), error_per_sample_change.shape)
print(type(correctness_per_sample_change), correctness_per_sample_change.shape)
print(type(prediction_per_sample_change), prediction_per_sample_change.shape)
print(type(norm_logits_per_sample_change), norm_logits_per_sample_change.shape)

# text, label, loss_per_sample, error_per_sample, correctness_per_sample, prediction_per_sample, norm_logits_per_sample = torch.load(pred_file)
# text, label, loss_per_sample, error_per_sample, correctness_per_sample, prediction_per_sample, norm_logits_per_sample = text, torch.tensor(label[0]).cpu(), torch.cat(loss_per_sample,axis=1).cpu(), torch.cat(error_per_sample,axis=1).cpu(), torch.cat(correctness_per_sample,axis=1).cpu(), torch.cat(prediction_per_sample,axis=1).cpu(), torch.cat(norm_logits_per_sample,axis=1).cpu()
# print(type(text))
# print(type(label), label.shape)
# print(type(loss_per_sample), loss_per_sample.shape)
# print(type(error_per_sample), error_per_sample.shape)
# print(type(correctness_per_sample), correctness_per_sample.shape)
# print(type(prediction_per_sample), prediction_per_sample.shape)
# print(type(norm_logits_per_sample), norm_logits_per_sample.shape)

num_classes = len(list(set(list(label.numpy()))))
print(f"{num_classes=}")

num_models = len(local_accumulate_samples)-1
print(f"{num_models=}")

# with jsonlines.open("./analysis/votes_and_pred_change/high_scoring_sample.jsonl", 'w') as writer:
#     for i_class in range(num_classes):
#         sorted_indices = np.argsort(nearest_sample_voting[i_class])[::-1]
#         for i in range(20):
#             writer.write({
#                 "confidence": float(confidence[sorted_indices[i]]), "variability": float(variability[sorted_indices[i]]), "voting": float(nearest_sample_voting[i_class][sorted_indices[i]]),
#                 "type": "within-class", "real_sample_class": i_class,
#                 "text": text[sorted_indices[i]], "label": int(label[sorted_indices[i]])
#             })
#         writer.write({})
#     for i_class in range(num_classes):
#         sorted_indices = np.argsort(nearest_other_class_sample_voting_per_party[i_class])[::-1]
#         for i in range(20):
#             writer.write({
#                 "confidence": float(confidence[sorted_indices[i]]), "variability": float(variability[sorted_indices[i]]), "voting": float(nearest_other_class_sample_voting_per_party[i_class][sorted_indices[i]]),
#                 "type": "other-class", "real_sample_class": i_class,
#                 "text": text[sorted_indices[i]], "label": int(label[sorted_indices[i]])
#             })
#         writer.write({})


# fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(10.5,6), sharex=False, sharey=False)
# for _i, (value, value_name) in enumerate(zip([loss_per_sample_change, error_per_sample_change, correctness_per_sample_change, prediction_per_sample_change, norm_logits_per_sample_change], ["loss", "error", "correctness", "prediction", "logits norm"])):
#     axs[_i].scatter(nearest_sample_voting, value, alpha=0.5)
#     axs[_i].set_xlabel(f"sample votes")
#     axs[_i].set_ylabel(f"{value_name} change")

# plt.tight_layout()
# plt.savefig(f'./analysis/votes_and_variability_and_pred_change/change_after_gold.png',dpi=200)

# fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(10.5,6), sharex=False, sharey=False)
# for _i, (value, value_name) in enumerate(zip([loss_per_sample_change, error_per_sample_change, correctness_per_sample_change, prediction_per_sample_change, norm_logits_per_sample_change], ["loss", "error", "correctness", "prediction", "logits norm"])):
#     axs[_i].scatter(nearest_sample_voting[nearest_sample_voting!=0.0], value[nearest_sample_voting!=0.0], alpha=0.5)
#     axs[_i].set_xlabel(f"sample votes")
#     axs[_i].set_ylabel(f"{value_name} change")

# plt.tight_layout()
# plt.savefig(f'./analysis/votes_and_variability_and_pred_change/change_after_gold_only_voted.png',dpi=200)

PLM_name = ['GPT-2', 'Llama-2', 'Vicuna', 'OPT', 'ChatGLM3', 'Flan-T5', 'GPT-3.5', 'GPT-4']

fig, axs = plt.subplots(nrows=2, ncols=num_models+1, figsize=(18,6), sharex=False, sharey=False)
for i_model in range(num_models):
    sample_start = int(local_accumulate_samples[i_model])
    sample_end = int(local_accumulate_samples[i_model+1])
    value_x = variability[sample_start:sample_end]
    value_y = confidence[sample_start:sample_end]
    for _i, (value, value_name) in enumerate(zip([loss_per_sample_change[sample_start:sample_end], error_per_sample_change[sample_start:sample_end]], ["loss", "error"])):
        axs[_i][i_model].scatter(value_x, value_y, c=value, cmap='cool', alpha=0.4, s=10)
        axs[_i][i_model].set_xlabel(f"variability")
        axs[_i][i_model].set_ylabel(f"confidence")
        axs[_i][i_model].set_title(f"{value_name} change for {PLM_name[i_model]}")
for _i, (value, value_name) in enumerate(zip([loss_per_sample_change, error_per_sample_change], ["loss", "error"])):
    axs[_i][num_models].scatter(variability, confidence, c=value, cmap='cool', alpha=0.4, s=10)
    axs[_i][num_models].set_xlabel(f"variability")
    axs[_i][num_models].set_ylabel(f"confidence")
    axs[_i][num_models].set_title(f"{value_name} change for all")
plt.tight_layout()
plt.savefig(f'./analysis/votes_and_variability_and_pred_change/{gold_distribution}_variability.png',dpi=200)


fig, axs = plt.subplots(nrows=2, ncols=num_models+1, figsize=(18,6), sharex=False, sharey=False)
for i_model in range(num_models):
    sample_start = int(local_accumulate_samples[i_model])
    sample_end = int(local_accumulate_samples[i_model+1])
    # value_x = variability[sample_start:sample_end]
    # value_y = confidence[sample_start:sample_end]
    for _i, (value, value_name) in enumerate(zip([loss_per_sample_change[sample_start:sample_end], error_per_sample_change[sample_start:sample_end]], ["loss", "error"])):
        axs[_i][i_model].scatter(variability[sample_start:sample_end], value, c=confidence[sample_start:sample_end], cmap='cool', alpha=0.4, s=10)
        axs[_i][i_model].set_xlabel(f"variability")
        axs[_i][i_model].set_ylabel(f"change")
        axs[_i][i_model].set_title(f"{value_name} change for {PLM_name[i_model]}")
for _i, (value, value_name) in enumerate(zip([loss_per_sample_change, error_per_sample_change], ["loss", "error"])):
    axs[_i][num_models].scatter(variability, value, c=confidence, cmap='cool', alpha=0.4, s=10)
    axs[_i][num_models].set_xlabel(f"variability")
    axs[_i][num_models].set_ylabel(f"change")
    axs[_i][num_models].set_title(f"{value_name} change for all")
plt.tight_layout()
plt.savefig(f'./analysis/votes_and_variability_and_pred_change/{gold_distribution}_variability_huefor_confidence.png',dpi=200)


fig, axs = plt.subplots(nrows=2, ncols=num_models+1, figsize=(18,6), sharex=False, sharey=False)
for i_model in range(num_models):
    sample_start = int(local_accumulate_samples[i_model])
    sample_end = int(local_accumulate_samples[i_model+1])
    # value_x = variability[sample_start:sample_end]
    # value_y = confidence[sample_start:sample_end]
    for _i, (value, value_name) in enumerate(zip([loss_per_sample_change[sample_start:sample_end], error_per_sample_change[sample_start:sample_end]], ["loss", "error"])):
        axs[_i][i_model].scatter(confidence[sample_start:sample_end], value, c=variability[sample_start:sample_end], cmap='cool', alpha=0.4, s=10)
        axs[_i][i_model].set_xlabel(f"confidence")
        axs[_i][i_model].set_ylabel(f"change")
        axs[_i][i_model].set_title(f"{value_name} change for {PLM_name[i_model]}")
for _i, (value, value_name) in enumerate(zip([loss_per_sample_change, error_per_sample_change], ["loss", "error"])):
    axs[_i][num_models].scatter(confidence, value, c=variability, cmap='cool', alpha=0.4, s=10)
    axs[_i][num_models].set_xlabel(f"confidence")
    axs[_i][num_models].set_ylabel(f"change")
    axs[_i][num_models].set_title(f"{value_name} change for all")
plt.tight_layout()
plt.savefig(f'./analysis/votes_and_variability_and_pred_change/{gold_distribution}_variability_huefor_variability.png',dpi=200)


for i_model in range(num_models):
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(7,6), sharex=False, sharey=False)
    sample_start = int(local_accumulate_samples[i_model])
    sample_end = int(local_accumulate_samples[i_model+1])
    value_x = variability[sample_start:sample_end]
    value_y = confidence[sample_start:sample_end]
    print(len(value_x), len(value_y))
    print(value_x)
    print(value_y)
    for _i, (value, value_name) in enumerate(zip([loss_per_sample_change[sample_start:sample_end],error_per_sample_change[sample_start:sample_end]], ["loss", "error"])):
        # print(len(value))
        # print(value)
        axs[_i][0].scatter(value_x, value_y, c=value, cmap='cool', alpha=0.4, s=10)
        axs[_i][0].set_xlabel(f"variability")
        axs[_i][0].set_ylabel(f"confidence")
        axs[_i][0].set_title(f"{value_name} change for {PLM_name[i_model]}")

        axs[_i][1].hist(value, bins=50)
        axs[_i][0].set_xlabel(f"{value_name} change")
        axs[_i][0].set_ylabel(f"histogram")

    plt.tight_layout()
    plt.savefig(f'./analysis/votes_and_variability_and_pred_change/{gold_distribution}_variability_{PLM_name[i_model]}.png',dpi=200)
