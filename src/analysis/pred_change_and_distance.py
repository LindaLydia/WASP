


import torch
import numpy as np
import matplotlib.pyplot as plt
import jsonlines
# from scipy.stats import entropy

# gpt2, gold=[76,24], iter1, 1080
votes_within_and_others = "/shared/project/PrivateGenerateEnhancement/src/results/multiGold_eval_on_real/with_real_few_shot_accumulate_votingCLASS_8_0.0_random_top/gold_100_1_0.0_OODgold/bert-base-uncased/sentence-t5-base/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD1_FuseDataset1/0_1_init1200_steps4_unbalance_temp3/fewshotK8_15_0.5/imdb/gpt2-xl_6000/0/real_vote_for_syn_tensor([   0, 1080], device='cuda:0').pth"
pred_change_file = "/shared/project/PrivateGenerateEnhancement/src/results/multiGold_eval_on_real/with_real_few_shot_accumulate_votingCLASS_8_0.0_random_top/gold_100_1_0.0_OODgold/bert-base-uncased/sentence-t5-base/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD1_FuseDataset1/0_1_init1200_steps4_unbalance_temp3/fewshotK8_15_0.5/imdb/gpt2-xl_6000/0/sample_pred_change.pth"
pred_file = "/shared/project/PrivateGenerateEnhancement/src/results/multiGold_eval_on_real/with_real_few_shot_accumulate_votingCLASS_8_0.0_random_top/gold_100_1_0.0_OODgold/bert-base-uncased/sentence-t5-base/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD1_FuseDataset1/0_1_init1200_steps4_unbalance_temp3/fewshotK8_15_0.5/imdb/gpt2-xl_6000/0/sample_pred.pth"

# opt, gold=[76,24], iter?, 1080
votes_within_and_others = "/shared/project/PrivateGenerateEnhancement/src/results/multiGold_eval_on_real/with_real_few_shot_accumulate_votingCLASS_8_0.0_random_top/gold_100_1_0.0_OODgold/bert-base-uncased/sentence-t5-base/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD1_FuseDataset1/0_1_init1200_steps4_unbalance_temp3/fewshotK8_15_0.5/imdb/opt-6.7b_6000/0/real_vote_for_syn_tensor([   0, 2160], device='cuda:0').pth"
pred_change_file = "/shared/project/PrivateGenerateEnhancement/src/results/multiGold_eval_on_real/with_real_few_shot_accumulate_votingCLASS_8_0.0_random_top/gold_100_1_0.0_OODgold/bert-base-uncased/sentence-t5-base/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD1_FuseDataset1/0_1_init1200_steps4_unbalance_temp3/fewshotK8_15_0.5/imdb/opt-6.7b_6000/0/sample_pred_change.pth"
pred_file = "/shared/project/PrivateGenerateEnhancement/src/results/multiGold_eval_on_real/with_real_few_shot_accumulate_votingCLASS_8_0.0_random_top/gold_100_1_0.0_OODgold/bert-base-uncased/sentence-t5-base/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD1_FuseDataset1/0_1_init1200_steps4_unbalance_temp3/fewshotK8_15_0.5/imdb/opt-6.7b_6000/0/sample_pred.pth"

# [[votes for label0], [votes for label1], ...], [[votes for label other than 0], [votes for label other than 1], ...]
nearest_sample_voting, model_voting_score, local_accumulate_samples = torch.load(votes_within_and_others)
print(type(nearest_sample_voting), type(model_voting_score), type(local_accumulate_samples))
print(nearest_sample_voting.shape, model_voting_score.shape, local_accumulate_samples.shape)
print(nearest_sample_voting, model_voting_score, local_accumulate_samples)

# [confidence for all samples], [variability for all samples]
text, label, loss_per_sample_change, error_per_sample_change, correctness_per_sample_change, prediction_per_sample_change, norm_logits_per_sample_change = torch.load(pred_change_file)
text, label, loss_per_sample_change, error_per_sample_change, correctness_per_sample_change, prediction_per_sample_change, norm_logits_per_sample_change = text[0], label[0], loss_per_sample_change[0].cpu(), error_per_sample_change[0].cpu(), correctness_per_sample_change[0].cpu(), prediction_per_sample_change[0].cpu(), norm_logits_per_sample_change[0].cpu()
print(type(text))
print(type(label), label.shape)
print(type(loss_per_sample_change), loss_per_sample_change.shape)
print(type(error_per_sample_change), error_per_sample_change.shape)
print(type(correctness_per_sample_change), correctness_per_sample_change.shape)
print(type(prediction_per_sample_change), prediction_per_sample_change.shape)
print(type(norm_logits_per_sample_change), norm_logits_per_sample_change.shape)

text, label, loss_per_sample, error_per_sample, correctness_per_sample, prediction_per_sample, norm_logits_per_sample = torch.load(pred_file)
text, label, loss_per_sample, error_per_sample, correctness_per_sample, prediction_per_sample, norm_logits_per_sample = text[0], label[0], loss_per_sample[0].cpu(), error_per_sample[0].cpu(), correctness_per_sample[0].cpu(), prediction_per_sample[0].cpu(), norm_logits_per_sample[0].cpu()
print(type(text))
print(type(label), label.shape)
print(type(loss_per_sample), loss_per_sample.shape)
print(type(error_per_sample), error_per_sample.shape)
print(type(correctness_per_sample), correctness_per_sample.shape)
print(type(prediction_per_sample), prediction_per_sample.shape)
print(type(norm_logits_per_sample), norm_logits_per_sample.shape)

num_classes = len(list(set(list(label.numpy()))))
print(f"{num_classes=}")


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


fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(10.5,6), sharex=False, sharey=False)
for _i, (value, value_name) in enumerate(zip([loss_per_sample_change, error_per_sample_change, correctness_per_sample_change, prediction_per_sample_change, norm_logits_per_sample_change], ["loss", "error", "correctness", "prediction", "logits norm"])):
    axs[_i].scatter(nearest_sample_voting, value, alpha=0.5)
    axs[_i].set_xlabel(f"sample votes")
    axs[_i].set_ylabel(f"{value_name} change")

plt.tight_layout()
plt.savefig(f'./analysis/votes_and_pred_change/change_after_gold.png',dpi=200)

fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(10.5,6), sharex=False, sharey=False)
for _i, (value, value_name) in enumerate(zip([loss_per_sample_change, error_per_sample_change, correctness_per_sample_change, prediction_per_sample_change, norm_logits_per_sample_change], ["loss", "error", "correctness", "prediction", "logits norm"])):
    axs[_i].scatter(nearest_sample_voting[nearest_sample_voting!=0.0], value[nearest_sample_voting!=0.0], alpha=0.5)
    axs[_i].set_xlabel(f"sample votes")
    axs[_i].set_ylabel(f"{value_name} change")

plt.tight_layout()
plt.savefig(f'./analysis/votes_and_pred_change/change_after_gold_only_voted.png',dpi=200)


fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(18,3), sharex=False, sharey=False)
for _i, (value, value_name) in enumerate(zip([loss_per_sample, error_per_sample, correctness_per_sample, prediction_per_sample, norm_logits_per_sample], ["loss", "error", "correctness", "prediction", "logits norm"])):
    if len(list(value[0].shape)) > 1:
        # value_x = entropy(value[0], axis=1)
        # value_y = entropy(value[1], axis=1)
        value_x = -torch.sum(value[0] * np.log(value[0] + 1e-10), axis=1)
        value_y = -torch.sum(value[1] * np.log(value[1] + 1e-10), axis=1)
    else:
        value_x = value[0]
        value_y = value[1]
    axs[_i].plot(np.linspace(0, max(torch.max(value_x),torch.max(value_y)), 100),np.linspace(0, max(torch.max(value_x),torch.max(value_y)), 100), c="grey")
    axs[_i].scatter(value_x, value_y, c=nearest_sample_voting, cmap='cool', alpha=0.4, s=10)
    axs[_i].set_xlabel(f"before")
    axs[_i].set_ylabel(f"after")
    axs[_i].set_title(f"{value_name}")

plt.tight_layout()
plt.savefig(f'./analysis/votes_and_pred_change/before_and_after_gold.png',dpi=200)


fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(18,3), sharex=False, sharey=False)
for _i, (value, value_name) in enumerate(zip([loss_per_sample, error_per_sample, correctness_per_sample, prediction_per_sample, norm_logits_per_sample], ["loss", "error", "correctness", "prediction", "logits norm"])):
    if len(list(value[0].shape)) > 1:
        # value_x = entropy(value[0], axis=1)
        # value_y = entropy(value[1], axis=1)
        value_x = -torch.sum(value[0] * np.log(value[0] + 1e-10), axis=1)
        value_y = -torch.sum(value[1] * np.log(value[1] + 1e-10), axis=1)
    else:
        value_x = value[0]
        value_y = value[1]
    axs[_i].plot(np.linspace(0, max(torch.max(value_x),torch.max(value_y)), 100),np.linspace(0, max(torch.max(value_x),torch.max(value_y)), 100), c="grey")
    axs[_i].scatter(value_x[nearest_sample_voting!=0.0], value_y[nearest_sample_voting!=0.0], c=nearest_sample_voting[nearest_sample_voting!=0.0], cmap='cool', alpha=0.4, s=10)
    axs[_i].set_xlabel(f"before")
    axs[_i].set_ylabel(f"after")
    axs[_i].set_title(f"{value_name}")

plt.tight_layout()
plt.savefig(f'./analysis/votes_and_pred_change/before_and_after_gold_only_voted.png',dpi=200)


fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(18,3), sharex=False, sharey=False)
for _i, (value, value_change, value_name) in enumerate(zip([loss_per_sample, error_per_sample, correctness_per_sample, prediction_per_sample, norm_logits_per_sample], [loss_per_sample_change, error_per_sample_change, correctness_per_sample_change, prediction_per_sample_change, norm_logits_per_sample_change], ["loss", "error", "correctness", "prediction", "logits norm"])):
    if len(list(value[0].shape)) > 1:
        value_x = -torch.sum(value[0] * np.log(value[0] + 1e-10), axis=1)
        # value_y = -torch.sum(value[1] * np.log(value[1] + 1e-10), axis=1)
    else:
        value_x = value[0]
        # value_y = value[1]
    value_y = value_change
    # axs[_i].plot(np.linspace(0, max(torch.max(value_x),torch.max(value_y)), 100),np.linspace(0, max(torch.max(value_x),torch.max(value_y)), 100), c="grey")
    axs[_i].scatter(value_x[nearest_sample_voting!=0.0], value_y[nearest_sample_voting!=0.0], c=nearest_sample_voting[nearest_sample_voting!=0.0], cmap='cool', alpha=0.4, s=10)
    axs[_i].set_xlabel(f"before")
    axs[_i].set_ylabel(f"change")
    axs[_i].set_title(f"{value_name}")

plt.tight_layout()
plt.savefig(f'./analysis/votes_and_pred_change/before_and_change_gold_only_voted.png',dpi=200)

fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(18,3), sharex=False, sharey=False)
for _i, (value, value_change, value_name) in enumerate(zip([loss_per_sample, error_per_sample, correctness_per_sample, prediction_per_sample, norm_logits_per_sample], [loss_per_sample_change, error_per_sample_change, correctness_per_sample_change, prediction_per_sample_change, norm_logits_per_sample_change], ["loss", "error", "correctness", "prediction", "logits norm"])):
    if len(list(value[0].shape)) > 1:
        value_x = -torch.sum(value[1] * np.log(value[1] + 1e-10), axis=1)
    else:
        value_x = value[1]
    value_y = value_change
    # axs[_i].plot(np.linspace(0, max(torch.max(value_x),torch.max(value_y)), 100),np.linspace(0, max(torch.max(value_x),torch.max(value_y)), 100), c="grey")
    axs[_i].scatter(value_x[nearest_sample_voting!=0.0], value_y[nearest_sample_voting!=0.0], c=nearest_sample_voting[nearest_sample_voting!=0.0], cmap='cool', alpha=0.4, s=10)
    axs[_i].set_xlabel(f"after")
    axs[_i].set_ylabel(f"change")
    axs[_i].set_title(f"{value_name}")

plt.tight_layout()
plt.savefig(f'./analysis/votes_and_pred_change/after_and_change_gold_only_voted.png',dpi=200)
