import torch
import numpy as np
import matplotlib.pyplot as plt
import jsonlines
# from scipy.stats import entropy
import re


# flan-t5-xl, gold=[76,24], iter1, 1080
confidence_and_variability_file = "/home/DAIR/zouty/ModelFederation/PrivateGenerateEnhancement/src/results/multiGold_eval_on_real/with_real_few_shot_accumulate_votingCLASS_promptCLASS_8_0.0_random_sampling/gold_100_1_0.0_OODgold/bert-base-uncased/sentence-t5-base/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD1_FuseDataset1/0_1_init1200_steps4_unbalance_temp1.0/fewshotK8_15_0.5/imdb/flan-t5-xl_6000/0/iter0_self_confidence_variability.pth"
votes_within_and_others = "/home/DAIR/zouty/ModelFederation/PrivateGenerateEnhancement/src/results/multiGold_eval_on_real/with_real_few_shot_accumulate_votingCLASS_promptCLASS_8_0.0_random_sampling/gold_100_1_0.0_OODgold/bert-base-uncased/sentence-t5-base/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD1_FuseDataset1/0_1_init1200_steps4_unbalance_temp1.0/fewshotK8_15_0.5/imdb/flan-t5-xl_6000/0/real_vote_for_syn_tensor([   0, 1080], device='cuda:0').pth"
furthest_file = "/home/DAIR/zouty/ModelFederation/PrivateGenerateEnhancement/src/results/multiGold_eval_on_real/with_real_few_shot_accumulate_votingCLASS_promptCLASS_8_0.0_random_sampling/gold_100_1_0.0_OODgold/bert-base-uncased/sentence-t5-base/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD1_FuseDataset1/0_1_init1200_steps4_unbalance_temp1.0/fewshotK8_15_0.5/imdb/flan-t5-xl_6000/0/real_vote_for_furthest_syn_tensor([   0, 1080], device='cuda:0').pth"
pred_change_file = "/home/DAIR/zouty/ModelFederation/PrivateGenerateEnhancement/src/results/multiGold_eval_on_real/with_real_few_shot_accumulate_votingCLASS_promptCLASS_8_0.0_random_sampling/gold_100_1_0.0_OODgold/bert-base-uncased/sentence-t5-base/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD1_FuseDataset1/0_1_init1200_steps4_unbalance_temp1.0/fewshotK8_15_0.5/imdb/flan-t5-xl_6000/0/iter0_sample_pred_change.pth"
pred_file = "/home/DAIR/zouty/ModelFederation/PrivateGenerateEnhancement/src/results/multiGold_eval_on_real/with_real_few_shot_accumulate_votingCLASS_promptCLASS_8_0.0_random_sampling/gold_100_1_0.0_OODgold/bert-base-uncased/sentence-t5-base/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD1_FuseDataset1/0_1_init1200_steps4_unbalance_temp1.0/fewshotK8_15_0.5/imdb/flan-t5-xl_6000/0/iter0_sample_pred.pth"


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

# [[votes for label0], [votes for label1], ...], [[votes for label other than 0], [votes for label other than 1], ...]
furthest_sample_voting, text, label, local_accumulate_samples = torch.load(furthest_file)
print(type(furthest_sample_voting), type(text), type(label), type(local_accumulate_samples))
# furthest_sample_voting, model_voting_score, local_accumulate_samples = torch.load(furthest_file)
# print(type(furthest_sample_voting), type(model_voting_score), type(local_accumulate_samples))
print(furthest_sample_voting.shape, local_accumulate_samples.shape)
print(furthest_sample_voting, local_accumulate_samples)


# [confidence for all samples], [variability for all samples]
text, label, loss_per_sample_change, error_per_sample_change, correctness_per_sample_change, prediction_per_sample_change, norm_logits_per_sample_change = torch.load(pred_change_file)
text, label, loss_per_sample_change, error_per_sample_change, correctness_per_sample_change, prediction_per_sample_change, norm_logits_per_sample_change = text, torch.tensor(label[0]).cpu(), torch.cat(loss_per_sample_change,axis=0).cpu(), torch.cat(error_per_sample_change,axis=0).cpu(), torch.cat(correctness_per_sample_change,axis=0).cpu(), torch.cat(prediction_per_sample_change,axis=0).cpu(), torch.cat(norm_logits_per_sample_change,axis=0).cpu()
print(type(text), len(text))
print(type(label), label.shape)
print(type(loss_per_sample_change), loss_per_sample_change.shape)
print(type(error_per_sample_change), error_per_sample_change.shape)
print(type(correctness_per_sample_change), correctness_per_sample_change.shape)
print(type(prediction_per_sample_change), prediction_per_sample_change.shape)
print(type(norm_logits_per_sample_change), norm_logits_per_sample_change.shape)



num_classes = len(list(set(list(label.numpy()))))
print(f"{num_classes=}")

num_models = len(local_accumulate_samples)-1
print(f"{num_models=}")


_, futher_sorted_indices = torch.sort(furthest_sample_voting, dim=-1, descending=True, stable=True)
print(f"{futher_sorted_indices[:30]}")
for _indice in futher_sorted_indices[:30]:
    print(f"top-{_indice} furthest, voting={furthest_sample_voting[_indice]}, variability={variability[_indice]}, confidence={confidence[_indice]}, label={label[_indice]}, text={text[0][_indice]}")

_, nearest_sorted_indices = torch.sort(furthest_sample_voting, dim=-1, descending=True, stable=True)
print(f"{nearest_sorted_indices[:30]}")
for _indice in nearest_sorted_indices[:30]:
    print(f"top-{_indice} nearest, voting={furthest_sample_voting[_indice]}, variability={variability[_indice]}, confidence={confidence[_indice]}, label={label[_indice]}, text={text[0][_indice]}")

_, confidence_sorted_indices = torch.sort(torch.tensor(confidence), dim=-1, descending=False, stable=True)
print(f"{confidence_sorted_indices[:30]}")
for _indice in confidence_sorted_indices[:30]:
    print(f"top-{_indice} lowest confidence, confidence={confidence[_indice]}, variability={variability[_indice]}, furthest_voting={furthest_sample_voting[_indice]}, nearest_voting={nearest_sample_voting[_indice]}, label={label[_indice]}, text={text[0][_indice]}")

_, confidence_sorted_indices = torch.sort(torch.tensor(variability), dim=-1, descending=True, stable=True)
print(f"{confidence_sorted_indices[:30]}")
for _indice in confidence_sorted_indices[:30]:
    print(f"top-{_indice} highest variability, variability={variability[_indice]}, confidence={confidence[_indice]}, furthest_voting={furthest_sample_voting[_indice]}, nearest_voting={nearest_sample_voting[_indice]}, label={label[_indice]}, text={text[0][_indice]}")
_, confidence_sorted_indices = torch.sort(torch.tensor(variability), dim=-1, descending=False, stable=True)
print(f"{confidence_sorted_indices[:30]}")
for _indice in confidence_sorted_indices[:30]:
    print(f"top-{_indice} lowest variability, variability={variability[_indice]}, confidence={confidence[_indice]}, furthest_voting={furthest_sample_voting[_indice]}, nearest_voting={nearest_sample_voting[_indice]}, label={label[_indice]}, text={text[0][_indice]}")
