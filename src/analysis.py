import torch
import numpy as np
import sys, os
import matplotlib.pyplot as plt
import jsonlines

# text_per_sample, label_per_sample, loss_per_sample_change, error_per_sample_change, correctness_per_sample_change, prediction_per_sample_change, norm_logits_per_sample_change = torch.load('/home/DAIR/zouty/ModelFederation/PrivateGenerateEnhancement/src/results/eval_on_real/few_shot_accumulate_influenceCartography/bert-base-uncased/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD1_FuseDataset1/0_1_init200_steps4/fewshotK8_5_0.5/imdb/gpt2-xl_1000/12345/sample_pred_change.pth')

# print(f"{text_per_sample=}\n{label_per_sample=}\n{loss_per_sample_change=}\n{error_per_sample_change=}\n{correctness_per_sample_change=}\n{prediction_per_sample_change=}\n{norm_logits_per_sample_change=}")

# with jsonlines.open('analysis_temp.jsonl', 'w') as writer:
#     for i in range(1):
#         for loss, error, correctness, prediction, norm_of_logits, text, label in zip(loss_per_sample_change[i], error_per_sample_change[i], correctness_per_sample_change[i], prediction_per_sample_change[i], norm_logits_per_sample_change[i], text_per_sample[i], label_per_sample[i]):
#             print(f"{loss=}")
#             print(f"{error=}")
#             print(f"{correctness=}")
#             print(f"{prediction=}")
#             print(f"{norm_of_logits=}")
#             print(f"{text=}")
#             print(f"{label=}")
#             writer.write({'loss': loss.item(), 'error': error.item(), 'correctness': correctness.item(), 'prediction': correctness.item(), 'norm_of_logits': norm_of_logits.item(), 'text': text, 'label': label.item()})



text_per_sample, label_per_sample, loss_per_sample_change, error_per_sample_change, correctness_per_sample_change, prediction_per_sample_change, norm_logits_per_sample_change = torch.load('/home/DAIR/zouty/ModelFederation/PrivateGenerateEnhancement/src/results/eval_on_real/few_shot_accumulate_influenceCartography/bert-base-uncased/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD1_FuseDataset1/0_1_init200_steps4/fewshotK8_5_0.5/imdb/gpt2-xl_1000/12345/sample_pred.pth')
# load = torch.load('/home/DAIR/zouty/ModelFederation/PrivateGenerateEnhancement/src/results/eval_on_real/few_shot_accumulate_influenceCartography/bert-base-uncased/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD1_FuseDataset1/0_1_init20_steps4/fewshotK8_5_0.5/imdb/gpt2-xl_100/12345/sample_pred.pth')
# print(load)
# for i, item in enumerate(load):
#     print(i, item.shape)
print(f"{len(loss_per_sample_change)=}, {loss_per_sample_change[0].shape=}")
print(f"{len(norm_logits_per_sample_change)=}, {norm_logits_per_sample_change[0].shape=}, {norm_logits_per_sample_change[0].transpose(1,0).shape=}")
with jsonlines.open('analysis_detail_temp.jsonl', 'w') as writer:
    for i in range(1):
        for text, label, loss, error, correctness, prediction, norm_of_logits in zip(text_per_sample[i], label_per_sample[i], loss_per_sample_change[i].transpose(1,0), error_per_sample_change[i].transpose(1,0), correctness_per_sample_change[i].transpose(1,0), prediction_per_sample_change[i].transpose(1,0), norm_logits_per_sample_change[i].transpose(1,0)):
            print(f"{loss=}, {error=}, {correctness=}, {prediction=}, {norm_of_logits=}, {text=}, {label=}")
            # print(f"")
            # writer.write({'loss': list(loss.cpu().numpy()), 'error': list(error.cpu().numpy()), 'correctness': list(correctness.cpu().numpy()), 'prediction': list(correctness.cpu().numpy()), 'text': text, 'label': label.item()})
            writer.write({'loss': str(list(loss.float().cpu().numpy())), 'error': str(list(error.float().cpu().numpy())), 'correctness': str(list(correctness.cpu().numpy())), 'prediction': str(list(correctness.cpu().numpy())), 'norm_of_logits': str(list(norm_of_logits.cpu().numpy())), 'text': text, 'label': label.item()})
            # writer.write({'loss': list(loss.cpu().numpy()), 'error': list(error.cpu().numpy()), 'correctness': list(correctness.cpu().numpy()), 'prediction': list(correctness.cpu().numpy()), 'norm_of_logits': list(norm_of_logits.cpu().numpy()), 'text': text, 'label': label.item()})
            # writer.write({'loss': list(loss.cpu().numpy()), 'error': list(error.cpu().numpy()), 'correctness': list(correctness.cpu().numpy()), 'prediction': list(correctness.cpu().numpy()), 'norm_of_logits': list(norm_of_logits.cpu().numpy()), 'text': text, 'label': label.item()})
            # writer.write({'loss': list(loss.cpu().numpy()), 'error': list(error.cpu().numpy()), 'correctness': list(correctness.cpu().numpy()), 'prediction': list(correctness.cpu().numpy()), 'norm_of_logits': list(norm_of_logits.cpu().numpy()), 'text': text, 'label': label.item()})
            # writer.write({'loss': list(loss.cpu().numpy()), 'error': list(error.cpu().numpy()), 'correctness': list(correctness.cpu().numpy()), 'prediction': list(correctness.cpu().numpy()), 'norm_of_logits': list(norm_of_logits.cpu().numpy()), 'text': text, 'label': label.item()})
            # writer.write({'loss': list(loss.cpu().numpy()), 'error': list(error.cpu().numpy()), 'correctness': list(correctness.cpu().numpy()), 'prediction': list(correctness.cpu().numpy()), 'norm_of_logits': list(norm_of_logits.cpu().numpy()), 'text': text, 'label': label.item()})
