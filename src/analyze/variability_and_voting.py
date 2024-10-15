import torch
import jsonlines

torch_file_path = '/home/DAIR/zouty/ModelFederation/PrivateGenerateEnhancement/src/results/eval_on_real/with_real_few_shot_accumulate_voting_8/bert-base-uncased/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD1_FuseDataset1/0_1_init200_steps4_unbalance_temp3/fewshotK8_5_0.5/imdb/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__chatglm3-6b-base_1000__flan-t5-xl_1000/12345/iter0_variability_and_voting.pth'

confidence, variability, voting = torch.load(torch_file_path)

with jsonlines.open('./analyze/variability_voting/temp.jsonl', 'w') as writer:
    for _conf, _var, _vote in zip(confidence, variability, voting):
        writer.write({"confidence": _conf.item(), "variability": _var.item(), "voting": _vote.item()})

with jsonlines.open('./analyze/variability_voting/temp_voted.jsonl', 'w') as writer:
    for _conf, _var, _vote in zip(confidence, variability, voting):
        if _vote.item() > 0:
            writer.write({"confidence": _conf.item(), "variability": _var.item(), "voting": _vote.item()})    