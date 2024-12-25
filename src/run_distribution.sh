#!/bin/bash
 
# strings=(
#   # "data_accumulate/imdb/single_copied/bert-base-uncased/increasedTheta_KD1_FuseDataset1/fewshotK8_5_0.5/1/"
#   # "data_accumulate/imdb/influenceCartography/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__chatglm3-6b-base_1000__flan-t5-xl_1000/bert-base-uncased/increasedTheta_KD1_FuseDataset1/fewshotK8_5_0.5/12345/"
#   # "data_accumulate/influenceCartography/imdb/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__chatglm3-6b-base_1000__flan-t5-xl_1000/bert-base-uncased/increasedTheta_KD1_FuseDataset1/fewshotK8_5_0.5/12345/"
#   "data_accumulate/voting_8/imdb/gpt2-xl_100__llama-2-7b-chat-hf_100__vicuna-7b-1.5v_100__opt-6.7b_100__chatglm3-6b-base_100__flan-t5-xl_100/bert-base-uncased/increasedTheta_KD1_FuseDataset1/fewshotK8_5_0.5/12345/"
#   # "data_accumulate/influenceCartography/imdb/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__chatglm3-6b-base_1000__flan-t5-xl_1000+copy/bert-base-uncased/increasedTheta_KD1_FuseDataset1/fewshotK8_5_0.5/12345/"
#   # "data_new/"
# )
# for syn_data_path in "${strings[@]}"; do
#     python plotting/plot_embedding_distribution.py --small_model_name sentence-t5-base --syn_data_path $syn_data_path --llms gpt2-xl llama-2-7b-chat-hf vicuna-7b-1.5v opt-6.7b chatglm3-6b-base flan-t5-xl --task_name imdb --num_use_samples_inner 100 100 100 100 100 100 --gpu 5 --small_model_name bert-base-uncased --consider_real True --gold_data_num 10
#     # python plotting/plot_embedding_distribution.py --small_model_name sentence-t5-base --syn_data_path $syn_data_path --llms gpt2-xl llama-2-7b-chat-hf vicuna-7b-1.5v opt-6.7b chatglm3-6b-base flan-t5-xl --task_name imdb --num_use_samples_inner 400 400 400 400 400 400 --gpu 0 --small_model_name bert-base-uncased --consider_real True --gold_data_num 400
#     # python plotting/plot_embedding_distribution.py --small_model_name sentence-t5-base --syn_data_path $syn_data_path --llms gpt2-xl llama-2-7b-chat-hf vicuna-7b-1.5v opt-6.7b chatglm3-6b-base flan-t5-xl --task_name imdb --num_use_samples_inner 200 200 200 200 200 200 --gpu 0 --small_model_name bert-base-uncased --consider_real True --gold_data_num 400
#     # python plotting/plot_embedding_distribution.py --small_model_name sentence-t5-base --syn_data_path $syn_data_path --llms gpt2-xl llama-2-7b-chat-hf vicuna-7b-1.5v opt-6.7b chatglm3-6b-base flan-t5-xl --task_name imdb --num_use_samples_inner 1000 1000 1000 1000 1000 1000 --gpu 2 --small_model_name bert-base-uncased
# done

# # strings=(
# #   "data_accumulate/qnli/single_copied/bert-base-uncased/increasedTheta_KD1_FuseDataset1/fewshotK4_5_0.5/42/"
# #   "data_accumulate/qnli/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__flan-t5-xl_1000__chatglm3-6b-base_1000/bert-base-uncased/increasedTheta_KD1_FuseDataset1/fewshotK4_5_0.5/12345/"
# #   "data_new/"
# # )
# # for syn_data_path in "${strings[@]}"; do
# #     python plotting/plot_embedding_distribution.py --syn_data_path $syn_data_path --llms gpt2-xl llama-2-7b-chat-hf vicuna-7b-1.5v opt-6.7b chatglm3-6b-base flan-t5-xl --task_name qnli --num_use_samples_inner 1000 1000 1000 1000 1000 1000 --gpu 2 --small_model_name bert-base-uncased
# # done



# plm_list=("gpt2-xl" "llama-2-7b-chat-hf" "vicuna-7b-1.5v" "opt-6.7b" "chatglm3-6b-base" "flan-t5-xl" "gpt-3.5-turbo-instruct" "gpt-4-turbo-preview")
 
# # 枚举列表中的每个字符串
# for plm_name in "${plm_list[@]}"; do
#     echo "Processing PLM: $plm_name"
#     # 在这里执行对$string的操作
#     strings=(
#       "/home/DAIR/zouty/ModelFederation/PrivateGenerateEnhancement/src/data_accumulate/votingCLASS_8_random_top/imdb/${plm_name}_6000/bert-base-uncased/sentence-t5-base/increasedTheta_KD1_FuseDataset1/fewshotK8_15_0.5/12345/"
#       # "data_accumulate/influenceCartography/imdb/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__chatglm3-6b-base_1000__flan-t5-xl_1000+copy/bert-base-uncased/increasedTheta_KD1_FuseDataset1/fewshotK8_5_0.5/12345/"
#       # "data_new/"
#     )
#     for syn_data_path in "${strings[@]}"; do
#         python plotting/plot_embedding_distribution.py --small_model_name sentence-t5-base --syn_data_path $syn_data_path --llms $plm_name --task_name imdb --num_use_samples_inner 6000 --gpu 3 --small_model_name bert-base-uncased --consider_real True --gold_data_num 100
#         # python plotting/plot_embedding_distribution.py --small_model_name sentence-t5-base --syn_data_path $syn_data_path --llms gpt2-xl llama-2-7b-chat-hf vicuna-7b-1.5v opt-6.7b chatglm3-6b-base flan-t5-xl --task_name imdb --num_use_samples_inner 400 400 400 400 400 400 --gpu 0 --small_model_name bert-base-uncased --consider_real True --gold_data_num 400
#         # python plotting/plot_embedding_distribution.py --small_model_name sentence-t5-base --syn_data_path $syn_data_path --llms gpt2-xl llama-2-7b-chat-hf vicuna-7b-1.5v opt-6.7b chatglm3-6b-base flan-t5-xl --task_name imdb --num_use_samples_inner 200 200 200 200 200 200 --gpu 0 --small_model_name bert-base-uncased --consider_real True --gold_data_num 400
#         # python plotting/plot_embedding_distribution.py --small_model_name sentence-t5-base --syn_data_path $syn_data_path --llms gpt2-xl llama-2-7b-chat-hf vicuna-7b-1.5v opt-6.7b chatglm3-6b-base flan-t5-xl --task_name imdb --num_use_samples_inner 1000 1000 1000 1000 1000 1000 --gpu 2 --small_model_name bert-base-uncased
#     done
# done


# ########################### plot for imdb for privacy-few-shot gen ###########################
# # plm_list=("opt-6.7b")
# plm_list=("gpt2-xl" "opt-6.7b")
# task_list=("imdb")
# # plm_list=("gpt2-xl" "llama-2-7b-chat-hf" "vicuna-7b-1.5v" "opt-6.7b" "chatglm3-6b-base" "flan-t5-xl" "gpt-3.5-turbo-instruct" "gpt-4-turbo-preview")

# # 枚举列表中的每个字符串
# for task_name in "${task_list[@]}"; do
#     for plm_name in "${plm_list[@]}"; do
#         echo "Processing PLM: $plm_name, for task $task_name"
#         # 在这里执行对$string的操作
#         strings=(
#           # "/shared/project/PrivateGenerateEnhancement/src/data_accumulate/votingCLASS_promptALL_8_0.0_random_sampling/gold_10000_1_0.0_OODgold/imdb/${plm_name}_6000/bert-base-uncased/sentence-t5-base/increasedTheta_KD1_FuseDataset1/fewshotK8_15_0.5/0/"
#           "./data_accumulate/votingCLASS_promptCLASS_8_0.0_random_sampling/gold_10000_1_0.0_OODgold/imdb/${plm_name}_6000/bert-base-uncased/sentence-t5-base/increasedTheta_KD1_FuseDataset1/fewshotK8_15_0.5/0/"
#           # "data_new/"
#         )
#         for syn_data_path in "${strings[@]}"; do
#             python plotting/plot_embedding_distribution.py --small_model_name sentence-t5-base --syn_data_path $syn_data_path --llms $plm_name --task_name $task_name --num_use_samples_inner 6000 --gpu 0 --small_model_name bert-base-uncased --consider_real True --gold_data_num 10000
#             # python plotting/plot_embedding_distribution.py --small_model_name sentence-t5-base --syn_data_path $syn_data_path --llms gpt2-xl llama-2-7b-chat-hf vicuna-7b-1.5v opt-6.7b chatglm3-6b-base flan-t5-xl --task_name imdb --num_use_samples_inner 400 400 400 400 400 400 --gpu 0 --small_model_name bert-base-uncased --consider_real True --gold_data_num 400
#             # python plotting/plot_embedding_distribution.py --small_model_name sentence-t5-base --syn_data_path $syn_data_path --llms gpt2-xl llama-2-7b-chat-hf vicuna-7b-1.5v opt-6.7b chatglm3-6b-base flan-t5-xl --task_name imdb --num_use_samples_inner 200 200 200 200 200 200 --gpu 0 --small_model_name bert-base-uncased --consider_real True --gold_data_num 400
#             # python plotting/plot_embedding_distribution.py --small_model_name sentence-t5-base --syn_data_path $syn_data_path --llms gpt2-xl llama-2-7b-chat-hf vicuna-7b-1.5v opt-6.7b chatglm3-6b-base flan-t5-xl --task_name imdb --num_use_samples_inner 1000 1000 1000 1000 1000 1000 --gpu 2 --small_model_name bert-base-uncased
#         done
#     done
# done
# ########################### plot for imdb for privacy-few-shot gen ###########################


# # plm_list=("gpt2-xl" "llama-2-7b-chat-hf" "vicuna-7b-1.5v" "opt-6.7b" "chatglm3-6b-base" "flan-t5-xl" "gpt-3.5-turbo-instruct" "gpt-4-turbo-preview")
# plm_list=("gpt2-xl")
# task_list=("openreviewCategory" "openreviewRating" "yelpCategory" "yelpRating")
# plm_list=("opt-6.7b")
# task_list=("imdb")
# # plm_list=("gpt2-xl")
# # task_list=("openreviewCategory")
# # plm_list=("gpt2-xl")
# # task_list=("openreviewCategory" "openreviewRating" "yelpCategory" "yelpRating")
# plm_list=("flan-t5-xl")
# task_list=("imdb")


# # 枚举列表中的每个字符串
# for task_name in "${task_list[@]}"; do
#     for plm_name in "${plm_list[@]}"; do
#         echo "Processing PLM: $plm_name, for task $task_name"
#         # 在这里执行对$string的操作
#         strings=(
#           # # "/shared/project/PrivateGenerateEnhancement/src/data_accumulate/votingCLASS_promptALL_8_0.0_random_sampling/gold_10000_1_0.0_OODgold/imdb/${plm_name}_6000/bert-base-uncased/sentence-t5-base/increasedTheta_KD1_FuseDataset1/fewshotK8_15_0.5/0/"
#           # "./data_accumulate/votingCLASS_promptCLASS_8_0.0_random_sampling/gold_100_1_0.0_OODgold/${task_name}/${plm_name}_6000/bert-base-uncased/sentence-t5-base/increasedTheta_KD1_FuseDataset1/fewshotK8_15_0.5/0/"      # "data_accumulate/influenceCartography/imdb/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__chatglm3-6b-base_1000__flan-t5-xl_1000+copy/bert-base-uncased/increasedTheta_KD1_FuseDataset1/fewshotK8_5_0.5/12345/"
#           "./data_accumulate/votingCLASS_promptCLASS_8_0.0_random_sampling/gold_100_1_0.0_OODgold/${task_name}/${plm_name}_6000/bert-base-uncased/sentence-t5-base/increasedTheta_KD1_FuseDataset1/fewshotK8_15_0.5/0/"
#         #   "./data_accumulate/votingCLASS_promptCLASS_8_0.0_random_sampling/gold_100_1_0.0_OODgold/openreviewCategory/gpt2-xl_1000/bert-base-uncased/sentence-t5-base/increasedTheta_KD1_FuseDataset1/fewshotK8_15_0.5/0/"
#           # # "data_new/"
#         )
#         for syn_data_path in "${strings[@]}"; do
#             # python plotting/plot_embedding_distribution.py --small_model_name sentence-t5-base --syn_data_path $syn_data_path --llms $plm_name --task_name $task_name --num_use_samples_inner 2400 --steps 1 --gpu 0 --small_model_name bert-base-uncased --consider_real True --gold_data_num 100
#             # python plotting/plot_embedding_distribution.py --small_model_name sentence-t5-base --syn_data_path $syn_data_path --llms $plm_name --task_name $task_name --num_use_samples_inner 6000 --steps 4 --gpu 0 --small_model_name bert-base-uncased --consider_real True --gold_data_num 100
#             # python plotting/plot_embedding_distribution.py --small_model_name sentence-t5-base --syn_data_path $syn_data_path --llms $plm_name --task_name $task_name --num_use_samples_inner 1000 --steps 4 --gpu 0 --small_model_name bert-base-uncased --consider_real True --gold_data_num 100
#             python plotting/plot_embedding_distribution.py --small_model_name sentence-t5-base --syn_data_path $syn_data_path --llms $plm_name --task_name $task_name --num_use_samples_inner 3600 --steps 2 --gpu 1 --small_model_name bert-base-uncased --consider_real True --gold_data_num 100
#             # # python plotting/plot_embedding_distribution.py --small_model_name sentence-t5-base --syn_data_path $syn_data_path --llms gpt2-xl llama-2-7b-chat-hf vicuna-7b-1.5v opt-6.7b chatglm3-6b-base flan-t5-xl --task_name imdb --num_use_samples_inner 400 400 400 400 400 400 --gpu 0 --small_model_name bert-base-uncased --consider_real True --gold_data_num 400
#             # # python plotting/plot_embedding_distribution.py --small_model_name sentence-t5-base --syn_data_path $syn_data_path --llms gpt2-xl llama-2-7b-chat-hf vicuna-7b-1.5v opt-6.7b chatglm3-6b-base flan-t5-xl --task_name imdb --num_use_samples_inner 200 200 200 200 200 200 --gpu 0 --small_model_name bert-base-uncased --consider_real True --gold_data_num 400
#             # # python plotting/plot_embedding_distribution.py --small_model_name sentence-t5-base --syn_data_path $syn_data_path --llms gpt2-xl llama-2-7b-chat-hf vicuna-7b-1.5v opt-6.7b chatglm3-6b-base flan-t5-xl --task_name imdb --num_use_samples_inner 1000 1000 1000 1000 1000 1000 --gpu 2 --small_model_name bert-base-uncased
#         done
#     done
# done

task_list=("imdb")
# 枚举列表中的每个字符串
for task_name in "${task_list[@]}"; do
    echo "Processing PLM: 6PLM, for task $task_name"
    # 在这里执行对$string的操作
    strings=(
      # "./data_accumulate/votingCLASS_promptCLASS_8_0.0_randomCartographyOriginalContrast_top/gold_100_1_0.0_OODgold/${task_name}/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__chatglm3-6b-base_1000__flan-t5-xl_1000/bert-base-uncased/sentence-t5-base/increasedTheta_KD1_FuseDataset1/fewshotK8_15_0.5/12345/"
      # "./data_accumulate/votingCLASS_promptCLASS_8_0.0_randomContrast_top/gold_100_1_0.0_OODgold/${task_name}/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__chatglm3-6b-base_1000__flan-t5-xl_1000/bert-base-uncased/sentence-t5-base/increasedTheta_KD1_FuseDataset1/fewshotK8_15_0.5/0/"
      # "./data_accumulate/votingCLASS_promptCLASS_8_0.0_randomCartographyOriginal_top/gold_100_1_0.0_OODgold/${task_name}/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__chatglm3-6b-base_1000__flan-t5-xl_1000/bert-base-uncased/sentence-t5-base/increasedTheta_KD1_FuseDataset1/fewshotK8_15_0.5/12345/"
      # "./data_accumulate/votingCLASS_promptCLASS_8_0.0_random_top/gold_100_1_0.0_OODgold/${task_name}/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__chatglm3-6b-base_1000__flan-t5-xl_1000/bert-base-uncased/sentence-t5-base/increasedTheta_KD1_FuseDataset1/fewshotK8_15_0.5/12345/"
      
      "./data_accumulate/votingCLASS_promptCLASS_8_0.0_randomContrast_top/gold_100_1_0.0_OODgold/${task_name}/gpt2-xl_6000/bert-base-uncased/sentence-t5-base/increasedTheta_KD1_FuseDataset1/fewshotK8_15_0.5/0/"
      "./data_accumulate/votingCLASS_promptCLASS_8_0.0_random_top/gold_100_1_0.0_OODgold/${task_name}/gpt2-xl_6000/bert-base-uncased/sentence-t5-base/increasedTheta_KD1_FuseDataset1/fewshotK8_15_0.5/0/"
      # "./data_accumulate/votingCLASS_promptCLASS_8_0.0_randomContrast_top/gold_100_1_0.0_OODgold/${task_name}/llama-2-7b-chat-hf_6000/bert-base-uncased/sentence-t5-base/increasedTheta_KD1_FuseDataset1/fewshotK8_15_0.5/0/"
      # "./data_accumulate/votingCLASS_promptCLASS_8_0.0_random_top/gold_100_1_0.0_OODgold/${task_name}/llama-2-7b-chat-hf_6000/bert-base-uncased/sentence-t5-base/increasedTheta_KD1_FuseDataset1/fewshotK8_15_0.5/0/"
      # "./data_accumulate/votingCLASS_promptCLASS_8_0.0_randomContrast_top/gold_100_1_0.0_OODgold/${task_name}/opt-6.7b_6000/bert-base-uncased/sentence-t5-base/increasedTheta_KD1_FuseDataset1/fewshotK8_15_0.5/0/"
      # "./data_accumulate/votingCLASS_promptCLASS_8_0.0_random_top/gold_100_1_0.0_OODgold/${task_name}/opt-6.7b_6000/bert-base-uncased/sentence-t5-base/increasedTheta_KD1_FuseDataset1/fewshotK8_15_0.5/0/"
      # "./data_accumulate/votingCLASS_promptCLASS_8_0.0_randomContrast_top/gold_100_1_0.0_OODgold/${task_name}/vicuna-7b-1.5v_6000/bert-base-uncased/sentence-t5-base/increasedTheta_KD1_FuseDataset1/fewshotK8_15_0.5/0/"
      # "./data_accumulate/votingCLASS_promptCLASS_8_0.0_random_top/gold_100_1_0.0_OODgold/${task_name}/vicuna-7b-1.5v_6000/bert-base-uncased/sentence-t5-base/increasedTheta_KD1_FuseDataset1/fewshotK8_15_0.5/0/"
      
      # "./data_accumulate/votingCLASS_promptCLASS_8_0.0_randomContrast_top/gold_1000_1_0.0_OODgold/${task_name}/gpt2-xl_6000/bert-base-uncased/sentence-t5-base/increasedTheta_KD1_FuseDataset1/fewshotK8_15_0.5/0/"
      # "./data_accumulate/votingCLASS_promptCLASS_8_0.0_random_top/gold_1000_1_0.0_OODgold/${task_name}/gpt2-xl_6000/bert-base-uncased/sentence-t5-base/increasedTheta_KD1_FuseDataset1/fewshotK8_15_0.5/0/"
    )
    logging_strings=(
      # "./logging/multiGold_eval_on_real/with_real_few_shot_accumulate_votingCLASS_promptCLASS_8_0.0_randomCartographyOriginalContrast_top/gold_100_1_0.0_OODgold/bert-base-uncased/sentence-t5-base/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD1_FuseDataset1/0_1_init200_steps4_unbalance_temp1.0/fewshotK8_15_0.5/imdb/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__chatglm3-6b-base_1000__flan-t5-xl_1000/log_data_accumulate_4.408560690471575_12345.txt"
      # "./logging/multiGold_eval_on_real/with_real_few_shot_accumulate_votingCLASS_promptCLASS_8_0.0_randomContrast_top/gold_100_1_0.0_OODgold/bert-base-uncased/sentence-t5-base/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD1_FuseDataset1/0_1_init200_steps4_unbalance_temp1.0/fewshotK8_15_0.5/imdb/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__chatglm3-6b-base_1000__flan-t5-xl_1000/log_data_accumulate_4.408560690471575_0.txt"
      # "./logging/multiGold_eval_on_real/with_real_few_shot_accumulate_votingCLASS_promptCLASS_8_0.0_randomCartographyOriginal_top/gold_100_1_0.0_OODgold/bert-base-uncased/sentence-t5-base/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD1_FuseDataset1/0_1_init200_steps4_unbalance_temp1.0/fewshotK8_15_0.5/imdb/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__chatglm3-6b-base_1000__flan-t5-xl_1000/log_data_accumulate_4.408560690471575_12345.txt"
      # "./logging/multiGold_eval_on_real/with_real_few_shot_accumulate_votingCLASS_promptCLASS_8_0.0_random_top/gold_100_1_0.0_OODgold/bert-base-uncased/sentence-t5-base/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD1_FuseDataset1/0_1_init200_steps4_unbalance_temp1.0/fewshotK8_15_0.5/imdb/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__chatglm3-6b-base_1000__flan-t5-xl_1000/log_data_accumulate_4.408560690471575_12345.txt"
      
      "./logging/multiGold_eval_on_real/with_real_few_shot_accumulate_votingCLASS_promptCLASS_8_0.0_randomContrast_top/gold_100_1_0.0_OODgold/bert-base-uncased/sentence-t5-base/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD1_FuseDataset1/0_1_init1200_steps4_unbalance_temp1.0/fewshotK8_15_0.5/imdb/gpt2-xl_6000/log_data_accumulate_4.89894920704081_0.txt"
      "./logging/multiGold_eval_on_real/with_real_few_shot_accumulate_votingCLASS_promptCLASS_8_0.0_random_top/gold_100_1_0.0_OODgold/bert-base-uncased/sentence-t5-base/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD1_FuseDataset1/0_1_init1200_steps4_unbalance_temp1.0/fewshotK8_15_0.5/imdb/gpt2-xl_6000/log_data_accumulate_4.89894920704081_0.txt"
      # "./logging/multiGold_eval_on_real/with_real_few_shot_accumulate_votingCLASS_promptCLASS_8_0.0_randomContrast_top/gold_100_1_0.0_OODgold/bert-base-uncased/sentence-t5-base/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD1_FuseDataset1/0_1_init1200_steps4_unbalance_temp1.0/fewshotK8_15_0.5/imdb/llama-2-7b-chat-hf_6000/log_data_accumulate_4.89894920704081_0.txt"
      # "./logging/multiGold_eval_on_real/with_real_few_shot_accumulate_votingCLASS_promptCLASS_8_0.0_random_top/gold_100_1_0.0_OODgold/bert-base-uncased/sentence-t5-base/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD1_FuseDataset1/0_1_init1200_steps4_unbalance_temp1.0/fewshotK8_15_0.5/imdb/llama-2-7b-chat-hf_6000/log_data_accumulate_4.89894920704081_0.txt"
      # "./logging/multiGold_eval_on_real/with_real_few_shot_accumulate_votingCLASS_promptCLASS_8_0.0_randomContrast_top/gold_100_1_0.0_OODgold/bert-base-uncased/sentence-t5-base/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD1_FuseDataset1/0_1_init1200_steps4_unbalance_temp1.0/fewshotK8_15_0.5/imdb/opt-6.7b_6000/log_data_accumulate_4.89894920704081_0.txt"
      # "./logging/multiGold_eval_on_real/with_real_few_shot_accumulate_votingCLASS_promptCLASS_8_0.0_random_top/gold_100_1_0.0_OODgold/bert-base-uncased/sentence-t5-base/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD1_FuseDataset1/0_1_init1200_steps4_unbalance_temp1.0/fewshotK8_15_0.5/imdb/opt-6.7b_6000/log_data_accumulate_4.89894920704081_0.txt"
      # "./logging/multiGold_eval_on_real/with_real_few_shot_accumulate_votingCLASS_promptCLASS_8_0.0_randomContrast_top/gold_100_1_0.0_OODgold/bert-base-uncased/sentence-t5-base/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD1_FuseDataset1/0_1_init1200_steps4_unbalance_temp1.0/fewshotK8_15_0.5/imdb/vicuna-7b-1.5v_6000/log_data_accumulate_4.89894920704081_0.txt"
      # "./logging/multiGold_eval_on_real/with_real_few_shot_accumulate_votingCLASS_promptCLASS_8_0.0_random_top/gold_100_1_0.0_OODgold/bert-base-uncased/sentence-t5-base/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD1_FuseDataset1/0_1_init1200_steps4_unbalance_temp1.0/fewshotK8_15_0.5/imdb/vicuna-7b-1.5v_6000/log_data_accumulate_4.89894920704081_0.txt"
      
      # "./logging/multiGold_eval_on_real/with_real_few_shot_accumulate_votingCLASS_promptCLASS_8_0.0_randomContrast_top/gold_1000_1_0.0_OODgold/bert-base-uncased/sentence-t5-base/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD1_FuseDataset1/0_1_init1200_steps4_unbalance_temp1.0/fewshotK8_15_0.5/imdb/gpt2-xl_6000/log_data_accumulate_4.89894920704081_0.txt"
      # "./logging/multiGold_eval_on_real/with_real_few_shot_accumulate_votingCLASS_promptCLASS_8_0.0_random_top/gold_1000_1_0.0_OODgold/bert-base-uncased/sentence-t5-base/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD1_FuseDataset1/0_1_init1200_steps4_unbalance_temp1.0/fewshotK8_15_0.5/imdb/gpt2-xl_6000/log_data_accumulate_4.89894920704081_0.txt"
    )
    result_strings=(
      # "./results/multiGold_eval_on_real/with_real_few_shot_accumulate_votingCLASS_promptCLASS_8_0.0_random_top/gold_100_1_0.0_OODgold/bert-base-uncased/sentence-t5-base/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD1_FuseDataset1/0_1_init200_steps4_unbalance_temp1.0/fewshotK8_15_0.5/imdb/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__chatglm3-6b-base_1000__flan-t5-xl_1000/12345/" #correctness_prediction_logits_confidence_variability_for_dynamic_chatglm3-6b-base.pth

      "./results/multiGold_eval_on_real/with_real_few_shot_accumulate_votingCLASS_promptCLASS_8_0.0_randomContrast_top/gold_100_1_0.0_OODgold/bert-base-uncased/sentence-t5-base/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD1_FuseDataset1/0_1_init1200_steps4_unbalance_temp1.0/fewshotK8_15_0.5/imdb/gpt2-xl_6000/0/"
      "./results/multiGold_eval_on_real/with_real_few_shot_accumulate_votingCLASS_promptCLASS_8_0.0_random_top/gold_100_1_0.0_OODgold/bert-base-uncased/sentence-t5-base/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD1_FuseDataset1/0_1_init1200_steps4_unbalance_temp1.0/fewshotK8_15_0.5/imdb/gpt2-xl_6000/0/"
      # "./results/multiGold_eval_on_real/with_real_few_shot_accumulate_votingCLASS_promptCLASS_8_0.0_randomContrast_top/gold_100_1_0.0_OODgold/bert-base-uncased/sentence-t5-base/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD1_FuseDataset1/0_1_init1200_steps4_unbalance_temp1.0/fewshotK8_15_0.5/imdb/llama-2-7b-chat-hf_6000/0/"
      # "./results/multiGold_eval_on_real/with_real_few_shot_accumulate_votingCLASS_promptCLASS_8_0.0_random_top/gold_100_1_0.0_OODgold/bert-base-uncased/sentence-t5-base/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD1_FuseDataset1/0_1_init1200_steps4_unbalance_temp1.0/fewshotK8_15_0.5/imdb/llama-2-7b-chat-hf_6000/0/"
      # "./results/multiGold_eval_on_real/with_real_few_shot_accumulate_votingCLASS_promptCLASS_8_0.0_randomContrast_top/gold_100_1_0.0_OODgold/bert-base-uncased/sentence-t5-base/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD1_FuseDataset1/0_1_init1200_steps4_unbalance_temp1.0/fewshotK8_15_0.5/imdb/opt-6.7b_6000/0/"
      # "./results/multiGold_eval_on_real/with_real_few_shot_accumulate_votingCLASS_promptCLASS_8_0.0_random_top/gold_100_1_0.0_OODgold/bert-base-uncased/sentence-t5-base/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD1_FuseDataset1/0_1_init1200_steps4_unbalance_temp1.0/fewshotK8_15_0.5/imdb/opt-6.7b_6000/0/"
      # "./results/multiGold_eval_on_real/with_real_few_shot_accumulate_votingCLASS_promptCLASS_8_0.0_randomContrast_top/gold_100_1_0.0_OODgold/bert-base-uncased/sentence-t5-base/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD1_FuseDataset1/0_1_init1200_steps4_unbalance_temp1.0/fewshotK8_15_0.5/imdb/vicuna-7b-1.5v_6000/0/"
      # "./results/multiGold_eval_on_real/with_real_few_shot_accumulate_votingCLASS_promptCLASS_8_0.0_random_top/gold_100_1_0.0_OODgold/bert-base-uncased/sentence-t5-base/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD1_FuseDataset1/0_1_init1200_steps4_unbalance_temp1.0/fewshotK8_15_0.5/imdb/vicuna-7b-1.5v_6000/0/"
    
    )

    for i in $(seq 0 $((${#strings[@]}-1))); do
        syn_data_path=${strings[$i]}
        logging_path=${logging_strings[$i]}
        results_path=${result_strings[$i]}
        echo "syn_data_path = $syn_data_path"
        echo "logging_path = $logging_path"
        echo "results_path = $results_path"
        # python plotting/plot_embedding_distribution.py --small_model_name sentence-t5-base --syn_data_path $syn_data_path --llms gpt2-xl llama-2-7b-chat-hf vicuna-7b-1.5v opt-6.7b chatglm3-6b-base flan-t5-xl --task_name $task_name --num_use_samples_inner 1000 1000 1000 1000 1000 1000 --logging_path=$logging_path --results_path=$results_path --gpu 1 --steps 4 --small_model_name bert-base-uncased --consider_real True --gold_data_num 100

        python plotting/plot_embedding_distribution.py --small_model_name sentence-t5-base --syn_data_path $syn_data_path --llms gpt2-xl --task_name $task_name --num_use_samples_inner 6000 --logging_path=$logging_path --results_path=$results_path --gpu 0 --steps 4 --small_model_name bert-base-uncased --consider_real True --gold_data_num 100
        # python plotting/plot_embedding_distribution.py --small_model_name sentence-t5-base --syn_data_path $syn_data_path --llms llama-2-7b-chat-hf --task_name $task_name --num_use_samples_inner 6000 --logging_path=$logging_path --results_path=$results_path --gpu 0 --steps 4 --small_model_name bert-base-uncased --consider_real True --gold_data_num 100
        # python plotting/plot_embedding_distribution.py --small_model_name sentence-t5-base --syn_data_path $syn_data_path --llms opt-6.7b --task_name $task_name --num_use_samples_inner 6000 --logging_path=$logging_path --results_path=$results_path --gpu 0 --steps 4 --small_model_name bert-base-uncased --consider_real True --gold_data_num 100
        # python plotting/plot_embedding_distribution.py --small_model_name sentence-t5-base --syn_data_path $syn_data_path --llms vicuna-7b-1.5v --task_name $task_name --num_use_samples_inner 6000 --logging_path=$logging_path --results_path=$results_path --gpu 0 --steps 4 --small_model_name bert-base-uncased --consider_real True --gold_data_num 100
        
        # python plotting/plot_embedding_distribution.py --small_model_name sentence-t5-base --syn_data_path $syn_data_path --llms gpt2-xl --task_name $task_name --num_use_samples_inner 6000 --logging_path=$logging_path --gpu 0 --steps 4 --small_model_name bert-base-uncased --consider_real True --gold_data_num 1000
        
        # # python plotting/plot_embedding_distribution.py --small_model_name sentence-t5-base --syn_data_path $syn_data_path --llms gpt2-xl llama-2-7b-chat-hf vicuna-7b-1.5v opt-6.7b chatglm3-6b-base flan-t5-xl --task_name imdb --num_use_samples_inner 400 400 400 400 400 400 --gpu 0 --small_model_name bert-base-uncased --consider_real True --gold_data_num 400
        # # python plotting/plot_embedding_distribution.py --small_model_name sentence-t5-base --syn_data_path $syn_data_path --llms gpt2-xl llama-2-7b-chat-hf vicuna-7b-1.5v opt-6.7b chatglm3-6b-base flan-t5-xl --task_name imdb --num_use_samples_inner 200 200 200 200 200 200 --gpu 0 --small_model_name bert-base-uncased --consider_real True --gold_data_num 400
        # # python plotting/plot_embedding_distribution.py --small_model_name sentence-t5-base --syn_data_path $syn_data_path --llms gpt2-xl llama-2-7b-chat-hf vicuna-7b-1.5v opt-6.7b chatglm3-6b-base flan-t5-xl --task_name imdb --num_use_samples_inner 1000 1000 1000 1000 1000 1000 --gpu 2 --small_model_name bert-base-uncased
    done
done