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


# plm_list=("gpt2-xl" "llama-2-7b-chat-hf" "vicuna-7b-1.5v" "opt-6.7b" "chatglm3-6b-base" "flan-t5-xl" "gpt-3.5-turbo-instruct" "gpt-4-turbo-preview")
plm_list=("gpt2-xl")
task_list=("openreviewCategory" "openreviewRating" "yelpCategory" "yelpRating")
plm_list=("opt-6.7b")
task_list=("imdb")
# plm_list=("gpt2-xl")
# task_list=("openreviewCategory")

# 枚举列表中的每个字符串
for task_name in "${task_list[@]}"; do
    for plm_name in "${plm_list[@]}"; do
        echo "Processing PLM: $plm_name, for task $task_name"
        # 在这里执行对$string的操作
        strings=(
          # # "/shared/project/PrivateGenerateEnhancement/src/data_accumulate/votingCLASS_promptALL_8_0.0_random_sampling/gold_10000_1_0.0_OODgold/imdb/${plm_name}_6000/bert-base-uncased/sentence-t5-base/increasedTheta_KD1_FuseDataset1/fewshotK8_15_0.5/0/"
          # "./data_accumulate/votingCLASS_promptCLASS_8_0.0_random_sampling/gold_100_1_0.0_OODgold/${task_name}/${plm_name}_6000/bert-base-uncased/sentence-t5-base/increasedTheta_KD1_FuseDataset1/fewshotK8_15_0.5/0/"      # "data_accumulate/influenceCartography/imdb/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__chatglm3-6b-base_1000__flan-t5-xl_1000+copy/bert-base-uncased/increasedTheta_KD1_FuseDataset1/fewshotK8_5_0.5/12345/"
          "./data_accumulate/votingCLASS_promptCLASS_8_0.0_random_sampling/gold_100_1_0.0_OODgold/${task_name}/${plm_name}_6000/bert-base-uncased/sentence-t5-base/increasedTheta_KD1_FuseDataset1/fewshotK8_15_0.5/0/"
        #   "./data_accumulate/votingCLASS_promptCLASS_8_0.0_random_sampling/gold_100_1_0.0_OODgold/openreviewCategory/gpt2-xl_1000/bert-base-uncased/sentence-t5-base/increasedTheta_KD1_FuseDataset1/fewshotK8_15_0.5/0/"
          # # "data_new/"
        )
        for syn_data_path in "${strings[@]}"; do
            # python plotting/plot_embedding_distribution.py --small_model_name sentence-t5-base --syn_data_path $syn_data_path --llms $plm_name --task_name $task_name --num_use_samples_inner 2400 --steps 1 --gpu 0 --small_model_name bert-base-uncased --consider_real True --gold_data_num 100
            python plotting/plot_embedding_distribution.py --small_model_name sentence-t5-base --syn_data_path $syn_data_path --llms $plm_name --task_name $task_name --num_use_samples_inner 6000 --steps 4 --gpu 0 --small_model_name bert-base-uncased --consider_real True --gold_data_num 100
            # python plotting/plot_embedding_distribution.py --small_model_name sentence-t5-base --syn_data_path $syn_data_path --llms $plm_name --task_name $task_name --num_use_samples_inner 1000 --steps 4 --gpu 0 --small_model_name bert-base-uncased --consider_real True --gold_data_num 100
            # python plotting/plot_embedding_distribution.py --small_model_name sentence-t5-base --syn_data_path $syn_data_path --llms gpt2-xl llama-2-7b-chat-hf vicuna-7b-1.5v opt-6.7b chatglm3-6b-base flan-t5-xl --task_name imdb --num_use_samples_inner 400 400 400 400 400 400 --gpu 0 --small_model_name bert-base-uncased --consider_real True --gold_data_num 400
            # python plotting/plot_embedding_distribution.py --small_model_name sentence-t5-base --syn_data_path $syn_data_path --llms gpt2-xl llama-2-7b-chat-hf vicuna-7b-1.5v opt-6.7b chatglm3-6b-base flan-t5-xl --task_name imdb --num_use_samples_inner 200 200 200 200 200 200 --gpu 0 --small_model_name bert-base-uncased --consider_real True --gold_data_num 400
            # python plotting/plot_embedding_distribution.py --small_model_name sentence-t5-base --syn_data_path $syn_data_path --llms gpt2-xl llama-2-7b-chat-hf vicuna-7b-1.5v opt-6.7b chatglm3-6b-base flan-t5-xl --task_name imdb --num_use_samples_inner 1000 1000 1000 1000 1000 1000 --gpu 2 --small_model_name bert-base-uncased
        done
    done
done