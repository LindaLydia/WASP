
task_list=("imdb")
# plm_list=("gpt2-xl" "llama-2-7b-chat-hf" "vicuna-7b-1.5v" "opt-6.7b" "chatglm3-6b-base" "flan-t5-xl" "gpt-3.5-turbo-instruct" "gpt-4-turbo-preview" "gpt-4o")
# plm_list=("gpt2-xl" "llama-2-7b-chat-hf" "vicuna-7b-1.5v" "opt-6.7b" "chatglm3-6b-base" "flan-t5-xl")
plm_list=("gpt2-xl" "llama-2-7b-chat-hf" "flan-t5-xl")
# plm_list=("llama-2-7b-chat-hf" "flan-t5-xl")
# plm_list=("flan-t5-xl")
# plm_list=("llama-2-7b-chat-hf")
# plm_list=("gpt2-xl")
# 枚举列表中的每个字符串
for task_name in "${task_list[@]}"; do
    for plm_name in "${plm_list[@]}"; do
        echo "Processing single PLM $plm_name PE, for task $task_name"
        # 在这里执行对$string的操作
        strings=(
          # "./data_accumulate/votingCLASS_promptCLASS_8_0.0_randomContrast_top/gold_100_1_0.0_OODgold/${task_name}/flan-t5-xl_6000/bert-base-uncased/sentence-t5-base/increasedTheta_KD1_FuseDataset1/fewshotK8_15_0.5/0/"
          # "./data_accumulate/votingCLASS_promptCLASS_8_0.0_random_top/gold_100_1_0.0_OODgold/${task_name}/flan-t5-xl_6000/bert-base-uncased/sentence-t5-base/increasedTheta_KD1_FuseDataset1/fewshotK8_15_0.5/0/"
          
          "./data_accumulate/votingCLASS_promptCLASS_1_2.4224026313026945_randomPE_top/gold_100_1_0.0_OODgold/${task_name}/${plm_name}_6000/bert-base-uncased/sentence-t5-base/increasedTheta_KD1_FuseDataset1/fewshotK8_15_0.5/0/"
        )
        logging_strings=(
          # "./logging/multiGold_eval_on_real/with_real_few_shot_accumulate_votingCLASS_promptCLASS_8_0.0_randomContrast_top/gold_100_1_0.0_OODgold/bert-base-uncased/sentence-t5-base/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD1_FuseDataset1/0_1_init1200_steps4_unbalance_temp1.0/fewshotK8_15_0.5/imdb/flan-t5-xl_6000/log_data_accumulate_4.89894920704081_0.txt"
          # "./logging/multiGold_eval_on_real/with_real_few_shot_accumulate_votingCLASS_promptCLASS_8_0.0_random_top/gold_100_1_0.0_OODgold/bert-base-uncased/sentence-t5-base/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD1_FuseDataset1/0_1_init1200_steps4_unbalance_temp1.0/fewshotK8_15_0.5/imdb/flan-t5-xl_6000/log_data_accumulate_4.89894920704081_0.txt"
          
          "./logging/multiGold_eval_on_real/with_real_few_shot_accumulate_votingCLASS_promptCLASS_1_2.4224026313026945_randomPE_top/gold_100_1_0.0_OODgold/bert-base-uncased/sentence-t5-base/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD1_FuseDataset1/0_1_init1200_steps4_unbalance_temp1.0/fewshotK8_15_0.5/${task_name}/${plm_name}_6000/log_data_accumulate_4.89894920704081_0.txt"
        )
        result_strings=(
          # "./results/multiGold_eval_on_real/with_real_few_shot_accumulate_votingCLASS_promptCLASS_8_0.0_randomContrast_top/gold_100_1_0.0_OODgold/bert-base-uncased/sentence-t5-base/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD1_FuseDataset1/0_1_init1200_steps4_unbalance_temp1.0/fewshotK8_15_0.5/imdb/flan-t5-xl_6000/0/"
          # "./results/multiGold_eval_on_real/with_real_few_shot_accumulate_votingCLASS_promptCLASS_8_0.0_random_top/gold_100_1_0.0_OODgold/bert-base-uncased/sentence-t5-base/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD1_FuseDataset1/0_1_init1200_steps4_unbalance_temp1.0/fewshotK8_15_0.5/imdb/flan-t5-xl_6000/0/"

          "./results/multiGold_eval_on_real/with_real_few_shot_accumulate_votingCLASS_promptCLASS_1_2.4224026313026945_randomPE_top/gold_100_1_0.0_OODgold/bert-base-uncased/sentence-t5-base/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD1_FuseDataset1/0_1_init1200_steps4_unbalance_temp1.0/fewshotK8_15_0.5/${task_name}/${plm_name}_6000/0"
        )

        for i in $(seq 0 $((${#strings[@]}-1))); do
            syn_data_path=${strings[$i]}
            logging_path=${logging_strings[$i]}
            results_path=${result_strings[$i]}
            echo "syn_data_path = $syn_data_path"
            echo "logging_path = $logging_path"
            echo "results_path = $results_path"
            # python plotting/plot_embedding_distribution.py --small_model_name sentence-t5-base --syn_data_path $syn_data_path --llms flan-t5-xl --task_name $task_name --num_use_samples_inner 6000 --logging_path=$logging_path --results_path=$results_path --gpu 0 --steps 4 --small_model_name bert-base-uncased --consider_real True --gold_data_num 100
            python plotting/plot_embedding_distribution.py --small_model_name sentence-t5-base --syn_data_path $syn_data_path --llms $plm_name --task_name $task_name --num_use_samples_inner 6000 --logging_path=$logging_path --results_path=$results_path --gpu 0 --steps 4 --small_model_name bert-base-uncased --consider_real True --gold_data_num 100
        done
    done
done