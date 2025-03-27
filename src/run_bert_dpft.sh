
task_list=("imdb")
# task_list=("banking")
# task_list=("imdb" "yelpRating" "yelpCategory" "openreviewRating" "openreviewCategory" "banking")
# task_list=("imdb" "yelpRating")
model_name_list=("bert-base-uncased")
# originally, batch-size=8 for train and eval
num_real_samples=(100 300 500)
num_real_samples=(500)
num_real_samples=(300)
num_real_samples_list=(100)
for task_name in "${task_list[@]}"; do
    for model_name in "${model_name_list[@]}"; do
        for num_real in "${num_real_samples_list[@]}"; do
            # if [ "$task_name" == "imdb" ]; then
            #     if [ "$model_name" == "gpt2-xl" ]; then
            #         echo "条件满足，通过。"
            #         continue
            #     fi
            # fi
            echo $task_name, $model_name
            # python -m torch.distributed.run --nproc_per_node 3 main_stm_dpft.py \
            # --output_dir "{$task_name}_{$num_real}_{$model_name}_10epoch_lora" \
            # --model_name $model_name \
            # --sequence_len 512 \
            # --num_classes 2 \
            # --label_name labels \
            # --per_device_train_batch_size 8 \
            # --gradient_accumulation_steps 1 \
            # --evaluation_strategy steps \
            # --eval_steps 5 \
            # --log_level info \
            # --per_device_eval_batch_size 8 \
            # --eval_accumulation_steps 1 \
            # --seed 42 \
            # --target_epsilon 100 \
            # --per_sample_max_grad_norm 1.0 \
            # --prediction_loss_only \
            # --weight_decay 0.01 \
            # --remove_unused_columns False \
            # --num_train_epochs 10 \
            # --logging_steps 1 \
            # --lora_dim 4 \
            # --lora_alpha 32 \
            # --lora_dropout 0.0 \
            # --max_grad_norm 0 \
            # --lr_scheduler_type constant \
            # --learning_rate 3e-4 \
            # --disable_tqdm True \
            # --dataloader_num_workers 2 \
            # --label_names labels \
            # --enable_lora \
            # --gold_dataset $task_name \
            # --ood_gold True \
            # --num_gold_samples $num_real


            
            python -m torch.distributed.run --nproc_per_node 3 main_stm_dpft.py \
            --output_dir "{$task_name}_{$num_real}_{$model_name}_10epoch_full" \
            --model_name $model_name \
            --sequence_len 512 \
            --num_classes 2 \
            --label_name labels \
            --per_device_train_batch_size 8 \
            --gradient_accumulation_steps 2 \
            --evaluation_strategy steps \
            --eval_steps 5 \
            --log_level info \
            --per_device_eval_batch_size 8 \
            --eval_accumulation_steps 1 \
            --seed 42 \
            --target_epsilon 100 \
            --per_sample_max_grad_norm 1.0 \
            --prediction_loss_only \
            --weight_decay 0.01 \
            --remove_unused_columns False \
            --num_train_epochs 10 \
            --logging_steps 1 \
            --max_grad_norm 0 \
            --lr_scheduler_type constant \
            --learning_rate 1e-4 \
            --disable_tqdm True \
            --dataloader_num_workers 2 \
            --label_names labels \
            --enable_lora \
            --gold_dataset $task_name \
            --ood_gold True \
            --num_gold_samples $num_real

        done
    done
done
