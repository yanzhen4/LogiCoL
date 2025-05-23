cd /shared/yanzhen4/LogiCoL
export HF_HOME=/shared/yanzhen4/models_cache
export HUGGINGFACE_HUB_CACHE=/shared/yanzhen4/models_cache
export PYTHONPATH=$(pwd):PYTHONPATH
export NCCL_P2P_DISABLE=1

experiment_name="quest_setIR"
exclusion_loss_weight=0.1
exclusion_loss_margin=0.1
subset_loss_weight=0.2
subset_loss_margin=0.2

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 main.py \
#     --train \
#     --validate \
#     --experiment_name $experiment_name \
#     --model_name  sentence-transformers/gtr-t5-base \
#     --train_queries /shared/yanzhen4/Set-based-Retrieval/data/quest/train_filtered_augment_full/quest_train_filtered_augment_full.jsonl \
#     --val_queries /shared/yanzhen4/Set-based-Retrieval/data/quest/train_filtered_augment_full/quest_val_augment_full.jsonl \
#     --test_queries /shared/yanzhen4/Set-based-Retrieval/data/quest/train_filtered_augment_full/quest_test_augment_full.jsonl \
#     --documents /shared/yanzhen4/Set-based-Retrieval/data/quest/train_filtered_augment_full/quest_text_w_id_filtered_augmented_full.jsonl \
#     --train_batch_size 1 \
#     --gpus 4 \
#     --num_workers 16 \
#     --precision "fp16" \
#     --learning_rate 1e-5 \
#     --max_seq_length 512 \
#     --warmup_steps 25 \
#     --num_epoch 10 \
#     --lr_scheduler constant \
#     --project_name set-ir_release \
#     --batch_strategy group \
#     --alpha 0.5 \
#     --exclusion_loss_weight $exclusion_loss_weight \
#     --exclusion_loss_margin $exclusion_loss_margin \
#     --subset_loss_weight $subset_loss_weight \
#     --subset_loss_margin $subset_loss_margin \
#     --output_dir /shared/yanzhen4/Set-based-Retrieval/output_release/trained_models \
#     --cache_dir /shared/yanzhen4/Set-based-Retrieval/output_release/cache \

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 main.py \
#     --train \
#     --validate \
#     --experiment_name $experiment_name \
#     --model_name  thenlper/gte-base \
#     --train_queries /shared/yanzhen4/Set-based-Retrieval/data/quest/train_filtered_augment_full/quest_train_filtered_augment_full.jsonl \
#     --val_queries /shared/yanzhen4/Set-based-Retrieval/data/quest/train_filtered_augment_full/quest_val_augment_full.jsonl \
#     --test_queries /shared/yanzhen4/Set-based-Retrieval/data/quest/train_filtered_augment_full/quest_test_augment_full.jsonl \
#     --documents /shared/yanzhen4/Set-based-Retrieval/data/quest/train_filtered_augment_full/quest_text_w_id_filtered_augmented_full.jsonl \
#     --train_batch_size 1 \
#     --gpus 4 \
#     --num_workers 16 \
#     --precision "fp16" \
#     --learning_rate 2e-5 \
#     --max_seq_length 512 \
#     --warmup_steps 25 \
#     --num_epoch 10 \
#     --lr_scheduler constant \
#     --project_name set-ir_final_verify_0424 \
#     --batch_strategy group \
#     --alpha 0.5 \
#     --exclusion_loss_weight $exclusion_loss_weight \
#     --exclusion_loss_margin $exclusion_loss_margin \
#     --subset_loss_weight $subset_loss_weight \
#     --subset_loss_margin $subset_loss_margin \
#     --output_dir /shared/yanzhen4/Set-based-Retrieval/output_analysis/trained_models \
#     --cache_dir /shared/yanzhen4/Set-based-Retrieval/output_analysis/cache \


# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 main.py \
#     --train \
#     --validate \
#     --experiment_name $experiment_name \
#     --model_name  facebook/contriever \
#     --train_queries /shared/yanzhen4/Set-based-Retrieval/data/quest/train_filtered_augment_full/quest_train_filtered_augment_full.jsonl \
#     --val_queries /shared/yanzhen4/Set-based-Retrieval/data/quest/train_filtered_augment_full/quest_val_augment_full.jsonl \
#     --test_queries /shared/yanzhen4/Set-based-Retrieval/data/quest/train_filtered_augment_full/quest_test_augment_full.jsonl \
#     --documents /shared/yanzhen4/Set-based-Retrieval/data/quest/train_filtered_augment_full/quest_text_w_id_filtered_augmented_full.jsonl \
#     --train_batch_size 1 \
#     --gpus 4 \
#     --num_workers 16 \
#     --precision "fp16" \
#     --learning_rate 1e-5 \
#     --max_seq_length 512 \
#     --warmup_steps 25 \
#     --num_epoch 10 \
#     --lr_scheduler constant \
#     --project_name set-ir_final_verify_0424 \
#     --batch_strategy group \
#     --alpha 0.5 \
#     --exclusion_loss_weight $exclusion_loss_weight \
#     --exclusion_loss_margin $exclusion_loss_margin \
#     --subset_loss_weight $subset_loss_weight \
#     --subset_loss_margin $subset_loss_margin \
#     --output_dir /shared/yanzhen4/Set-based-Retrieval/output_analysis/trained_models \
#     --cache_dir /shared/yanzhen4/Set-based-Retrieval/output_analysis/cache \

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 main.py \
    --train \
    --validate \
    --experiment_name $experiment_name \
    --model_name  intfloat/e5-base-v2 \
    --train_queries /shared/yanzhen4/LogiCoL/data/quest_train_withVarients.jsonl \
    --val_queries /shared/yanzhen4/LogiCoL/data/quest_val_withVarients.jsonl \
    --test_queries /shared/yanzhen4/LogiCoL/data/quest_test_withVarients.jsonl \
    --documents /shared/yanzhen4/LogiCoL/data/quest_text_w_id_withVarients.jsonl \
    --train_batch_size 1 \
    --gpus 4 \
    --num_workers 16 \
    --precision "fp16" \
    --learning_rate 1e-5 \
    --max_seq_length 512 \
    --warmup_steps 25 \
    --num_epoch 10 \
    --lr_scheduler constant \
    --project_name set-ir_release \
    --batch_strategy group \
    --alpha 0.5 \
    --exclusion_loss_weight $exclusion_loss_weight \
    --exclusion_loss_margin $exclusion_loss_margin \
    --subset_loss_weight $subset_loss_weight \
    --subset_loss_margin $subset_loss_margin \
    --output_dir /shared/yanzhen4/LogiCoL/output/trained_models \
    --cache_dir /shared/yanzhen4/LogiCoL/output/cache \
