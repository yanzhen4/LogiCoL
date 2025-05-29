]# Change the following paths to your own paths
cd /shared/yanzhen4/LogiCoL
export HF_HOME=/shared/yanzhen4/models_cache
export HUGGINGFACE_HUB_CACHE=/shared/yanzhen4/models_cache
export PYTHONPATH=$(pwd):PYTHONPATH
export NCCL_P2P_DISABLE=1

experiment_name="quest_LogiCol"
model_name="intfloat/e5-base-v2" # Also try GTR, GTE, and Contriever! 

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 main.py \
    --train \
    --validate \
    --experiment_name $experiment_name \
    --model_name  $model_name \
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
    --output_dir /shared/yanzhen4/Set-based-Retrieval/output_release/trained_models \
    --cache_dir /shared/yanzhen4/Set-based-Retrieval/output_release/cache \