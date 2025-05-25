# Change the following paths to your own paths
cd /shared/yanzhen4/LogiCoL
export HF_HOME=/shared/yanzhen4/models_cache
export HUGGINGFACE_HUB_CACHE=/shared/yanzhen4/models_cache
export PYTHONPATH=$(pwd):PYTHONPATH
export NCCL_P2P_DISABLE=1

model_path="/shared/yanzhen4/LogiCoL/output/trained_models/quest_setIR_e5-base-v2_lr1e-05_bs4_len512_fp16/epoch=9-step=3930.ckpt"

# Quest
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 evaluation/evaluate.py \
    --model_name $model_path \
    --document_path /shared/yanzhen4/LogiCoL/data/quest_text_w_id.jsonl \
    --document_text_key text \
    --query_path /shared/yanzhen4/LogiCoL/data/quest_test.jsonl \
    --query_text_key nl_query \
    --gpus 4 \
    --cache_dir /shared/yanzhen4/LogiCoL/output/cache \
    --result_dir /shared/yanzhen4/LogiCoL/output/results \
    --batch_size 64 \
    --precision 32 \
    --max_seq_length 512 \
    --num_workers 32 \
    --load_cached_embeddings \

# Quest with variants 
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 evaluation/evaluate.py \
    --model_name $model_path \
    --document_path /shared/yanzhen4/LogiCoL/data/quest_text_w_id_withVarients.jsonl \
    --document_text_key text \
    --query_path /shared/yanzhen4/LogiCoL/data/quest_test_withVarients.jsonl \
    --query_text_key nl_query \
    --gpus 4 \
    --cache_dir /shared/yanzhen4/LogiCoL/output/cache \
    --result_dir /shared/yanzhen4/LogiCoL/output/results \
    --batch_size 64 \
    --precision 32 \
    --max_seq_length 512 \
    --num_workers 32 \
    --load_cached_embeddings \