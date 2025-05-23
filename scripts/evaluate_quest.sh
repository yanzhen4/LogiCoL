#!/bin/bash
#SBATCH --partition=p_nlp
#SBATCH --job-name=multi_model_eval
#SBATCH --output=/shared/sihaoc/project/set_ir/Set-based-Retrieval/log/multi_model_eval_large_quest.txt
#SBATCH --mem=500G
#SBATCH --nodelist=nlpgpu10
#SBATCH --gpus=8
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=48

cd /shared/yanzhen4/LogiCoL
export HF_HOME=/shared/yanzhen4/models_cache
export HUGGINGFACE_HUB_CACHE=/shared/yanzhen4/models_cache
export PYTHONPATH=$(pwd):PYTHONPATH
export NCCL_P2P_DISABLE=1

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 evaluation/evaluate.py \
    --model_name /shared/yanzhen4/LogiCoL/output/trained_models/quest_setIR_e5-base-v2_lr1e-05_bs4_len512_fp16/epoch=9-step=3930.ckpt \
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
    
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 evaluation/evaluate.py \
    --model_name /shared/yanzhen4/LogiCoL/output/trained_models/quest_setIR_e5-base-v2_lr1e-05_bs4_len512_fp16/epoch=9-step=3930.ckpt \
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