export PYTHONPATH=$PYTHONPATH:$(pwd)

folder=
python3 evaluation/evaluate_bm25.py    \
    --document_path /shared/yanzhen4/Set-based-Retrieval/data/quest/quest_text_w_id.jsonl \
    --query_path /shared/yanzhen4/Set-based-Retrieval/data/quest/quest_test.jsonl \
    --result_dir /shared/yanzhen4/Set-based-Retrieval/output_baselines/results

python3 evaluation/evaluate_bm25.py    \
    --document_path /shared/yanzhen4/Set-based-Retrieval/data/quest/train_filtered_augment_full/quest_text_w_id_filtered_augmented_full.jsonl \
    --query_path /shared/yanzhen4/Set-based-Retrieval/data/quest/train_filtered_augment_full/quest_test_augment_full.jsonl \
    --result_dir /shared/yanzhen4/Set-based-Retrieval/output_baselines/results_augment