export PYTHONPATH=$PYTHONPATH:$(pwd)

folder="/shared/yanzhen4/Set-based-Retrieval/data/"

python3 evaluation/evaluate_bm25.py    \
    --document_path $folder/quest_text_w_id.jsonl \
    --query_path $folder/quest_test.jsonl \
    --result_dir $folder/output/results

python3 evaluation/evaluate_bm25.py    \
    --document_path $folder/quest_text_w_id_withVarients.jsonl \
    --query_path $folder/quest_test_withVarients.jsonl \
    --result_dir $folder/output/results