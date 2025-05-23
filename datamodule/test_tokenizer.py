import json
from transformers import AutoTokenizer
from tqdm import tqdm

doc_file_path = '/shared/yanzhen4/Set-based-Retrieval/data/dbpedia/dbpedia_text_w_id.jsonl'

model_path = "sentence-transformers/sentence-t5-base"

tokenizer = AutoTokenizer.from_pretrained(model_path)

doc_data = []
token_lengths = []
doc_count = 0

with open(f"{doc_file_path}") as fin:
    for line in tqdm(fin):
        data = json.loads(line)
        text = data['text']
        tokens = tokenizer.tokenize(text)
        token_lengths.append(len(tokens))
        print(len(tokens))
        doc_count += 1
        if doc_count > 100:
            break

max_length = max(token_lengths)
avg_length = sum(token_lengths) / len(token_lengths)

print(f"Max token length: {max_length}")
print(f"Average token length: {avg_length}")