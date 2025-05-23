from enum import Enum
import os
import json
from typing import Dict, List, Any, Optional, Literal
import pickle as pkl
import torch
from tqdm import tqdm
import numpy as np

def read_jsonl(file: str) -> List[dict]:
    examples = []
    with open(file, 'r') as fin:
        for line in fin:
            examples.append(json.loads(line))

    return examples

def load_idx2title(file: str) -> dict:
    data = read_jsonl(file)
    idx2title = {}

    for doc_data in data:
        title = doc_data['title']
        idx = doc_data['idx']
        idx2title[idx] = title
    
    return idx2title

def read_all_jsonl(list_of_files: List[str]) -> List[dict]:
    """
    Read a list of jsonl files into one list of dictionaries.
    
    Args:
        list_of_files: List of file paths.
    """
    examples = []
    for f in list_of_files:
        examples += read_jsonl(f)

    return examples

def get_file_name(path):
    """
    Given a file path, return the file name without the extension.

    Args:
        path: POSIX style file path
    """
    return os.path.splitext(os.path.basename(path))[0]

def get_hf_base_model_name(model_name):
    return model_name.split("/")[-1]

def get_ckpt_model_name(ckpt_path):
    """
    Get the name of the finetuned model from its checkpoint path.
    e.g. .../output/quest_gtr-t5-base_lr0.0001_bs64_32/epoch=0.ckpt
    """
    return os.path.basename(os.path.dirname(ckpt_path))

def save_jsonl(examples: List[Any], output_path: str) -> None:
    with open(output_path, 'w') as fout:
        for ex in tqdm(examples):
            fout.write(json.dumps(ex))
            fout.write("\n")

def save_matrix(matrix: torch.Tensor, queries: List[dict], output_path: str) -> None:
    examples = []
    num_queries, num_docs = matrix.size()
    
    for i in tqdm(range(num_queries)):
        query_result = {
            'query': queries[i],
            'results': matrix[i].tolist()
        }
        examples.append(query_result)
    
    save_jsonl(examples, output_path)

def save_dict(subquery2doc_sim: Dict[str, torch.Tensor], output_path: str) -> None:
    examples = []
    
    for subquery, sim in subquery2doc_sim.items():
        if isinstance(sim, torch.Tensor):
            sim = sim.tolist()
            
        subquery_result = {
            'query': subquery,
            'results': sim
        }
        examples.append(subquery_result)
    
    save_jsonl(examples, output_path)

def save_query2doc_result(queries, all_result, all_doc_ids, result_dir, query_path, base_model_name, file_name):
    query2doc_result = {}

    all_doc_ids_array = np.array(all_doc_ids)

    for query, doc_result in tqdm(zip(queries, all_result)):
        similarities = np.zeros(len(all_doc_ids))
        similarities[all_doc_ids_array] = doc_result.cpu().numpy()
        query2doc_result[query] = list(similarities)

    save_dict(query2doc_result, f'{result_dir}/{get_file_name(query_path)}_{base_model_name}_{file_name}.json')

def save_dict_sparse(subquery2doc_sim: Dict[str, torch.Tensor], output_path: str) -> None:
    examples = []
    
    for subquery, sim_tensor in tqdm(subquery2doc_sim.items()):
        non_zero_entries = [(i, sim_tensor[i].item()) for i in range(sim_tensor.size(0)) if sim_tensor[i] != 0]
        subquery_result = {
            'query': subquery,
            'similarities': non_zero_entries
        }
        examples.append(subquery_result)
    
    save_jsonl(examples, output_path)

def read_tab_sep(file: str) -> List[List[str]]:
    examples = []
    with open(file, 'r') as fin:
        for line in fin:
            examples.append(line.strip().split("\t"))

    return examples

def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pkl.dump(obj, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pkl.load(f)


class CacheType(str, Enum):
    EMBEDDINGS: str = "embeddings"
    TOKENIZATION: str = "tokenized"

def get_cache_path(
    cache_dir: str,
    base_model_name: str,
    file_name: str,
    cache_type: Optional[CacheType] = None):
    """
    Given a path, return the cache path.
    
    Args:
    - cache_dir: Directory of the cache
    - base_model_name: name of the model being tested
    - file_name: name of the file being processed and cached
    - cache_type: type of cache (e.g. embeddings, tokenization, etc.)
    """
    dir_name = file_name + "_" + base_model_name
    if cache_type is not None:
        dir_path = os.path.join(cache_dir, dir_name, cache_type)
    else:
        dir_path = os.path.join(cache_dir, dir_name)

    return dir_path

def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    else:
        print(f"Directory {directory_path} already exists.")
    return directory_path