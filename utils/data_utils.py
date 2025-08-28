import json
import os
from config import paths
from datasets import Dataset as HFDataset
from typing import Dict, List, Iterable, Callable, Tuple, Dict, Optional
import pandas as pd
import torch
import numpy as np

import multiprocessing
from utils.file_utils import read_jsonl, read_all_jsonl, save_jsonl
from utils.tensor_utils import make_square_matrix_symmetric, sparse_indices_to_dense_matrix, make_strict_neg_mask, make_neg_mask

def get_entities_from_query(query: str, query_to_ent_map: Dict[str, List[str]]):
    """Extract entities from the query."""
    query = query.lower()
    if query in query_to_ent_map:
        return query_to_ent_map[query]
    else:
        raise ValueError(f"Query {query} not found in the query to entity map.")
        
class LogicalQuery(object):
    def __init__(self,
                 sub_queries: List[str],
                 operators: List[str] = [],
                 nl_query: Optional[str] = None):
        
        if operators is not None:
            assert len(sub_queries) == len(operators) + 1, "Number of sub-queries should be one more than the number of operators"
    
        self.sub_queries = sub_queries
        self.operators = operators

        self.nl_query = nl_query

    def get_ground_truth(self, query_to_ent_map: Dict[str, List[str]]):
        """Get the ground truth entities for the logical query.
        Note that this assumes that the first query is always a positive query, not negation.
        """
        ground_truth = set()
        for i in range(len(self.sub_queries)):
            q = self.sub_queries[i]
            cur_set = set(get_entities_from_query(q, query_to_ent_map))

            if i == 0:
                ground_truth = cur_set
            else:
                cur_op = self.operators[i - 1]
                if cur_op == 'AND':
                    ground_truth = ground_truth.intersection(cur_set)
                elif cur_op == 'OR':
                    ground_truth = ground_truth.union(cur_set)
                elif cur_op == 'NOT':
                    ground_truth = ground_truth.difference(cur_set)
                else:
                    raise ValueError(f"Unknown operator {self.template.ops[i - 1]}")

        return list(ground_truth)     
        
class BatchQueryDocInput:
    """
    Contains input to the model + labels for loss computation, including query/document indices
    and relations.
    """
    def __init__(self,
                 input_ids,
                 attention_mask,
                 pos_mask=None,
                 unallowed_neg_mask=None,
                 query_indices=None,
                 document_indices=None,
                 subset_relation=None,
                 exclusion_relation=None,
                 **kwargs):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.pos_mask = pos_mask

        self.query_indices = query_indices if query_indices is not None else torch.tensor([])
        self.document_indices = document_indices if document_indices is not None else torch.tensor([])

        self.subset_relation = subset_relation if subset_relation is not None else torch.tensor([])
        self.exclusion_relation = exclusion_relation if exclusion_relation is not None else torch.tensor([])

        self.unallowed_neg_mask = unallowed_neg_mask if unallowed_neg_mask is not None else torch.tensor([])

        self.metadata = kwargs

    def to(self, device):
        """
        Transfer all tensors to the specified device.
        """
        self.input_ids = self.input_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)

        self.pos_mask = self.pos_mask.to(device)
        self.unallowed_neg_mask = self.unallowed_neg_mask.to(device)

        self.query_indices = self.query_indices.to(device)
        self.document_indices = self.document_indices.to(device)

        self.subset_relation = self.subset_relation.to(device)
        self.exclusion_relation = self.exclusion_relation.to(device)

        return self

#TODO: Compute negative mask here
def random_sample_to_features_random(
    batch_examples: List[dict],
    square_matrix_symmetric: bool=False) -> BatchQueryDocInput:
    """
    Collate a batch of examples into input for the model.
    """
    all_input_ids = []
    all_attention_mask = []
    all_pos_indices = []  # List of (row, col) indices for positive pairs
    queries_nl_queries = []
    queries_documents = []
    queries_length = []

    for example in batch_examples:
        query_pos_in_batch = len(all_input_ids)
        query = example['query']
        documents = example['documents']
            
        # Add query
        all_input_ids.append(query["input_ids"])
        all_attention_mask.append(query["attention_mask"])
        queries_nl_queries.append(query['nl_query'])
        queries_length.append(len(query['queries']))
        
        assert len(documents) == 1 # Since using multiple documents will harm performance, we only consider one document per query

        # Add each document
        for doc in documents:
            doc_idx = len(all_input_ids)
            all_input_ids.append(doc["input_ids"])
            all_attention_mask.append(doc["attention_mask"])
            queries_documents.append(doc['title'])
            
            # Add the query as positive example
            all_pos_indices.append((query_pos_in_batch, doc_idx))

    # Make positive and negative masks for the batch
    pos_mask = sparse_indices_to_dense_matrix(
        indices=all_pos_indices, 
        shape=(len(all_input_ids), len(all_input_ids))
    )

    if square_matrix_symmetric:
        pos_mask = make_square_matrix_symmetric(pos_mask)

    neg_mask = make_neg_mask(pos_mask)
    
    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)

    return BatchQueryDocInput(
        input_ids=all_input_ids, 
        attention_mask=all_attention_mask, 
        pos_mask=pos_mask
    )

def is_not_single_queries_group(queries: List[dict])-> bool:
    """
    Check if the all the queries in the group are related, not randomly batched single queries
    """
    for query in queries:
        if len(query['queries']) > 1:
            return True
    return False

def random_sample_to_features_group(
    batch_examples: List[dict], 
    max_text: int = 8, 
    num_pos_docs: int = 1, 
    flip_false_negative_full: bool = False,
    square_matrix_symmetric: bool = False) -> BatchQueryDocInput:

    """
    Collate a batch of examples into input for the model, accommodating query groups and
    storing subset and exclusion relations.
    max_test: maximum number of text to consider in the batch, to avoid memory issues
    """

    all_input_ids = []
    all_attention_mask = [] 

    query_indices = [] 
    document_indices = []

    subset_relations = []
    exclusion_relations = []

    all_pos_indices = []    
    all_non_negative_indices = []
    subset_indices_list = []
    exclusion_indices_list = []

    num_queries_curr = 0
   
    queries_nl_queries = []
    queries_documents_title = []
    queries_documents_idx = []
    queries_length = []
    
    query_idx2documents = {} # Mapping from query index to document global index
    query_idx2pos_document = {} # Mapping from query index to positive document global index
    query_idx2name = {} # Mapping from query index to nl_query name

    for example in batch_examples:
        
        queries = example["queries"] 
        
        pos_documents = example["pos_documents"] # A list of positive documents
        documents = example["documents"] # A list of list of documents 
        subset_relation = example['subset_relation']
        exclusion_relation = example['exclusion_relation']
        
        queries_idx_group = [] # List of group queries' indices in batch in the group
        documents_idx_group = [] # List of group documents' indices in batch in the groups

        for i, query in enumerate(queries):

            query_pos_in_batch = len(all_input_ids)
            query_indices.append(query_pos_in_batch)
            queries_idx_group.append(query_pos_in_batch)

            all_input_ids.append(query["input_ids"])
            all_attention_mask.append(query["attention_mask"])
    
            queries_nl_queries.append(query['nl_query'])
            queries_length.append(len(query['queries']))

            query_pos_docs = pos_documents[i]

            query_docs = documents[i]
            query_idx2documents[query_pos_in_batch] = [doc['idx'] for doc in query_docs]

            assert len(query_pos_docs) == num_pos_docs # Since using multiple documents will harm performance, we only consider one document per query
            query_idx2pos_document[query_pos_in_batch] = query_pos_docs[0]['idx']
            query_idx2name[query_pos_in_batch] = query['nl_query']

            for doc in query_pos_docs:
                doc_idx = len(all_input_ids)
                document_indices.append(doc_idx) 
                documents_idx_group.append(doc_idx)

                all_input_ids.append(doc["input_ids"])
                all_attention_mask.append(doc["attention_mask"])
    
                queries_documents_title.append(doc['title'])

                all_pos_indices.append((query_pos_in_batch, doc_idx))
            
        if is_not_single_queries_group(queries):
            for query_idx in queries_idx_group:
                for doc_idx in documents_idx_group:
                    all_non_negative_indices.append((query_idx, doc_idx))

        subset_indices = [(i + num_queries_curr, j + num_queries_curr) for i, j in subset_relation]
        exclusion_indices = [(i + num_queries_curr, j + num_queries_curr) for i, j in exclusion_relation]

        subset_indices_list.extend(subset_indices)
        exclusion_indices_list.extend(exclusion_indices)

        num_queries_curr += len(queries)
        
    if num_pos_docs == 1 and flip_false_negative_full:
        for idx1, (query_idx, doc_idx) in enumerate(zip(query_indices, document_indices)):
            pos_doc = query_idx2pos_document[query_idx]
            for query_idx2 in query_idx2documents:
                if pos_doc in query_idx2documents[query_idx2] and (query_idx2, doc_idx) not in all_pos_indices:
                    all_pos_indices.append((query_idx2, doc_idx))

    subset_relations = subset_indices_list
    exclusion_relations = exclusion_indices_list

    # If a query1 is a subset of another query2 , then the positive document of the query1 should also be a positive document of query2. This is a quick way to flip false negatives. 
    for (idx1, idx2) in subset_indices_list:
        if (query_indices[idx2], document_indices[idx1]) not in all_pos_indices:
            all_pos_indices.append((query_indices[idx2], document_indices[idx1]))

    pos_mask = sparse_indices_to_dense_matrix(
        indices=all_pos_indices, 
        shape=(len(all_input_ids), len(all_input_ids))
    )

    if square_matrix_symmetric:
        pos_mask = make_square_matrix_symmetric(pos_mask)

    unallowed_neg_mask = sparse_indices_to_dense_matrix(
        indices=all_non_negative_indices,
        shape=(len(all_input_ids), len(all_input_ids)),
        reverse=True
        # indices appeared in the list will be set to 0, otherwise 1
    )   

    assert len(all_input_ids) == len(query_indices) + len(document_indices)
    assert len(all_input_ids) == len(query_indices) * 2
    
    subset_relation_mask = sparse_indices_to_dense_matrix(
        indices=subset_indices_list,
        shape=(len(query_indices), len(query_indices))
    )

    exclusion_relation_mask = sparse_indices_to_dense_matrix(
        indices=exclusion_indices_list,
        shape=(len(query_indices), len(query_indices))
    )

    assert len(unallowed_neg_mask) == len(pos_mask)

    all_input_ids = torch.tensor(np.array(all_input_ids), dtype=torch.long)
    all_attention_mask = torch.tensor(np.array(all_attention_mask), dtype=torch.long)

    query_indices = torch.tensor(query_indices, dtype=torch.long)
    document_indices = torch.tensor(document_indices, dtype=torch.long)
   
    return BatchQueryDocInput(
        input_ids=all_input_ids, 
        attention_mask=all_attention_mask, 
        pos_mask=pos_mask,
        unallowed_neg_mask=unallowed_neg_mask,
        query_indices=query_indices,
        document_indices=document_indices,
        subset_relation=subset_relation_mask, 
        exclusion_relation=exclusion_relation_mask,
    )

#####################
### Dataset utils ###
#####################

def load_jsonl_as_hf_dataset(file: str) -> torch.utils.data.Dataset:
    """
    Load a jsonl file as a huggingface dataset.
    """
    data = read_jsonl(file)
    doc_df = pd.DataFrame(data)
    doc_ds = HFDataset.from_pandas(doc_df)
    return doc_ds

def load_jsonl_as_hf_dataset_combine(file: str) -> torch.utils.data.Dataset:
    """
    Load a the query file subqueries as huggingface dataset. 
    Each row has one subqueries and corresponding operators, nl_query, and documents
    """
    data = read_jsonl(file)
    subqueries_data = []
    all_queries_reorg = []

    operators = []
    documents = []
    unique_subqueries = set()

    idx = 0
    for query_data in data:
        queries = query_data['queries']
        nl_query = query_data['nl_query']
        query_operators = query_data['operators']
        query_documents = query_data['documents']

        for query in queries:
            
            if query not in unique_subqueries:
                subqueries_data.append({'subquery': query})
                unique_subqueries.add(query)

        idx += 1

        all_queries_reorg.append({'queries': queries, 'operators': query_operators, 'nl_query': nl_query, 'documents': query_documents})
        
    query_df = pd.DataFrame(subqueries_data)
    query_ds = HFDataset.from_pandas(query_df)

    return query_ds, all_queries_reorg

def add_tokenization_to_dataset(
    dataset: HFDataset,
    tokenizer: Callable,
    key_to_tokenize: str,
    max_seq_length: int,
    num_workers: int = 32) -> HFDataset:

    
    dataset = dataset.map(
        lambda x: tokenizer(
            x[key_to_tokenize], 
            return_tensors="pt",
            max_length=max_seq_length,
            padding='max_length',
            truncation=True), 
        batched=True,
        num_proc=num_workers
    )

    return dataset


#####################
### DBPEDIA utils ###
#####################

def load_dbpedia_title2id(file: str = paths.DBPEDIA_TITLE2IDX) -> Dict[int, str]:
    
    title2id = {}
    
    _maps = read_jsonl(file)

    for m in _maps:
        title2id[m['title']] = m['idx']

    return title2id

def construct_dbpedia_id2text(
    title2id: str = paths.DBPEDIA_TITLE2IDX,
    title2text: str = paths.DBPEDIA_TITLE2TEXT,
    output_file: str = paths.DBPEDIA_TEXT) -> Dict[str, str]:
    """
    One time script to construct a dbpedia text jsonl file with id, title, text, and topic label all in one place. 
    """
    
    title2id = load_dbpedia_title2id(file = title2id)

    title2text = read_jsonl(title2text)
    
    with open(output_file, 'w') as fout:
        for ex in title2text:
            ex['idx'] = title2id[ex['title']]
            fout.write(json.dumps(ex))
            fout.write("\n")

def merge_dbpedia_splits(    
    split_info: dict = paths.DBPEDIA_QUERY_SPLIT,
    query_dir: str = paths.DBPEDIA_QUERY_DIR,
    query_file_pattern: str = paths.DBPEDIA_QUERY_FILE_PATTERN,
    output_dir: str = paths.DBPEDIA_DIR) -> None:
    """
    One time script to merge all the dbpedia splits into one file. 
    """
    for split in split_info:
        
        output_path = os.path.join(output_dir, f"dbpedia_{split}.jsonl")
        all_jsonl_path = [
            os.path.join(query_dir, query_file_pattern.format(s))
            for s in split_info[split]]
            
        _data = read_all_jsonl(all_jsonl_path)
        save_jsonl(_data, output_path)
        

def load_dbpedia(file: str = paths.DBPEDIA_TEXT) -> Dict[str, str]:
    return read_jsonl(file)

def load_dbpedia_id2text(file: str = paths.DBPEDIA_TEXT) -> Dict[str, str]:
    dbpedia = load_dbpedia(file)

    id2text = {}
    for ex in dbpedia:
        id2text[ex['idx']] = ex['text']

    return id2text

def pmap(func: Callable,
         items: Iterable,
         num_threads: int):
    """
    Applies a function on each element in a list in parallel
    :param items:
    :param func:
    :param num_threads:
    :return:
    """
    with multiprocessing.Pool(num_threads) as p:
        res = p.map(func, items)

    return res

# if __name__ == "__main__":
    # construct_dbpedia_id2text()
    # merge_dbpedia_splits()
