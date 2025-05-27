"""
Dataset class for query document pair
"""
import logging
import os
import random
import json

from datasets import Dataset as HFDataset
from functools import partial
from itertools import chain, combinations
from pytorch_lightning import LightningDataModule
from pandas import DataFrame
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset as TorchDataset
from transformers import AutoTokenizer
from typing import List, Dict
from utils.file_utils import read_jsonl, get_file_name, get_hf_base_model_name, get_cache_path, CacheType
from utils.data_utils import random_sample_to_features_random, random_sample_to_features_group, load_jsonl_as_hf_dataset, add_tokenization_to_dataset
from utils.logic_utils import is_subset, is_strictly_exclusive, is_loosely_exclusive

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
    
class ComplexQueryRandomDataset(TorchDataset):
    """
    Dataset class for retrieval dataset. Handles how sampling in training and eval is done.
    """

    def __init__(self,
                 queries: HFDataset,
                 id2doc: dict = None,
                 max_seq_length: int = 512,
                 train_batch_size: int = 32,
                 eval_batch_size: int = 32,
                 num_workers: int = 32,
                 alpha: float = 0,
                 seed: int = 42,
                 num_pos: int = 1,
                 cache_dir: str = "output/cache/",
                 **kwargs) -> None:
        """
        Args:
            queries: A huggingface dataset of queries
            id2doc: A map of document id to a row of huggingface dataset
            batch_strategy: batch strategy to sample training instances
                - "random" - randomly sample instances
            max_seq_length: int, maximum sequence length
            train_batch_size: int, batch size for training
            eval_batch_size: int, batch size for evaluation
            num_workers: int, number of workers to prefetch data
        """
        super().__init__()

        self.queries = queries
        self.id2doc = id2doc

        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers

        # seed used for random sampling training instances
        self.seed = seed

        # Number of positive docs per query
        self.num_pos = num_pos

        # Cache directory
        self.cache_dir = cache_dir

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        """
        Get a single query and multiple positive documents
        """
        
        query = self.queries[idx]
        
        cand_docs = query["documents"]

        if len(cand_docs) > 0:
            num_docs = min(self.num_pos, len(cand_docs))
            pos_doc_indices = random.sample(cand_docs, num_docs)
            # pos_doc_indices = cand_docs[:num_docs]
            pos_docs = [self.id2doc[doc_idx] for doc_idx in pos_doc_indices]
        else:
            pos_docs = []

        return {
            "query": query, 
            "documents": pos_docs
        }

class ComplexQueryMixDataset(TorchDataset):
    """
    Dataset for sampling queries grouped by atomic queries
    """
    def __init__(self,
                 queries: HFDataset,
                 id2doc: dict = None,
                 max_seq_length: int = 512,
                 train_batch_size: int = 32,
                 eval_batch_size: int = 32,
                 num_workers: int = 32,
                 alpha: float = 0.25,
                 num_pos: int = 1, 
                 seed: int = 42,
                 cache_dir: str = "output/cache/",
                 loose_exclusion_loss: int = 0,
                 **kwargs) -> None:

        self.queries = queries
        self.id2doc = id2doc

        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers

        # Ratio and the random strategy and the group strategy
        self.alpha = alpha

        # iterate over huggingface dataset to group queries by atomic queries
        self.queries_dict = self.queries.to_pandas().to_dict(orient='records')

        self.num_pos = num_pos
        # Keep a map of query + operators to queries

        self.max_degree = -1
        self.atom_to_query_map = {}
        self.query_group_keys = []

        self.write_result = True

        self.cache_dir = cache_dir

        for ex in self.queries_dict:
            # Before turning it into tuples, rank the queries
            atoms = tuple(sorted(ex["queries"]))
            operators = tuple(ex["operators"])

            if len(atoms) > self.max_degree:
                self.max_degree = len(atoms)

            if atoms not in self.atom_to_query_map:
                self.atom_to_query_map[atoms] = {
                    operators: ex
                }
            else:
                self.atom_to_query_map[atoms][operators] = ex

        # Create groups of queries w/ (1) queries of the max # atoms, and (2) queries with subset atoms
        # e.g. The group A B C 
        self.query_group = {}

        added_queries = set()
        # First, add all queries with max # of atoms and create groups [A B C]
        for atoms, op_map in self.atom_to_query_map.items():
            if len(atoms) == self.max_degree:
                if atoms not in self.query_group:
                    self.query_group[atoms] = {
                        "queries": []
                    }
                    
                for op, ex in op_map.items():
                    self.query_group[atoms]["queries"].append(ex)
                added_queries.add(atoms)
        
        # Next, add queries with subsets of atoms to the groups
        # Only add single queries to decrease the size of the group [A, B, C] -> [A], [B], [C]
        # Also adding queries with two subqueries will harm the performance [A, B, C] -> [A B], [A C], [B C]
        for max_atoms in self.query_group:
            cur_group = self.query_group[max_atoms]["queries"]
            atom_subsets = chain.from_iterable(combinations(max_atoms, r) for r in range(1, len(max_atoms)))

            for cur_atoms in atom_subsets:
                if cur_atoms in self.atom_to_query_map and len(cur_atoms) == 1:
                    for op, q in self.atom_to_query_map[cur_atoms].items():
                        self.query_group[max_atoms]["queries"].append(q)
                    added_queries.add(cur_atoms)
        
        # Adding complex queries with two subqueries [A, B]
        for atoms, op_map in self.atom_to_query_map.items():
            if len(atoms) == self.max_degree - 1 and atoms not in added_queries:
                if atoms not in self.query_group:
                    self.query_group[atoms] = {
                        "queries": []
                    }
                    
                for op, ex in op_map.items():
                    self.query_group[atoms]["queries"].append(ex)
                
                added_queries.add(atoms)  

        # Adding single queries that is their subset [A, B] -> [A], [B]
        for max_atoms in self.query_group:
            if len(max_atoms) == self.max_degree - 1:
                cur_group = self.query_group[max_atoms]["queries"]
                atom_subsets = chain.from_iterable(combinations(max_atoms, r) for r in range(1, len(max_atoms)))

                for cur_atoms in atom_subsets:
                    if cur_atoms in self.atom_to_query_map and len(cur_atoms) == 1:
                        for op, q in self.atom_to_query_map[cur_atoms].items():
                            self.query_group[max_atoms]["queries"].append(q)
                        added_queries.add(cur_atoms)

        # Also adding other queries that are not in the group [E], [F], [G]
        for atoms, op_map in self.atom_to_query_map.items():
            if atoms not in added_queries and len(atoms) == 1:   
                if atoms not in self.query_group:
                    self.query_group[atoms] = {
                        "queries": []
                    }     
                for op, ex in op_map.items():
                    self.query_group[atoms]["queries"].append(ex)

        # Next, generate constraints for each group
        for max_atoms in self.query_group:
            
            cur_group = self.query_group[max_atoms]["queries"]
            grp_size = len(cur_group)
            subset_indices = []
            exclusion_indices = []
            
            for i in range(len(cur_group) - 1):
                cur_query = cur_group[i]
                
                for j in range(i + 1, len(cur_group)):
                    other_query = cur_group[j]

                    cur_query_atoms = cur_query['queries'].tolist()
                    cur_query_ops = cur_query['operators'].tolist()
                    other_query_atoms = other_query['queries'].tolist()
                    other_query_ops = other_query['operators'].tolist()

                    # Check if i is the subset of j
                    ij_subset = is_subset(
                        superset_atoms=other_query_atoms,
                        superset_ops=other_query_ops,
                        subset_atoms=cur_query_atoms,
                        subset_ops=cur_query_ops
                    )

                    # Check if j is the subset of i
                    ji_subset = is_subset(
                        superset_atoms=cur_query_atoms,
                        superset_ops=cur_query_ops,
                        subset_atoms=other_query_atoms,
                        subset_ops=other_query_ops
                    )

                    if ij_subset:
                        subset_indices.append((i, j))
                    if ji_subset:
                        subset_indices.append((j, i))

                    if loose_exclusion_loss == 1:
                        # print("Using loose exclusion loss")
                        ij_exclusion = is_loosely_exclusive(
                        left_atoms=cur_query_atoms,
                        left_ops=cur_query_ops,
                        right_atoms=other_query_atoms,
                        right_ops=other_query_ops
                        )
                    else:
                        # print("Using strict exclusion loss")
                        ij_exclusion = is_strictly_exclusive(
                            left_atoms=cur_query_atoms,
                            left_ops=cur_query_ops,
                            right_atoms=other_query_atoms,
                            right_ops=other_query_ops
                        )
                    
                    if ij_exclusion:
                        exclusion_indices.append((i, j))
                        exclusion_indices.append((j, i))

            self.query_group[max_atoms]["subset_relation"] = subset_indices
            self.query_group[max_atoms]["exclusion_relation"] = exclusion_indices

        self.query_group_keys = list(self.query_group.keys())

        # Compute the size of groups to determine the size of the single query groups
        group_sizes_total = 0
        group_count = 0
        for max_atoms in self.query_group:
            if len(max_atoms) > 1:
                group_sizes_total += len(self.query_group[max_atoms]["queries"])
                group_count += 1

        # Get all the single queries
        random_groups_keys = []
        random_groups = []
        for max_atoms in self.query_group:
            if len(max_atoms) == 1:
                random_groups_keys.append(max_atoms)
                self.query_group_keys.remove(max_atoms)

        random.shuffle(self.query_group_keys)

        # print("Number of complex queries groups: ", len(self.query_group_keys))

        # For self.alpha portion of the groups (exclude single queries), put them together and create random groups
        for i in range(int(self.alpha * len(self.query_group_keys))):
            max_atoms = self.query_group_keys[i]
            random_groups_keys.append(max_atoms)

        # print("Newly added random groups: ", int(self.alpha * len(self.query_group_keys)))

        for max_atoms in random_groups_keys:
            for query_idx, query in enumerate(self.query_group[max_atoms]["queries"]):
                random_groups.append(query)
                # Erase max_atoms from self.query_group

            del self.query_group[max_atoms]

            if max_atoms in self.query_group_keys:
                self.query_group_keys.remove(max_atoms)
        
        #Random shuffle the random groups
        random.shuffle(random_groups)

        # From query in random_groups into random groups, also add the max_atoms of the first query into query_group as keys and self.query_group_keys

        average_query_group_size = group_sizes_total // group_count

        print("Average query group size: ", average_query_group_size)
        for i in range(0, len(random_groups), average_query_group_size):
            max_atoms = random_groups[i]['queries']
            self.query_group[tuple(max_atoms)] = {
                "queries": random_groups[i:i + average_query_group_size],
                "subset_relation": [],
                "exclusion_relation": [],
            }
            self.query_group_keys.append(tuple(max_atoms))

        # Random shuffle the query_group_keys
        random.shuffle(self.query_group_keys)
        
    def __len__(self):
        return len(self.query_group_keys)
    
    def __getitem__(self, idx):
        """
        Sample a Group of queries instead of a single query
        """

        max_atoms = self.query_group_keys[idx]
        cur_group = self.query_group[max_atoms]

        queries = [q for q in cur_group["queries"]]

        pos_documents = []
        documents = []
        for q in cur_group["queries"]:
            cand_docs = list(q["documents"])

            num_docs = min(self.num_pos, len(cand_docs))
            pos_doc_indices = random.sample(cand_docs, num_docs)
            # pos_doc_indices = cand_docs[:num_docs]
            pos_docs = [self.id2doc[doc_idx] for doc_idx in pos_doc_indices]
                
            pos_documents.append(pos_docs)
            documents.append([self.id2doc[doc_idx] for doc_idx in cand_docs])

        subset_relation = cur_group["subset_relation"]
        exclusion_relation = cur_group["exclusion_relation"]
        
        with open(f'{self.cache_dir}/group_queries_train_data_ComplexQueryMixDataset.jsonl', 'a') as fout:
            for i in range(len(queries)):
                output_dict = {'nl_query': queries[i]['nl_query'], 'documents': pos_documents[i][0]['title']}
                json.dump(output_dict, fout)
                fout.write('\n')
                
        return {
            "queries": queries,  # List of lists of strings (queries)
            "pos_documents": pos_documents,  # List of lists of positive documents for each query
            "documents": documents,  # List of lists of documents for each query
            "subset_relation": subset_relation,  # Subset relation matrix
            "exclusion_relation": exclusion_relation  # Exclusion relation matrix
        }

class ComplexQueryDataModule(LightningDataModule): 
    """
    The actual datamodule that lightning trainer uses for training and evaluation.
    """
    def __init__(self,
                 train_queries_path: str,
                 val_queries_path: str,
                 test_queries_path: str,
                 doc_path: str,
                 model_name_or_path: str,
                 cache_dir: str,
                 batch_strategy: str = "random",
                 max_seq_length: int = 512,
                 train_batch_size: int = 32,
                 eval_batch_size: int = 32,
                 num_workers: int = 32,
                 alpha: float = 0,
                 seed: int = 42,
                 sanity: int = None,
                 query_key: str = "nl_query",
                 doc_key: str = "text",
                 num_pos: int = 1,
                 loose_exclusion_loss: int = 0,
                 **kwargs
                 ):
        super().__init__()
        
        # Load tokenizer
        self.model_name_or_path = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        # Path to dataset
        self.train_queries_path = train_queries_path
        self.val_queries_path = val_queries_path
        self.test_queries_path = test_queries_path
        self.doc_path = doc_path

        # Set the keys for the queries and documents
        self.query_key = query_key
        self.doc_key = doc_key
        
        # Set sampling strategy for the dataloader
        self.batch_strategy = batch_strategy

        # Set up cache directory for storing the tokenized datasets
        self.cache_dir = cache_dir

        # If sanity is not None, only use a subset of the dataset
        self.sanity = sanity

        # Number of positive docs per query
        self.num_pos = num_pos

        # Alpha for the mix strategy
        self.alpha = alpha

        self.loose_exclusion_loss = loose_exclusion_loss

        _base_model_name = get_hf_base_model_name(self.model_name_or_path)
        
        # Set up cache paths for the tokenized datasets
        get_tok_cache_path = partial(
            get_cache_path,
            cache_dir=self.cache_dir,
            base_model_name=_base_model_name,
            cache_type=CacheType.TOKENIZATION
        )

        self.train_cache_file = get_tok_cache_path(file_name=get_file_name(self.train_queries_path))
        self.val_cache_file = get_tok_cache_path(file_name=get_file_name(self.val_queries_path))
        self.test_cache_file = get_tok_cache_path(file_name=get_file_name(self.test_queries_path))
        self.doc_cache_file = get_tok_cache_path(file_name=get_file_name(self.doc_path))
        
        # Hyperparams
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.seed = seed

    def _add_tokenization(
        self, 
        data: List[Dict], 
        key_to_tokenize: str) -> HFDataset:
        """
        Tokenize the queries and documents
        """
        _df = DataFrame(data)
        _ds = HFDataset.from_pandas(_df)

        _ds = add_tokenization_to_dataset(
            dataset=_ds,
            tokenizer=self.tokenizer,
            key_to_tokenize=key_to_tokenize,
            max_seq_length=self.max_seq_length,
            num_workers=self.num_workers
        )
        
        return _ds

    def prepare_data(self) -> None:
        """
        Pre-fetch and tokenize the complex queries and dbpedia text 

        The prepare_data() function is only called on 1 GPU in distributed settings.
        """
    
        # Check if cached datasets exists, if so, skip the data preparation step
        if os.path.exists(self.train_cache_file) \
        and os.path.exists(self.val_cache_file) \
        and os.path.exists(self.test_cache_file) \
        and os.path.exists(self.doc_cache_file):
            logger.info("Tokenized datasets already exist, skipping data preparation.")
            return

        # Load documents
        query_key = self.query_key
        doc_key = self.doc_key

        logger.info("Reading documents from {}".format(self.doc_path))
        doc = read_jsonl(self.doc_path)

        # Load the complex queries by train/val/test split
        logger.info("Reading queries from {}".format(self.train_queries_path))
        train_queries = read_jsonl(self.train_queries_path)
        val_queries = read_jsonl(self.val_queries_path)
        test_queries = read_jsonl(self.test_queries_path)

        # Tokenize the queries and documents
        logger.info("Adding tokenization...")
        train_queries_toks = self._add_tokenization(train_queries, query_key)
        val_queries_toks = self._add_tokenization(val_queries, query_key)
        test_queries_toks = self._add_tokenization(test_queries, query_key)
        doc_toks = self._add_tokenization(doc, doc_key)

        # # Save the tokenized datasets
        train_queries_toks.save_to_disk(self.train_cache_file, max_shard_size="1GB")
        val_queries_toks.save_to_disk(self.val_cache_file, max_shard_size="1GB")
        test_queries_toks.save_to_disk(self.test_cache_file, max_shard_size="1GB")
        doc_toks.save_to_disk(self.doc_cache_file, max_shard_size="1GB")

    def setup(self, stage):

        """
        Load the tokenized data during training.
        
        The setup() function is called on every GPU in distributed settings.
        """
        # Load the tokenized datasets
        logger.info("Loading tokenized documents...")
        self.doc = HFDataset.load_from_disk(self.doc_cache_file)

        id2doc = {}
        for i in range(len(self.doc)):
            cur_doc = self.doc[i] 
            id2doc[cur_doc["idx"]] = cur_doc

        # Create the dataset objects
        _ds_class = ComplexQueryRandomDataset if self.batch_strategy == "random" else ComplexQueryMixDataset
        
        if stage == "fit":
            logger.info("Loading training queries...")
            train_queries = HFDataset.load_from_disk(self.train_cache_file)
            logger.info("Loading validation queries...")
            val_queries = HFDataset.load_from_disk(self.val_cache_file)
            
            if self.sanity is not None:
                train_queries = train_queries.select(range(self.sanity))
            
            self.train_dataset = _ds_class(
                queries=train_queries,
                id2doc=id2doc,
                max_seq_length=self.max_seq_length,
                train_batch_size=self.train_batch_size,
                eval_batch_size=self.eval_batch_size,
                num_workers=self.num_workers,
                alpha=self.alpha,
                seed=self.seed,
                num_pos = self.num_pos,
                cache_dir=self.cache_dir,
                loose_exclusion_loss=self.loose_exclusion_loss
            )
            
            self.validation_dataset = _ds_class(
                queries=val_queries,
                id2doc=id2doc,
                max_seq_length=self.max_seq_length,
                train_batch_size=self.train_batch_size,
                eval_batch_size=self.eval_batch_size,
                num_workers=self.num_workers,
                alpha=self.alpha,
                seed=self.seed,
                num_pos = self.num_pos,
                cache_dir=self.cache_dir,
                loose_exclusion_loss=self.loose_exclusion_loss
            )

        else:
            test_queries = HFDataset.load_from_disk(self.test_cache_file)

            self.test_dataset = _ds_class(
                queries=test_queries,
                documents=id2doc,
                max_seq_length=self.max_seq_length,
                train_batch_size=self.train_batch_size,
                eval_batch_size=self.eval_batch_size,
                num_workers=self.num_workers,
                alpha=self.alpha,
                seed=self.seed,
                cache_dir=self.cache_dir,
                loose_exclusion_loss=self.loose_exclusion_loss
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.train_batch_size,
                          collate_fn=self.convert_to_features,
                          shuffle=True,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset,
                          batch_size=self.eval_batch_size,
                          collate_fn=self.convert_to_features,
                          shuffle=False,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.eval_batch_size,
                          collate_fn=self.convert_to_features,
                          shuffle=False,
                          num_workers=self.num_workers)

    def convert_to_features(self, batch_examples):
        """

        :param batch_examples:
        :return:
        """

        if self.batch_strategy == 'group':
            return random_sample_to_features_group(batch_examples)
        else:
            return random_sample_to_features_random(batch_examples) 


if __name__ == "__main__":
    mod = ComplexQueryDataModule(
        train_queries_path="/shared/yanzhen4/Set-based-Retrieval/data/quest/train_filtered_augment_full/quest_train_filtered_augment_full.jsonl",
        val_queries_path="data/quest/quest_val.jsonl",
        test_queries_path="data/quest/quest_test.jsonl",
        doc_path="/shared/yanzhen4/Set-based-Retrieval/data/quest/train_filtered_augment_full/quest_text_w_id_filtered_augmented_full.jsonl",
        model_name_or_path="sentence-transformers/gtr-t5-base",
        cache_dir="output/cache/",
        batch_strategy="group",
        max_seq_length=512,
        train_batch_size=32,
        eval_batch_size=32,
        num_workers=32,
        seed=42
    )

    mod.prepare_data()
    mod.setup("fit")