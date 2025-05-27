import argparse
from utils.eval_utils import compute_metrics
from dataclasses import dataclass
from datasets import Dataset as HFDataset
from functools import partial
import logging
import numpy as np

from model.encoder import LitEncoder
import os
import json
from pandas import DataFrame
import pytorch_lightning as pl
from pytorch_lightning.callbacks import BasePredictionWriter
from transformers import AutoTokenizer
from typing import List, Dict
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from utils.file_utils import save_jsonl, get_file_name, get_hf_base_model_name, get_cache_path, get_ckpt_model_name, CacheType, load_idx2title, save_dict, save_query2doc_result
from utils.data_utils import load_jsonl_as_hf_dataset, add_tokenization_to_dataset
from utils.eval_utils import EvalRecord, DistributedEncoderPredictionWriter, compute_metrics, compute_doc_ranks_from_predictions

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def encoder_input_collate_fn(examples):
    """
    Collate function for queries and documents.
    """
    batch = {}

    for ex in examples:
        for key, val in ex.items():
            if key not in batch:
                batch[key] = []
            batch[key].append(val)
    
    if "input_ids" in batch:
        batch["input_ids"] = torch.tensor(batch["input_ids"], dtype=torch.long)
    
    if "attention_mask" in batch:
        batch["attention_mask"] = torch.tensor(batch["attention_mask"], dtype=torch.long)

    return batch

def main(args):
    # Load model and trainer for distributed inference
    precision = int(args.precision) if args.precision != "bf16" else "bf16"

    if args.model_name.endswith('.ckpt'):
        model = LitEncoder.load_from_checkpoint(args.model_name)
        hf_model_name = model.params.model_name
        test_model_name = get_ckpt_model_name(args.model_name)
    else:
        model = LitEncoder(args) # will load the huggingface model weights specified by args.model_name
        hf_model_name = args.model_name
        test_model_name = get_hf_base_model_name(hf_model_name)

    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    hf_model_name = get_hf_base_model_name(hf_model_name)

    # Load cached tokenized document text, or tokenize document text (stored at the hf model level)
    doc_cache_path = get_cache_path(
        cache_dir=args.cache_dir,
        base_model_name=hf_model_name,
        file_name=get_file_name(args.document_path),
        cache_type=CacheType.TOKENIZATION
    )
    
    if os.path.exists(doc_cache_path):
        logger.info("Loading tokenized documents from cache {}".format(doc_cache_path))
        doc_ds = HFDataset.load_from_disk(doc_cache_path)
    else:
        # tokenize and cache the documents
        logger.info("Tokenizing documents and saving to cache {}".format(doc_cache_path))
        doc_ds = load_jsonl_as_hf_dataset(args.document_path)
        doc_ds = add_tokenization_to_dataset(
            dataset=doc_ds,
            tokenizer=tokenizer, 
            key_to_tokenize=args.document_text_key, 
            max_seq_length=args.max_seq_length,
            num_workers=args.num_workers
        )

        doc_ds.save_to_disk(doc_cache_path)

    # Define save callback for writing predicted results to disk
    # https://lightning.ai/docs/pytorch/stable/deploy/production_basic.html
    doc_embedding_cache_dir = get_cache_path(
        cache_dir=args.cache_dir,
        base_model_name=test_model_name,
        file_name=get_file_name(args.document_path),
        cache_type=CacheType.EMBEDDINGS
    )

    if not os.path.exists(doc_embedding_cache_dir):
        os.makedirs(doc_embedding_cache_dir, exist_ok=True)

    doc_writer_callback = DistributedEncoderPredictionWriter(
        output_dir=doc_embedding_cache_dir,
        text_key=args.document_text_key
    )

    if args.load_cached_embeddings and len(os.listdir(doc_embedding_cache_dir)) > 0:
        logger.info("Found existing document embeddings in cache. Skipping encoding...")

    else:    
        # Prepare data loader and trainer object for inference
        logger.info("Encoding documents with distributed inference. Num GPUs = {}".format(args.gpus))
        doc_loader = DataLoader(
            doc_ds,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=encoder_input_collate_fn
        )

        if args.gpus > 0:
            doc_trainer = pl.Trainer(
                accelerator='gpu',
                devices=args.gpus,
                strategy='ddp',
                precision=precision,
                callbacks=[doc_writer_callback]
            )
        else:
            doc_trainer = pl.Trainer(callbacks=[doc_writer_callback])

        # Encode documents
        doc_trainer.predict(model, doc_loader)

        doc_trainer.strategy.barrier()

    # Load and encode queries
    query_cache_path = get_cache_path(
        cache_dir=args.cache_dir,
        base_model_name=hf_model_name,
        file_name=get_file_name(args.query_path),
        cache_type=CacheType.TOKENIZATION
    )

    if os.path.exists(query_cache_path):
        logger.info("Loading tokenized queries from cache {}".format(query_cache_path))
        query_ds = HFDataset.load_from_disk(query_cache_path)
    else:
        # tokenize and cache the documents
        logger.info("Tokenizing queries and saving to cache {}".format(query_cache_path))
        query_ds = load_jsonl_as_hf_dataset(args.query_path)

        query_ds = query_ds.map(
            lambda x: tokenizer(
                x[args.query_text_key], 
                return_tensors="pt",
                max_length=args.max_seq_length,
                padding='max_length',
                truncation=True), 
            batched=True,
            num_proc=args.num_workers
        )

        query_ds.save_to_disk(query_cache_path)

    # Define save callback for writing predicted results to disk
    # https://lightning.ai/docs/pytorch/stable/deploy/production_basic.html
    query_embedding_cache_dir = get_cache_path(
        cache_dir=args.cache_dir,
        base_model_name=test_model_name,
        file_name=get_file_name(args.query_path),
        cache_type=CacheType.EMBEDDINGS
    )

    if not os.path.exists(query_embedding_cache_dir):
        os.makedirs(query_embedding_cache_dir, exist_ok=True)

    query_writer_callback = DistributedEncoderPredictionWriter(
        output_dir=query_embedding_cache_dir,
        text_key=args.query_text_key
    )

    if args.load_cached_embeddings and len(os.listdir(query_embedding_cache_dir)) > 0:
        logger.info("Found existing query embeddings in cache. Skipping encoding...")
    else:
        # Prepare data loader and trainer object for inference
        logger.info("Encoding queries with distributed inference. Num GPUs = {}".format(args.gpus))
        query_loader = DataLoader(
            query_ds,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=encoder_input_collate_fn
        )

        if args.gpus > 0:
            query_trainer = pl.Trainer(
                accelerator='gpu',
                devices=args.gpus,
                strategy='ddp',
                precision=precision,
                callbacks=[query_writer_callback]
            )
        else:
            query_trainer = pl.Trainer(callbacks=[query_writer_callback])

        # Encode documents
        query_trainer.predict(model, query_loader)

        query_trainer.strategy.barrier()
        # Terminate the distributed child processes
        if query_trainer.global_rank > 0:
            return
    
    doc_idx2title = load_idx2title(args.document_path)

    # Load the query and document embeddings and collate them respectively
    all_docs = doc_writer_callback.load_predictions()
    all_doc_embeds = torch.stack([d['encoded'] for d in all_docs], dim=0)  # (num_docs, emb_dim)

    # Store document embedding
    if args.doc_embedding_output_path:
        if not os.path.exists(args.doc_embedding_output_path):
            os.makedirs(args.doc_embedding_output_path, exist_ok=True)

        doc_embedding_output_path = os.path.join(args.doc_embedding_output_path, f"{get_file_name(args.document_path)}_{test_model_name}.jsonl")

        if os.path.exists(doc_embedding_output_path):
            logger.info(f"Document embeddings already exist at {doc_embedding_output_path}.")
        else:
            logger.info(f"Writing document embeddings to {doc_embedding_output_path}")
        
            outputs = []
            for d in all_docs:
                output_dict = {'title': d['title'], 'idx': d['idx'], 'encoded': d['encoded'].tolist()}
                outputs.append(output_dict)
            save_jsonl(outputs, doc_embedding_output_path)
            
    # Get the mapping between document idx and title
    all_doc_ids_to_pos = {d['idx']: i for i, d in enumerate(all_docs)}
    all_doc_ids = torch.tensor([int(d['idx'].strip('quest_')) for d in all_docs])
    
    all_queries = query_writer_callback.load_predictions()
    all_query_embeds = torch.stack([q['encoded'] for q in all_queries], dim=0) # (num_q, emb_dim)

    # Store query embedding
    if args.query_embedding_output_path:

        if not os.path.exists(args.query_embedding_output_path):
            os.makedirs(args.query_embedding_output_path, exist_ok=True)
            
        query_embedding_output_path = os.path.join(args.query_embedding_output_path, f"{get_file_name(args.query_path)}_{test_model_name}.jsonl")

        if os.path.exists(query_embedding_output_path):
            logger.info(f"Query embeddings already exist at {query_embedding_output_path}.")
        else:
            logger.info(f"Writing query embeddings to {query_embedding_output_path}")

            outputs = []
            for q in all_queries:
                print(q.keys())
                output_dict = {'nl_query': q['nl_query'], 'encoded': q['encoded'].tolist()}
                outputs.append(output_dict)
            save_jsonl(outputs, query_embedding_output_path)

    # Compute the dot-product similarity matrix w/ broadcasting for matmul
    # https://pytorch.org/docs/stable/generated/torch.matmul.html
    # Here the encoded embeddings are already normalized 
    logger.info("Computing similarity matrix by chunk...")
    results = EvalRecord(round_results_to_digits=4)

    all_docs_embeds = all_doc_embeds.T # (emb_dim, num_docs)
    all_query_embeds = all_query_embeds.unsqueeze(1) # (num_q, 1, emb_dim)
 
    if args.gpus > 0:
        all_docs_embeds = all_docs_embeds.cuda()
        all_query_embeds = all_query_embeds.cuda()

    # Partition and compute the similarity matrix.
    # (As the similarity matrix might be too large to fit in GPU memory)
    partition_size = args.partition_size
    
    if all_query_embeds.size(0) < partition_size:
        num_chunks = 1
    else:
        num_chunks = all_query_embeds.size(0) // partition_size
    all_queries_chunked = torch.chunk(all_query_embeds, num_chunks, dim=0)

    if args.result_output_path:
        if not os.path.exists(args.result_output_path):
            os.makedirs(args.result_output_path, exist_ok=True)
            
        result_output_path = os.path.join(args.result_output_path, f"{get_file_name(args.query_path)}_{test_model_name}.jsonl")

        with open(result_output_path, 'w') as f:
            f.write('')

    q_idx_offset = 0
    with torch.no_grad():
        for query_chunk in tqdm(all_queries_chunked):
            cur_sims = torch.matmul(query_chunk, all_docs_embeds) # (num_q_chunk, 1, num_docs)
            cur_sims = cur_sims.squeeze(1)

            cur_ranks = torch.argsort(cur_sims, dim=1, descending=True)  # (num_q_chunk, num_docs)

            if args.gpus > 0:
                cur_ranks = cur_ranks.cpu()

            for local_idx, cur_rank in enumerate(tqdm(cur_ranks)):
                
                cur_sim = cur_sims[local_idx]
                rank_doc_ids = cur_rank.tolist()
                # rank_doc_ids = [all_doc_ids[r] for r in cur_rank]
                # rank_doc_ids = all_doc_ids.index_select(0, cur_rank).tolist()

                q_idx = q_idx_offset + local_idx
                query_type = "_".join(all_queries[q_idx]["operators"])

                # nl_query = all_queries[q_idx]['nl_query']

                gold_docs = [
                    all_doc_ids_to_pos[did] 
                    for did in all_queries[q_idx]["documents"]
                ]
                for k in args.recall_at_k:
                    p, r, f1, mrecall = compute_metrics(
                        predicted_docs=rank_doc_ids,
                        gold_docs=gold_docs,
                        k=k
                    )

                    results.add_entry(metric_name=f"precision@{k}", example_type=query_type, score=p)
                    results.add_entry(metric_name=f"recall@{k}", example_type=query_type, score=r)
                    results.add_entry(metric_name=f"f1@{k}", example_type=query_type, score=f1)
                    results.add_entry(metric_name=f"mrecall@{k}", example_type=query_type, score=mrecall)
                
            q_idx_offset += query_chunk.size(0)
    
    logger.info("Computing evaluation metrics...")
    # compute the avg results
    results.compute_score()
    test_file_name = get_file_name(args.query_path)
    result_path = os.path.join(args.result_dir, f"{test_file_name}_{test_model_name}.json")

    logger.info("Writing results to {}".format(result_path))
    results.save_result(result_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        help="huggingface model name or trained model checkpoint (ends with .ckpt)",
        type=str,
        default='sentence-transformers/gtr-t5-base'
    )
    parser.add_argument(
        "--recall_at_k",
        help="Evaluate recall at a list of k values",
        nargs="+",
        type=int,
        default=[1, 5, 20, 50, 100, 1000]
    )
    parser.add_argument(
        "--document_path",
        help="Path to document jsonl file. Each document is a dictionary.",
        type=str,
        default='/shared/yanzhen4/Quest_data'
    )
    parser.add_argument(
        "--document_text_key",
        help="Key of the text of the document",
        type=str,
        default='text'
    )
    parser.add_argument(
        "--query_path",
        help="Path to query jsonl file. Each query is a dictionary.",
        type=str,
        default='/shared/yanzhen4/Quest_data'
    )
    parser.add_argument(
        "--query_text_key",
        help="Path to query jsonl file. Each document is a dictionary.",
        type=str,
        default='/shared/yanzhen4/Quest_data'
    )
    parser.add_argument(
        "--gpus",
        help="Number of GPUs to use for inference. if <=0, do cpu inference.",
        type=int,
        default=0
    )
    parser.add_argument(
        "--cache_dir",
        help="Folder to cache the intermediate outputs (e.g. embeddings)",
        type=str,
        default='/shared/sihaoc/project/set_ir/Set-based-Retrieval/output/cache'
    )
    parser.add_argument(
        "--result_dir",
        help="Folder to write the prediction results",
        type=str,
        default='/shared/sihaoc/project/set_ir/Set-based-Retrieval/output/evaluation'
    )
    parser.add_argument(
        "--batch_size",
        help="Batch size for encoding",
        default=32,
        type=int
    )
    parser.add_argument(
        "--max_seq_length",
        help="Max sequence length during tokenization",
        default=512,
        type=int
    )
    parser.add_argument(
        "--num_workers",
        help="Number of workers for tokenization",
        default=32,
        type=int
    )
    parser.add_argument(
        "--precision",
        help="Number of workers for tokenization",
        default="16",
        type=str
    )
    parser.add_argument(
        "--partition_size",
        help="Number of workers for tokenization",
        default=500,
        type=int
    )
    parser.add_argument(
        "--load_cached_embeddings",
        help="if True, load cached embeddings from cache_dir",
        action="store_true"
    )
    parser.add_argument(
        "--query_embedding_output_path",
        help="Output file for document embeddings", 
        default=None
    )
    parser.add_argument(
        "--doc_embedding_output_path",
        help="Output file for document embeddings", 
        default=None
    )
    parser.add_argument(
        "--result_output_path",
        help="Output file for document embeddings", 
        default=None
    )
    
    args = parser.parse_args()
    
    main(args)
