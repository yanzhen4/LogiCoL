import argparse
import json
from tqdm import tqdm
import numpy as np
from utils.file_utils import read_jsonl, save_jsonl, create_directory, get_file_name, save_dict, save_matrix
from model.bm25_retriever import BM25Retriever
from utils.eval_utils import EvalRecord, compute_metrics, make_predictions,  compute_doc_ranks_from_predictions
import os
import logging
import random
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_queries(file_path):
    query_data = read_jsonl(file_path)
    nl_queries = [data['nl_query'] for data in query_data]
    return query_data, nl_queries

def sample_documents(documents, sample_fraction=0.01):
    sampled_documents = random.sample(documents, int(len(documents) * sample_fraction))
    return sampled_documents

def main(args):
    documents_folder = args.document_path
    test_data_folder = args.query_path
    output_folder = args.result_dir
    test_file_name = get_file_name(test_data_folder)

    documents = read_jsonl(documents_folder)

    query_data, nl_queries = load_queries(test_data_folder)

    retriever_path = os.path.join(output_folder, 'bm25_retriever.pkl')

    if os.path.exists(retriever_path):
        logger.info("BM25Retriever loaded from file")
        retriever = BM25Retriever.load(retriever_path)
    else:
        logger.info("BM25Retriever trained and saved to file")
        retriever = BM25Retriever(documents)
        retriever.save(retriever_path)

    predictions, subquery2doc_sim = make_predictions(documents, nl_queries, retriever)
    
    # save_dict(subquery2doc_sim, f'{output_folder}/{test_file_name}_nl_query2doc_sim.jsonl')

    reordered_all_ranks = torch.stack([compute_doc_ranks_from_predictions(row) for row in predictions])

    # save_matrix(reordered_all_ranks, nl_queries, f'{output_folder}/{test_file_name}_nl_query2doc_rank.jsonl')

    results = EvalRecord(round_results_to_digits=4)

    for q_idx, prediction in enumerate(tqdm(predictions)):

        query_type = "_".join(query_data[q_idx]["operators"])

        for k in args.recall_at_k:

            p, r, f1, mrecall = compute_metrics(
                predicted_docs=prediction,
                gold_docs=query_data[q_idx]["documents"],
                k=k
            )

            results.add_entry(metric_name=f"precision@{k}", example_type=query_type, score=p)
            results.add_entry(metric_name=f"recall@{k}", example_type=query_type, score=r)
            results.add_entry(metric_name=f"f1@{k}", example_type=query_type, score=f1)
            results.add_entry(metric_name=f"mrecall@{k}", example_type=query_type, score=mrecall)

    results.compute_score()
    result_path = os.path.join(output_folder, f"{test_file_name}_results.json")

    logger.info("Writing results to {}".format(result_path))
    results.save_result(result_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--document_path', type=str, required=True, help='Path of the document file')
    parser.add_argument('--query_path', type=str, required=True, help='Path of the query file')
    parser.add_argument('--result_dir', type=str, required=True, help='Directory to save the results')
    parser.add_argument('--recall_at_k', nargs='+', type=int, default=[1, 5, 20, 50, 100, 1000], help='Evaluate recall at a list of k values')
    args = parser.parse_args()

    main(args)
