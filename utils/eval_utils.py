import json
from pandas import DataFrame
from typing import Dict
import os
import torch
import logging
from pytorch_lightning.callbacks import BasePredictionWriter
from utils.file_utils import save_jsonl
from tqdm import tqdm

logger = logging.getLogger(__name__)

class EvalRecord:
    def __init__(self, 
                 round_results_to_digits: int = None,
                 add_example_counts: bool = True):
        self.entries = []
        self.round_digits = round_results_to_digits
        self.add_example_counts = True
        self.result = None

    def add_entry(
        self, 
        metric_name: str,
        example_type: str,
        score) -> None:

        self.entries.append({
            "metric": metric_name,
            "type": example_type,
            "score": score
        })

    
    def add_counts(self):
        # take one of the metric in the record
        df = DataFrame(self.entries)
        metric = list(df["metric"].unique())[0]

        metric_df = df[df.metric == metric]

        self.result["example_count"] = len(metric_df.index)

        self.result["example_count_by_type"] = {}
        for t, tdf in metric_df.groupby("type"):
            self.result["example_count_by_type"][t] = len(tdf)
    
    def compute_score(self) -> Dict[str, float]:
        df = DataFrame(self.entries)

        result_dict = {
            "result": {},
            "result_by_type": {}
        }

        # First report an average over entire dataset
        for metric in df["metric"].unique():
            metric_df = df[df["metric"] == metric]
            avg_score = float(metric_df["score"].mean())

            if self.round_digits is not None:
                avg_score = round(avg_score, self.round_digits)

            result_dict["result"][metric] = avg_score

        # Next, report results based on example type
        df_by_type = df.groupby(["metric", "type"])
        for (metric, example_type), group in df_by_type:
            if example_type not in result_dict["result_by_type"]:
                result_dict["result_by_type"][example_type] = {}

            avg_score = float(group["score"].mean())
            if self.round_digits is not None:
                avg_score = round(avg_score, self.round_digits)
            result_dict["result_by_type"][example_type][metric] = avg_score

        self.result = result_dict

        if self.add_example_counts:
            self.add_counts()

    def save_result(self, path):
        if self.result is None:
            raise ValueError("No result to save. Please call compute_score() first.")

        with open(path, 'w') as fout:
            json.dump(self.result, fout, indent=4)

class DistributedEncoderPredictionWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval: str = "epoch", text_key: str = "text"):
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.text_key = text_key

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices, *args, **kwargs):
        if trainer.is_global_zero:
            logger.info("Saving Predictions...")
        
        # Uncollate the predictions
        prediction_by_row = []

        for batch in predictions:
            true_bs = len(batch[self.text_key])

            uncollated = [{key: batch[key][i] for key in batch} for i in range(true_bs)]
            prediction_by_row += uncollated

        output_path = os.path.join(self.output_dir, "{}.pt".format(trainer.global_rank)) 

        torch.save(prediction_by_row, output_path)

    def load_predictions(self):
        all_preds = []
        for p in os.listdir(self.output_dir):
            p = os.path.join(self.output_dir, p)
            all_preds += torch.load(p)

        return all_preds

def compute_metrics(predicted_docs, gold_docs, k):
    predicted_docs = set(predicted_docs[:k])
    gold_docs = set(gold_docs)

    tp = len(gold_docs.intersection(predicted_docs))
    fp = len(predicted_docs.difference(gold_docs))
    fn = len(gold_docs.difference(predicted_docs))
    m_recall = 1.0 if gold_docs.issubset(predicted_docs) else 0.0
    
    if tp:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
    else:
        precision = 0.0
        recall = 0.0
        f1 = 0.0

    return precision, recall, f1, m_recall

def make_predictions(documents, queries, retriever):
    subquery2doc_sim = {}
    doc_idx = [doc['idx'] for doc in documents]
    predictions = []
    for idx, query in tqdm(enumerate(queries)):
        
        docs_scores = retriever.get_docs_and_scores(query, len(documents))
        doc_to_score = {doc: score for doc, score in docs_scores}
        scores = torch.tensor([-doc_to_score[idx] for idx in doc_idx])
        subquery2doc_sim[query] = scores

        docs = [doc for doc, _ in docs_scores]
        predictions.append(docs)

    return predictions, subquery2doc_sim

def compute_doc_ranks_from_predictions(predictions):
    """
    Converts a list of sorted document indices into a ranking tensor where each 
    value represents the rank of the corresponding document index.
    
    Args:
    predictions (list of int): Sorted list of document indices with the highest index in the front.

    Returns:
    torch.Tensor: A tensor where each index represents a document and the value at that index is its rank.
    """
    if isinstance(predictions[0], str):
        unique_strings = list(set(predictions))
        string_to_index = {string: idx for idx, string in enumerate(unique_strings)}
        predictions_int = [string_to_index[prediction] for prediction in predictions]
    else:
        predictions_int = predictions

    predictions_tensor = torch.tensor(predictions_int, dtype=torch.long)
    ranks = torch.arange(predictions_tensor.size(0), dtype=torch.long)
    rankings_tensor = torch.empty_like(predictions_tensor)
    rankings_tensor.scatter_(0, predictions_tensor, ranks)

    return rankings_tensor

def sim_to_rank(subquery2doc_sim):
    subquery2doc_rank = {}
    for query, sim_scores in tqdm(subquery2doc_sim.items()):
        sorted_scores, indices = torch.sort(sim_scores, descending=True)
        ranks = compute_doc_ranks_from_predictions(indices)
        subquery2doc_rank[query] = ranks

    return subquery2doc_rank

def combine_sim(documents_sim, subquery_sim, operator):
    if operator == 'AND':
        documents_sim = torch.min(documents_sim, subquery_sim)
    elif operator == 'OR':
        documents_sim = torch.max(documents_sim, subquery_sim)
    elif operator == 'NOT':
        documents_sim = documents_sim - subquery_sim
    return documents_sim

def combine_ranks(documents_rank, subquery_rank, operator):
    if operator == 'AND':
        documents_rank = torch.max(documents_rank, subquery_rank)
    elif operator == 'OR':
        documents_rank = torch.min(documents_rank, subquery_rank)
    elif operator == 'NOT':
        documents_rank = documents_rank + subquery_rank
    return documents_rank

def combine_queries(subqueries_score, operators, combine_function):
    documents_score = subqueries_score[0]
    
    for subquery_score, operator in zip(subqueries_score[1:], operators):
        documents_score = combine_function(documents_score, subquery_score, operator)

    return documents_score

def combine_on_sim(subquery2doc_sim, query_data):
    all_sim_new = []
    nl_queries = []
    
    for data in tqdm(query_data):
        queries = data['queries']
        nl_query = data['nl_query']
        operators = data['operators']

        subqueries_sim = [subquery2doc_sim[query] for query in queries]

        documents_sim = combine_queries(subqueries_sim, operators, combine_sim)

        all_sim_new.append(documents_sim)
        nl_queries.append(nl_query)

    all_sim_new = torch.stack(all_sim_new)
    return all_sim_new, nl_queries

def combine_on_ranks(subquery2doc_sim, query_data):
    subquery2doc_rank = sim_to_rank(subquery2doc_sim)
    all_rank_new = []
    nl_queries = []

    for data in tqdm(query_data):
        print(query_data)
        queries = data['queries']
        nl_query = data['nl_query']
        operators = data['operators']

        subqueries_rank = [subquery2doc_rank[query] for query in queries]

        documents_rank = combine_queries(subqueries_rank, operators, combine_ranks)
        documents_rank = -documents_rank

        all_rank_new.append(documents_rank)
        nl_queries.append(nl_query)

    all_sim_new = torch.stack(all_rank_new)
    return all_sim_new, nl_queries

