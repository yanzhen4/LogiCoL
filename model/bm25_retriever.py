import nltk
import numpy as np
from tqdm import tqdm
import gensim.summarization.bm25
import pickle

class BM25Retriever(object):
    def __init__(self, documents):
        self.documents = documents
        self.idx_map = {}
        self.bm25 = self._init_bm25()

    def _init_bm25(self):
        samples_for_retrieval_tokenized = []
        for idx, document in tqdm(enumerate(self.documents)):
            tokenized_example = nltk.tokenize.word_tokenize(document['text'])
            samples_for_retrieval_tokenized.append(tokenized_example)
            self.idx_map[document['title']] = idx
        return gensim.summarization.bm25.BM25(samples_for_retrieval_tokenized)

    def _compute_scores(self, query):
        tokenized_query = nltk.tokenize.word_tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        scores = []
        for idx in range(len(self.documents)):
            scores.append(-bm25_scores[idx])
        return np.array(scores)

    def get_docs_and_scores(self, query, topk):
        scores = self._compute_scores(query)
        sorted_docs_ids = np.argsort(scores)
        topk_doc_ids = sorted_docs_ids[:topk]
        return [(self.documents[idx]['idx'], scores[idx]) for idx in topk_doc_ids]
    
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    def load(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
