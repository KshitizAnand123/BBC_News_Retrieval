from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class LexicalSearch:
    def __init__(self):
        self.tfidf = TfidfVectorizer()
        self.bm25 = None
        self.tfidf_matrix = None

    def fit(self, documents):
        self.tfidf_matrix = self.tfidf.fit_transform(documents)
        tokenized_corpus = [doc.split(" ") for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search_tfidf(self, query_processed, top_n=10):
        q_vec = self.tfidf.transform([query_processed])
        scores = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
        return self._get_top_n(scores, top_n)

    def search_bm25(self, query_processed, top_n=10):
        scores = self.bm25.get_scores(query_processed.split(" "))
        return self._get_top_n(scores, top_n)

    def _get_top_n(self, scores, top_n):
        idx = np.argsort(scores)[::-1][:top_n]
        return idx, [scores[i] for i in idx]
