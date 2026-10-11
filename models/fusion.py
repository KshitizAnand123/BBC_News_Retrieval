import numpy as np

class RankFusion:
    @staticmethod
    def rrf(bm25_indices, sem_indices, total_docs, top_n=10, k=60, alpha=0.5):
        """
        Reciprocal Rank Fusion with optional tunable weight alpha:
        alpha = 0.5: Equal 50/50 balance between BM25 and Semantic.
        alpha < 0.5: Emphasizes lexical BM25 matching.
        alpha > 0.5: Emphasizes dense semantic conceptual matching.
        Formula:
          Score(d) = (2 * (1 - alpha) / (k + rank_bm25 + 1)) + (2 * alpha / (k + rank_semantic + 1))
        Maximum theoretical score at rank 1 for both is 2 / (k + 1) = 2 / 61 = 0.032787.
        """
        scores = np.zeros(total_docs)
        w_bm25 = 2.0 * (1.0 - alpha)
        w_sem = 2.0 * alpha
        
        for rank, doc_idx in enumerate(bm25_indices):
            scores[doc_idx] += w_bm25 / (k + rank + 1)
        for rank, doc_idx in enumerate(sem_indices):
            scores[doc_idx] += w_sem / (k + rank + 1)
        
        top_idx = np.argsort(scores)[::-1][:top_n]
        return top_idx, [scores[i] for i in top_idx]
