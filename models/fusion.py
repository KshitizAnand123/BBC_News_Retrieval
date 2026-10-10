import numpy as np

class RankFusion:
    @staticmethod
    def rrf(bm25_indices, sem_indices, total_docs, top_n=10, k=60):
        scores = np.zeros(total_docs)
        for rank, doc_idx in enumerate(bm25_indices):
            scores[doc_idx] += 1.0 / (k + rank + 1)
        for rank, doc_idx in enumerate(sem_indices):
            scores[doc_idx] += 1.0 / (k + rank + 1)
        
        top_idx = np.argsort(scores)[::-1][:top_n]
        return top_idx, [scores[i] for i in top_idx]
