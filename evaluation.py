import numpy as np

class IREvaluator:
    @staticmethod
    def evaluate_metrics(retrieved_indices, relevant_category, categories):
        """
        Calculates Precision@10, Recall, MAP@K, and nDCG@K mathematically bounded by the retrieval window.
        """
        K = len(retrieved_indices)
        if K == 0:
            return {"Precision@10": 0.0, "Recall": 0.0, "MAP": 0.0, "nDCG": 0.0}

        y_true = [1 if categories[idx] == relevant_category else 0 for idx in retrieved_indices]
        
        # 1. Precision @ 10
        p_at_10 = sum(y_true[:10]) / 10.0 if K >= 10 else sum(y_true) / K
        
        # 2. Recall @ K
        total_relevant = categories.count(relevant_category)
        recall = sum(y_true) / total_relevant if total_relevant > 0 else 0
        
        # 3. Mean Average Precision (MAP @ K)
        hits = 0
        sum_precs = 0
        for i, p in enumerate(y_true):
            if p == 1:
                hits += 1
                sum_precs += hits / (i + 1.0)
        
        # THE FIX: Denominator must be bounded by K (the retrieval window). 
        # If we only retrieve 10 items, the max possible hits is 10, not 400.
        denominator = min(K, total_relevant)
        map_score = sum_precs / denominator if denominator > 0 else 0
        
        # 4. nDCG @ K
        dcg = sum([rel / np.log2(idx + 2) for idx, rel in enumerate(y_true)])
        
        # Ideal DCG: perfect ranking fills the top slots with 1s
        ideal_hits = min(K, total_relevant)
        ideal_y = [1] * ideal_hits + [0] * (K - ideal_hits)
        idcg = sum([rel / np.log2(idx + 2) for idx, rel in enumerate(ideal_y)])
        
        ndcg = dcg / idcg if idcg > 0 else 0
        
        return {"Precision@10": p_at_10, "Recall": recall, "MAP": map_score, "nDCG": ndcg}
