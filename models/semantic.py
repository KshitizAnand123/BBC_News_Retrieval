import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import numpy as np

class SemanticSearch:
    def __init__(self, cache_file=None):
        if cache_file is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            cache_file = os.path.join(base_dir, "data", "embeddings_cache.npy")
        self.cache_file = cache_file
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.pca = PCA(n_components=3)
        self.embeddings = None
        self.embeddings_3d = None

    def fit(self, raw_documents):
        num_docs = len(raw_documents)
        # Check if precomputed cache exists
        if self.cache_file and os.path.exists(self.cache_file):
            try:
                cached = np.load(self.cache_file)
                if len(cached) == num_docs:
                    self.embeddings = cached
                    self._update_pca(num_docs)
                    return
            except Exception:
                pass
        
        truncated = [" ".join(doc.split()[:300]) for doc in raw_documents]
        self.embeddings = self.model.encode(truncated, show_progress_bar=False, batch_size=64)
        
        # Save cache for instant subsequent loading
        if self.cache_file:
            try:
                np.save(self.cache_file, self.embeddings)
            except Exception:
                pass
                
        self._update_pca(num_docs)

    def add_document(self, raw_doc, save_to_disk=False):
        new_emb = self.model.encode([" ".join(raw_doc.split()[:300])], show_progress_bar=False)
        self.embeddings = np.vstack([self.embeddings, new_emb])
        self._update_pca(self.embeddings.shape[0])
        
        # Persist updated embeddings array to disk if requested
        if save_to_disk and self.cache_file:
            try:
                np.save(self.cache_file, self.embeddings)
            except Exception:
                pass

    def _update_pca(self, num_docs):
        if num_docs >= 3:
            self.embeddings_3d = self.pca.fit_transform(self.embeddings)
        else:
            self.embeddings_3d = np.zeros((num_docs, 3))

    def search(self, query, top_n=10):
        q_vec = self.model.encode([query])
        scores = cosine_similarity(q_vec, self.embeddings).flatten()
        idx = np.argsort(scores)[::-1][:top_n]
        return idx, [scores[i] for i in idx]
