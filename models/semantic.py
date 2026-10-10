from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import numpy as np

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.pca = PCA(n_components=3)
        self.embeddings = None
        self.embeddings_3d = None

    def fit(self, raw_documents):
        truncated = [" ".join(doc.split()[:300]) for doc in raw_documents]
        self.embeddings = self.model.encode(truncated, show_progress_bar=False)
        self._update_pca(len(raw_documents))

    def add_document(self, raw_doc):
        new_emb = self.model.encode([" ".join(raw_doc.split()[:300])], show_progress_bar=False)
        self.embeddings = np.vstack([self.embeddings, new_emb])
        self._update_pca(self.embeddings.shape[0])

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
