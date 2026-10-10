import numpy as np

# Import from modular files
from nlp_utils import NLPProcessor
from evaluation import IREvaluator
from models.lexical import LexicalSearch
from models.semantic import SemanticSearch
from models.classifier import NewsClassifier
from models.fusion import RankFusion

class BBCNewsIRSystem:
    def __init__(self):
        self.nlp = NLPProcessor()
        self.evaluator = IREvaluator()
        
        # Sub-systems
        self.lexical = LexicalSearch()
        self.semantic = SemanticSearch()
        self.classifier = NewsClassifier()
        
        # Data State
        self.documents = []
        self.categories = []
        self.raw_documents = []

    # Properties exposed for app.py
    @property
    def embeddings_3d(self): return self.semantic.embeddings_3d
    
    @property
    def is_trained(self): return self.classifier.is_trained
    
    @property
    def semantic_model(self): return self.semantic.model
    
    @property
    def pca_model(self): return self.semantic.pca

    # Delegated Utilities
    def expand_query(self, query): return self.nlp.expand_query(query)
    def query_focused_summary(self, text, query, top_sentences=2): return self.nlp.query_focused_summary(text, query, top_sentences)
    def evaluate_metrics(self, retrieved_indices, relevant_category): return self.evaluator.evaluate_metrics(retrieved_indices, relevant_category, self.categories)

    # Core Pipeline
    def fit(self, df, text_col='text', cat_col='category'):
        self.raw_documents = df[text_col].tolist()
        self.categories = df[cat_col].tolist() if cat_col in df.columns else ['unknown'] * len(self.raw_documents)
        
        print("Preprocessing documents...")
        self.documents = [self.nlp.preprocess(doc) for doc in self.raw_documents]
        
        print("Fitting Lexical Models...")
        self.lexical.fit(self.documents)
        
        print("Fitting Semantic Models...")
        self.semantic.fit(self.raw_documents)
        print("Indexing Complete!")

    def train_classifier(self):
        return self.classifier.train(self.lexical.tfidf_matrix, self.categories)

    def add_live_article(self, text):
        processed = self.nlp.preprocess(text)
        pred_cat = self.classifier.predict(self.lexical.tfidf, processed)
        
        self.raw_documents.append(text)
        self.categories.append(pred_cat)
        self.documents.append(processed)
        
        # Re-index
        self.lexical.fit(self.documents)
        self.semantic.add_document(text)
        return pred_cat

    # Search Interfaces
    def search_tfidf(self, query, top_n=10):
        return self.lexical.search_tfidf(self.nlp.preprocess(query), top_n)

    def search_bm25(self, query, top_n=10):
        return self.lexical.search_bm25(self.nlp.preprocess(query), top_n)

    def search_semantic(self, query, top_n=10):
        return self.semantic.search(query, top_n)

    def search_hybrid_rrf(self, query, top_n=10, k=60):
        total_docs = len(self.documents)
        bm25_idx, _ = self.search_bm25(query, total_docs)
        sem_idx, _ = self.search_semantic(query, total_docs)
        return RankFusion.rrf(bm25_idx, sem_idx, total_docs, top_n, k)
