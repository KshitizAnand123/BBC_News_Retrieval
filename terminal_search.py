import pandas as pd
import time
import sys
from core_ir import BBCNewsIRSystem

def main():
    print("=============================================")
    print("   BBC NEWS IR ENGINE - TERMINAL BACKEND     ")
    print("=============================================")
    
    # Load dataset
    print("\n[1] Loading BBC Dataset from data/bbc_news.csv...")
    try:
        df = pd.read_csv("data/bbc_news.csv")
        print(f"    -> Successfully loaded {len(df)} authentic articles.")
    except Exception as e:
        print(f"    -> ERROR: Dataset not found. {e}")
        sys.exit(1)

    # Initialize and index
    print("\n[2] Indexing System (TF-IDF, BM25, Sentence Transformers)...")
    system = BBCNewsIRSystem()
    start = time.time()
    system.fit(df)
    print(f"    -> Indexing Complete in {time.time() - start:.2f} seconds.")

    # Train ML Model
    print("\n[3] Training ML Category Classifier on TF-IDF Vectors...")
    acc = system.train_classifier()
    print(f"    -> Machine Learning Training Complete! Validation Accuracy: {acc*100:.2f}%")

    print("\n=============================================")
    print(" SYSTEM READY. TYPE 'exit' TO QUIT.")
    print("=============================================")

    # Interactive CLI Loop
    while True:
        query = input("\n🔍 Enter search query: ")
        if query.lower() in ['exit', 'quit']:
            print("Exiting system. Goodbye!")
            break
            
        if not query.strip():
            continue

        print("\n--- HYBRID RRF SEARCH RESULTS ---")
        start_search = time.time()
        
        # Expand Query
        expanded = system.expand_query(query)
        print(f"🧠 AI Query Expansion: {expanded}")
        
        # Search
        indices, scores = system.search_hybrid_rrf(query, top_n=3)
        search_time = time.time() - start_search
        
        print(f"⏱️  Retrieval time: {search_time:.4f} seconds")
        print("---------------------------------------------")
        
        for rank, (idx, score) in enumerate(zip(indices, scores), 1):
            category = system.categories[idx].upper()
            raw_text = system.raw_documents[idx]
            snippet = system.query_focused_summary(raw_text, query, top_sentences=2)
            
            print(f"#{rank} | Category: [{category}] | Fusion Score: {score:.4f}")
            print(f"📝 Snippet: {snippet}")
            print("-" * 45)

if __name__ == "__main__":
    main()
