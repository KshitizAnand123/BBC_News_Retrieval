# Project 12: BBC News Retrieval System & Next-Gen IR Pipeline

## 📖 Project Overview
This project is an advanced Information Retrieval (IR) system designed for the **BBC News Category dataset**. It satisfies all standard academic requirements for an IR pipeline (preprocessing, inverted indexing, lexical/semantic ranking, and hybrid fusion) while introducing three major production-grade innovations to solve real-world search problems.

## 🚀 The 3 Major Innovations (Report Highlights)
To make this project stand out from standard TF-IDF/BM25 implementations, we integrated the following state-of-the-art features:

1. **Query-Focused Extractive Summarization (AI Snippets)**
   * **Problem:** Users face information overload when clicking through articles.
   * **Solution:** Instead of displaying the first paragraph of an article, the system tokenizes retrieved articles into sentences, scores them against the query using intersection algorithms, and extracts the most relevant sentences. This functions exactly like Google's "Featured Snippets".
2. **Interactive 3D Semantic Knowledge Mapping (Visual XAI)**
   * **Problem:** Neural embeddings (Sentence Transformers) act as a "black box," making it hard to explain *why* an article was retrieved.
   * **Solution:** We implemented a dimensionality reduction pipeline using Principal Component Analysis (PCA) to compress 384-dimensional embeddings into 3D space. The UI features a `🌌 Semantic Universe` tab that visually plots the entire BBC dataset and dynamically drops the user's query into the 3D space, physically demonstrating semantic proximity.
3. **Automated Query Expansion (Solving Vocabulary Mismatch)**
   * **Problem:** Lexical models (BM25) fail when a query uses synonyms (e.g., searching "automobile" when the text says "car").
   * **Solution:** We integrated the NLTK WordNet lexical database to automatically fetch semantic synonyms for the query behind the scenes, injecting them into the search to vastly improve recall.

## ⚙️ Core Architecture & Requirements Satisfied
* **Text Preprocessing:** Automated tokenization, punctuation removal, stop-word filtering, and NLTK PorterStemmer application.
* **Vector Building:** `scikit-learn` for TF-IDF inverted indices and `sentence-transformers` (`all-MiniLM-L6-v2`) for dense embeddings.
* **Ranking Algorithms:** 
  * Lexical: TF-IDF & BM25 (via `rank_bm25`)
  * Semantic: Sentence Transformers
* **Hybrid Fusion:** Reciprocal Rank Fusion (RRF) to blend lexical precision with semantic context.
* **Evaluation Metrics:** On-the-fly calculation of **Precision@10, Recall, MAP (Mean Average Precision), and nDCG**, dynamically graphed on a Precision-Recall Curve.

## 🛠 Deployment & Setup Instructions (For the Team)

### Prerequisites
You need Python 3.8+ installed on your deployment server.

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare the Dataset
Run the data downloader script to fetch the BBC dataset. If your environment blocks the download, manually place the `bbc_news.csv` file into the `data/` folder.
```bash
python download_data.py
```

### 3. Run the Dashboard
Deploy the interactive Streamlit dashboard:
```bash
streamlit run app.py
```
*(If deploying on a cloud service like AWS, GCP, or Render, simply bind the port provided by Streamlit to your web proxy).*

---
**Team Note for Report:** When writing the final report, emphasize the **Precision-Recall Curve** tab in the UI. It dynamically proves mathematically that the **Hybrid RRF** model outperforms individual TF-IDF or BM25 models, giving you a perfect visual aid for the evaluation section!
