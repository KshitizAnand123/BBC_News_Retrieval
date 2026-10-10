import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from core_ir import BBCNewsIRSystem
import os
import time

st.set_page_config(page_title="BBC News IR System", layout="wide", page_icon="📰")

@st.cache_resource
def load_and_prepare_system():
    data_path = os.path.join("data", "bbc_news.csv")
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        st.warning("No dataset found. Loading synthetic dataset...")
        df = pd.DataFrame({
            "category": ["tech", "business", "politics", "sport", "entertainment"],
            "text": [
                "Apple releases new iPhone with advanced AI capabilities.",
                "Stock markets rally as inflation drops below 2 percent.",
                "Prime Minister announces new healthcare budget.",
                "Local football team wins championship.",
                "Award-winning actor stars in a thrilling new sci-fi movie."
            ]
        })
        
    system = BBCNewsIRSystem()
    system.fit(df)
    return system

st.title("📰 BBC News Next-Gen IR System")
st.markdown("**Hybrid Search (RRF)** featuring *Machine Learning Classification*, *Query Expansion*, *AI Snippets*, and *3D Semantic Mapping*.")

system = load_and_prepare_system()

# Sidebar for Search Settings
st.sidebar.header("🔍 Search Configuration")
original_query = st.sidebar.text_input("Enter Search Query:", value="technology and artificial intelligence")

# Innovations Toggle
st.sidebar.markdown("---")
st.sidebar.header("🚀 Innovations")
use_query_expansion = st.sidebar.checkbox("Enable Query Expansion", value=False)
use_summarization = st.sidebar.checkbox("Query-Focused Summaries", value=True)
st.sidebar.markdown("---")

model_choice = st.sidebar.selectbox("Ranking Algorithm", ["Hybrid (RRF)", "BM25", "TF-IDF", "Sentence Transformer"])
top_k = st.sidebar.slider("Results to Retrieve", min_value=5, max_value=50, value=10)
expected_category = st.sidebar.selectbox("Expected Category (for Evaluation Metrics)", list(set(system.categories)))

tabs = st.tabs(["🔎 Search Results", "🌌 Semantic Universe (3D)", "📊 Performance Evaluation", "📡 ML Training & Live Ingestion"])

query = original_query
if use_query_expansion and original_query:
    query = system.expand_query(original_query)
    st.info(f"**Expanded Query:** {query}")

with tabs[0]:
    st.subheader(f"Results for: *{original_query}*")
    
    if query:
        with st.spinner(f"Searching using {model_choice}..."):
            if model_choice == "Hybrid (RRF)":
                indices, scores = system.search_hybrid_rrf(query, top_n=top_k)
            elif model_choice == "BM25":
                indices, scores = system.search_bm25(query, top_n=top_k)
            elif model_choice == "TF-IDF":
                indices, scores = system.search_tfidf(query, top_n=top_k)
            else:
                indices, scores = system.search_semantic(query, top_n=top_k)
                
            if len(indices) == 0:
                st.info("No results found.")
            else:
                for rank, (idx, score) in enumerate(zip(indices, scores), 1):
                    raw_text = system.raw_documents[idx]
                    
                    with st.expander(f"#{rank} | Category: {system.categories[idx].upper()} | Score: {score:.4f}"):
                        if use_summarization:
                            summary = system.query_focused_summary(raw_text, query)
                            st.markdown("**✨ AI Snippet (Query-Focused):**")
                            st.write(summary)
                            st.markdown("---")
                        st.write("**Full Text Snippet:**")
                        st.write(raw_text[:800] + "...")

with tabs[1]:
    st.subheader("Interactive 3D Semantic Knowledge Map")
    if query and system.embeddings_3d is not None:
        query_emb = system.semantic_model.encode([query])
        query_3d = system.pca_model.transform(query_emb)[0]
        
        df_plot = pd.DataFrame({
            'x': system.embeddings_3d[:, 0], 'y': system.embeddings_3d[:, 1], 'z': system.embeddings_3d[:, 2],
            'category': system.categories,
            'text_snippet': [doc[:100] + "..." for doc in system.raw_documents],
            'type': ['Article'] * len(system.documents)
        })
        
        query_row = pd.DataFrame({
            'x': [query_3d[0]], 'y': [query_3d[1]], 'z': [query_3d[2]],
            'category': ['QUERY'], 'text_snippet': [query], 'type': ['Search Query']
        })
        df_plot = pd.concat([df_plot, query_row], ignore_index=True)
        
        fig = px.scatter_3d(df_plot, x='x', y='y', z='z', color='category', symbol='type',
                            hover_data=['text_snippet'], title="Semantic Document Distribution", size_max=10, opacity=0.8)
        fig.update_traces(marker=dict(size=5), selector=dict(type='scatter3d', mode='markers'))
        st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    st.subheader("Model Performance Comparison")
    if query:
        metrics_data = []
        models = [("TF-IDF", system.search_tfidf), ("BM25", system.search_bm25),
                  ("Semantic", system.search_semantic), ("Hybrid RRF", system.search_hybrid_rrf)]
        
        pr_curves = []
        for name, func in models:
            indices, scores = func(query, top_n=top_k)
            metrics = system.evaluate_metrics(indices, expected_category)
            metrics["Model"] = name
            metrics_data.append(metrics)
            
            y_true = [1 if system.categories[idx] == expected_category else 0 for idx in indices]
            total_rel = system.categories.count(expected_category)
            precs, recs, hits = [], [], 0
            for i, y in enumerate(y_true):
                if y == 1: hits += 1
                precs.append(hits / (i + 1))
                recs.append(hits / total_rel if total_rel > 0 else 0)
            pr_curves.append((name, precs, recs))
            
        metrics_df = pd.DataFrame(metrics_data).set_index("Model")
        st.dataframe(metrics_df.style.highlight_max(axis=0, color='lightgreen'))
        
        st.subheader("Precision-Recall Curve")
        fig2 = go.Figure()
        for name, precs, recs in pr_curves:
            fig2.add_trace(go.Scatter(x=recs, y=precs, mode='lines+markers', name=name))
        fig2.update_layout(xaxis_title="Recall", yaxis_title="Precision", hovermode="x unified")
        st.plotly_chart(fig2, use_container_width=True)

with tabs[3]:
    st.subheader("Machine Learning Training & Real-Time Document Ingestion")
    st.markdown("Train a Logistic Regression classifier on the TF-IDF vectors of the dataset. Once trained, you can paste breaking news into the system. The model will **predict its category**, append it to the corpus, and **dynamically re-index everything in real-time**.")
    
    if st.button("Train Classification Model"):
        with st.spinner("Training Logistic Regression Model on full dataset..."):
            accuracy = system.train_classifier()
            st.success(f"Model successfully trained! Validation Accuracy: **{accuracy*100:.2f}%**")
    
    st.markdown("---")
    st.subheader("Submit Breaking News")
    new_article = st.text_area("Paste a new article here to ingest it into the search engine in real-time:")
    if st.button("Ingest Article"):
        if not system.is_trained:
            st.error("Please click 'Train Classification Model' above first!")
        elif not new_article:
            st.error("Please paste an article text.")
        else:
            with st.spinner("Classifying and Re-Indexing Search Engine..."):
                start_t = time.time()
                pred_cat = system.add_live_article(new_article)
                end_t = time.time()
                st.success(f"Successfully processed in {end_t - start_t:.3f} seconds!")
                st.metric(label="Predicted News Category", value=pred_cat.upper())
                st.info("The new article is now live in the search index and 3D maps. Go to the Search Results tab and try to find it!")
