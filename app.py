import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from core_ir import BBCNewsIRSystem
import os
import time

st.set_page_config(page_title="BBC News IR System", layout="wide")

# Inject Font Awesome 6 CDN for black font icons across the app
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
<style>
    /* Ensure all font-awesome icons default to crisp black/dark slate */
    .fa, .fa-solid, .fa-regular {
        color: #111827;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_and_prepare_system():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data", "bbc_news.csv")
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

# Font-Awesome mapping for categories
CATEGORY_ICONS = {
    "tech": "fa-microchip",
    "business": "fa-briefcase",
    "politics": "fa-landmark",
    "sport": "fa-futbol",
    "entertainment": "fa-film"
}

CATEGORY_COLORS = {
    "tech": {"bg": "#e0e7ff", "text": "#3730a3", "border": "#c7d2fe"},
    "business": {"bg": "#dcfce7", "text": "#166534", "border": "#bbf7d0"},
    "politics": {"bg": "#fef3c7", "text": "#92400e", "border": "#fde68a"},
    "sport": {"bg": "#ccfbf1", "text": "#115e59", "border": "#99f6e4"},
    "entertainment": {"bg": "#fce7f3", "text": "#9d174d", "border": "#fbcfe8"}
}

def render_category_badge(cat_name):
    cat_lower = str(cat_name).lower()
    style = CATEGORY_COLORS.get(cat_lower, {"bg": "#f3f4f6", "text": "#1f2937", "border": "#e5e7eb"})
    icon_cls = CATEGORY_ICONS.get(cat_lower, "fa-tag")
    return (
        f'<span style="background-color: {style["bg"]}; color: {style["text"]}; '
        f'padding: 3px 8px; border-radius: 4px; font-weight: 700; font-size: 11px; '
        f'text-transform: uppercase; border: 1px solid {style["border"]}; letter-spacing: 0.5px; display: inline-flex; align-items: center; gap: 5px;">'
        f'<i class="fa-solid {icon_cls}" style="color: {style["text"]}; font-size: 10px;"></i>'
        f'{cat_name.upper()}</span>'
    )

def format_score_badge(raw_score, model_name):
    """
    Formats scores mathematically and logically:
    - Cosine Similarity (TF-IDF & Semantic): strictly [0.0, 1.0], converted directly to % Match.
    - BM25: Unbounded raw score, displayed as raw number without artificial percent conversion.
    - Hybrid RRF: Normalized against theoretical maximum (2 / 61 = 0.0328) to show relative fusion saturation.
    """
    if model_name == "Hybrid (RRF)":
        max_possible = 2.0 / 61.0
        saturation_pct = min(100.0, (raw_score / max_possible) * 100.0)
        return f"RRF Score: {raw_score:.4f} ({saturation_pct:.1f}% Rank Saturation)"
    elif model_name in ["TF-IDF", "Sentence Transformer"]:
        match_pct = max(0.0, min(100.0, raw_score * 100.0))
        return f"Cosine Sim: {raw_score:.4f} ({match_pct:.1f}% Match)"
    else:
        return f"BM25 Score: {raw_score:.2f} (Unbounded Lexical Weight)"

def render_calculation_breakdown(idx, score, model_name, query, system, alpha=0.5, bm25_rank_dict=None, sem_rank_dict=None):
    """
    Renders an in-depth mathematical trace of how this specific document was scored and ranked.
    """
    raw_doc = system.raw_documents[idx]
    doc_words = set(raw_doc.lower().split())
    query_words = [w.lower() for w in query.split() if w.lower() not in system.nlp.stop_words and len(w) > 2]
    matched_terms = [w for w in query_words if any(w in dw for dw in doc_words)]
    
    icon_calc = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#111827" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:text-bottom; margin-right:5px;"><rect x="4" y="2" width="16" height="20" rx="2"></rect><line x1="8" y1="6" x2="16" y2="6"></line><line x1="16" y1="14" x2="16" y2="14.01"></line><line x1="12" y1="14" x2="12" y2="14.01"></line><line x1="8" y1="14" x2="8" y2="14.01"></line><line x1="12" y1="18" x2="12" y2="18.01"></line><line x1="8" y1="18" x2="8" y2="18.01"></line></svg>'

    if model_name == "Hybrid (RRF)":
        r_bm = bm25_rank_dict.get(idx, 1) if bm25_rank_dict else 1
        r_sem = sem_rank_dict.get(idx, 1) if sem_rank_dict else 1
        w_bm = 2.0 * (1.0 - alpha)
        w_sem = 2.0 * alpha
        comp_bm = w_bm / (60.0 + r_bm)
        comp_sem = w_sem / (60.0 + r_sem)
        max_possible = 2.0 / 61.0
        saturation = min(100.0, (score / max_possible) * 100.0)
        
        return f"""<div style="background-color: #f8fafc; border-radius: 6px; padding: 10px 12px; font-size: 13px; color: #1e293b; line-height: 1.5;">
<div style="font-weight: 700; margin-bottom: 6px; color: #0f172a; font-size: 13.5px;">{icon_calc}Calculation Trace: Hybrid Reciprocal Rank Fusion (RRF)</div>
<div><strong>Active Configuration:</strong> Alpha = {alpha:.2f} ({int((1-alpha)*100)}% BM25 Lexical / {int(alpha*100)}% Semantic Transformer), Smoothing Constant k = 60</div>
<div style="margin-top: 6px;"><strong>1. Lexical BM25 Ranking:</strong> Ranked <strong>#{r_bm}</strong> across all {len(system.documents)} articles</div>
<div style="margin-left: 10px; font-family: monospace; background:#f1f5f9; padding: 3px 8px; border-radius: 4px; display:inline-block; margin-top:2px;">Component Score = (2 * (1 - {alpha:.2f})) / (60 + {r_bm}) = {w_bm:.2f} / {60 + r_bm} = {comp_bm:.5f}</div>
<div style="margin-top: 6px;"><strong>2. Semantic Transformer Ranking:</strong> Ranked <strong>#{r_sem}</strong> across all {len(system.documents)} articles</div>
<div style="margin-left: 10px; font-family: monospace; background:#f1f5f9; padding: 3px 8px; border-radius: 4px; display:inline-block; margin-top:2px;">Component Score = (2 * {alpha:.2f}) / (60 + {r_sem}) = {w_sem:.2f} / {60 + r_sem} = {comp_sem:.5f}</div>
<div style="margin-top: 8px; border-top: 1px solid #e2e8f0; padding-top: 6px;"><strong>Combined Formula:</strong> Total Score = BM25 Component + Semantic Component<br><span style="font-family: monospace;">Total Score = {comp_bm:.5f} + {comp_sem:.5f} = <strong>{score:.4f}</strong></span></div>
<div style="margin-top: 5px;"><strong>Relative Rank Saturation:</strong> {score:.4f} / 0.0328 (Theoretical max at Rank #1 on both engines) = <strong>{saturation:.1f}%</strong></div>
</div>"""
    elif model_name == "Sentence Transformer":
        match_pct = max(0.0, min(100.0, score * 100.0))
        return f"""<div style="background-color: #f8fafc; border-radius: 6px; padding: 10px 12px; font-size: 13px; color: #1e293b; line-height: 1.5;">
<div style="font-weight: 700; margin-bottom: 6px; color: #0f172a; font-size: 13.5px;">{icon_calc}Calculation Trace: Dense Semantic Cosine Similarity</div>
<div><strong>Embedding Architecture:</strong> 384-dimensional continuous vector space via 'all-MiniLM-L6-v2'</div>
<div style="margin-top: 5px;"><strong>Mathematical Formula:</strong><br><span style="font-family: monospace; background:#f1f5f9; padding: 3px 8px; border-radius: 4px; display:inline-block; margin-top:2px;">Cosine Similarity = DotProduct(Vector_Q, Vector_D) / (||Vector_Q|| * ||Vector_D||)</span></div>
<div style="margin-top: 5px;">Since Sentence Transformer embeddings are L2 unit-normalized (Length = 1.0):<br><span style="font-family: monospace; background:#f1f5f9; padding: 3px 8px; border-radius: 4px; display:inline-block; margin-top:2px;">Cosine Similarity = DotProduct(Vector_Q, Vector_D) = <strong>{score:.4f}</strong></span></div>
<div style="margin-top: 8px; border-top: 1px solid #e2e8f0; padding-top: 6px;"><strong>Relevance Match:</strong> <strong>{match_pct:.1f}% Match</strong><div style="color: #64748b; font-size: 12px; margin-top: 2px;">Captures deep contextual meaning through self-attention across 6 transformer layers, matching conceptually related ideas even when different words are used.</div></div>
</div>"""
    elif model_name == "TF-IDF":
        match_pct = max(0.0, min(100.0, score * 100.0))
        matched_str = ", ".join([f"'{m}'" for m in matched_terms]) if matched_terms else "Stemmed/sub-word overlap"
        vocab_size = len(system.lexical.tfidf.vocabulary_) if system.lexical.tfidf and hasattr(system.lexical.tfidf, 'vocabulary_') else 30000
        return f"""<div style="background-color: #f8fafc; border-radius: 6px; padding: 10px 12px; font-size: 13px; color: #1e293b; line-height: 1.5;">
<div style="font-weight: 700; margin-bottom: 6px; color: #0f172a; font-size: 13.5px;">{icon_calc}Calculation Trace: TF-IDF Vector Space Model</div>
<div><strong>Vocabulary Dimension:</strong> {vocab_size:,} distinct terms across the BBC corpus</div>
<div><strong>Matched Query Terms:</strong> <span style="font-family: monospace; background:#f1f5f9; padding: 2px 6px; border-radius: 3px;">{matched_str}</span></div>
<div style="margin-top: 5px;"><strong>Mathematical Formula:</strong><br><span style="font-family: monospace; background:#f1f5f9; padding: 3px 8px; border-radius: 4px; display:inline-block; margin-top:2px;">Cosine Similarity = Sum over t of [ TF_IDF(t, Q) * TF_IDF(t, D) ] / (||Vector_Q|| * ||Vector_D||)</span></div>
<div style="margin-top: 8px; border-top: 1px solid #e2e8f0; padding-top: 6px;"><strong>Resulting Cosine Similarity:</strong> <strong>{score:.4f}</strong> (<strong>{match_pct:.1f}% Match</strong>)<div style="color: #64748b; font-size: 12px; margin-top: 2px;">Why is this number in the 0.20 - 0.45 range? Short queries contain only ~3 non-zero coordinates out of {vocab_size:,} total dimensions, so the inner product angle reflects high-dimensional sparsity.</div></div>
</div>"""
    else:  # BM25
        doc_len = len(raw_doc.split())
        matched_str = ", ".join([f"'{m}'" for m in matched_terms]) if matched_terms else "Stemmed term overlap"
        return f"""<div style="background-color: #f8fafc; border-radius: 6px; padding: 10px 12px; font-size: 13px; color: #1e293b; line-height: 1.5;">
<div style="font-weight: 700; margin-bottom: 6px; color: #0f172a; font-size: 13.5px;">{icon_calc}Calculation Trace: Okapi BM25 Lexical Score</div>
<div><strong>Model Parameters:</strong> k1 = 1.5 (term frequency saturation factor), b = 0.75 (length normalization penalty)</div>
<div><strong>Document Length:</strong> {doc_len} words</div>
<div><strong>Matched Terms:</strong> <span style="font-family: monospace; background:#f1f5f9; padding: 2px 6px; border-radius: 3px;">{matched_str}</span></div>
<div style="margin-top: 5px;"><strong>Mathematical Formula:</strong><br><span style="font-family: monospace; background:#f1f5f9; padding: 3px 8px; border-radius: 4px; display:inline-block; margin-top:2px;">Score = Sum over q of [ IDF(q) * ( TF(q, D) * 2.5 ) / ( TF(q, D) + 1.5 * (0.25 + 0.75 * (|D| / avgdl)) ) ]</span></div>
<div style="margin-top: 8px; border-top: 1px solid #e2e8f0; padding-top: 6px;"><strong>Computed BM25 Score:</strong> <strong>{score:.2f}</strong><div style="color: #64748b; font-size: 12px; margin-top: 2px;">BM25 produces an unbounded score. Higher values indicate higher frequency of rare, discriminative keywords while penalizing excessively long documents.</div></div>
</div>"""

st.markdown("""
<div style="padding: 10px 0px 15px 0px; border-bottom: 2px solid #e5e7eb; margin-bottom: 20px;">
    <h1 style="margin: 0; font-size: 28px; font-weight: 800; color: #111827;">
        <i class="fa-solid fa-newspaper" style="color: #111827; margin-right: 8px;"></i>BBC News Information Retrieval System
    </h1>
    <p style="margin: 5px 0 0 0; color: #4b5563; font-size: 14px;">
        Hybrid Multi-Stage Search Engine combining Lexical BM25, TF-IDF, Dense Transformer Embeddings, and Machine Learning Classification.
    </p>
</div>
""", unsafe_allow_html=True)

system = load_and_prepare_system()

# Sidebar: Search Settings
st.sidebar.markdown('### <i class="fa-solid fa-sliders" style="color: #111827; margin-right: 6px;"></i> Search Configuration', unsafe_allow_html=True)
original_query = st.sidebar.text_input("Search Query:", value="technology and artificial intelligence")

# Innovations & Pipeline Controls
st.sidebar.markdown("---")
st.sidebar.markdown('### <i class="fa-solid fa-lightbulb" style="color: #111827; margin-right: 6px;"></i> Pipeline Options', unsafe_allow_html=True)
use_query_expansion = st.sidebar.checkbox("Enable Query Expansion (WordNet)", value=False)
use_summarization = st.sidebar.checkbox("Query-Focused Snippets", value=True)
highlight_matches = st.sidebar.checkbox("Highlight Keyword Matches", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown('### <i class="fa-solid fa-layer-group" style="color: #111827; margin-right: 6px;"></i> Retrieval Settings', unsafe_allow_html=True)
model_choice = st.sidebar.selectbox("Ranking Algorithm", ["Hybrid (RRF)", "BM25", "TF-IDF", "Sentence Transformer"])

# Tunable Hybrid Weight Slider
if model_choice == "Hybrid (RRF)":
    hybrid_alpha = st.sidebar.slider(
        "Hybrid Balance (Alpha): Lexical vs Semantic",
        min_value=0.0, max_value=1.0, value=0.5, step=0.05,
        help="0.0 = 100% BM25 Lexical, 0.5 = Balanced 50/50 RRF, 1.0 = 100% Dense Semantic Transformer"
    )
    if hybrid_alpha == 0.5:
        st.sidebar.caption("Balance: 50% BM25 Keyword + 50% Semantic Meaning")
    elif hybrid_alpha < 0.5:
        bm25_pct = int((1.0 - hybrid_alpha) * 100)
        sem_pct = int(hybrid_alpha * 100)
        st.sidebar.caption(f"Balance: {bm25_pct}% Keyword-Heavy (BM25) / {sem_pct}% Semantic")
    else:
        bm25_pct = int((1.0 - hybrid_alpha) * 100)
        sem_pct = int(hybrid_alpha * 100)
        st.sidebar.caption(f"Balance: {sem_pct}% Concept-Heavy (Semantic) / {bm25_pct}% Lexical")
else:
    hybrid_alpha = 0.5

top_k = st.sidebar.slider("Results to Retrieve (Top K)", min_value=5, max_value=50, value=10)
expected_category = st.sidebar.selectbox("Target Category (Evaluation Tab)", sorted(list(set(system.categories))))
st.sidebar.markdown('<p style="font-size: 12px; color: #6b7280;"><i class="fa-solid fa-circle-info" style="color: #111827; margin-right: 4px;"></i> Target category used as benchmark ground-truth on the Performance Evaluation tab.</p>', unsafe_allow_html=True)

# Main Navigation Tabs
tabs = st.tabs(["Search Results", "3D Semantic Universe", "Performance Evaluation", "ML Classifier & Ingestion"])

# Process Query Expansion
query = original_query
if original_query:
    if use_query_expansion:
        expansion_details = system.get_expansion_details(original_query)
        query = expansion_details["expanded_query"]
        with st.expander("[Query Expansion Analysis] Original Query vs Injected Synonyms", expanded=False):
            st.markdown(f"**Original Query:** `{original_query}`")
            st.markdown(f"**Extracted Content Terms:** `{', '.join(expansion_details['original_terms']) if expansion_details['original_terms'] else 'None'}`")
            st.markdown(f"**Injected Synonyms (WordNet):** `{', '.join(expansion_details['added_synonyms']) if expansion_details['added_synonyms'] else 'None'}`")
            st.markdown(f"**Final Search Query:** `{query}`")
            st.caption("Drift Guard active: Original terms are placed first and synset lemmas are capped at 2 per term to preserve primary search intent.")

# TAB 0: Search Results
with tabs[0]:
    col_head, col_filter = st.columns([3, 2])
    with col_head:
        st.markdown(f'<h3 style="margin:0; font-size:20px; color:#111827;"><i class="fa-solid fa-magnifying-glass" style="color: #111827; margin-right: 8px;"></i>Results for: <em>{original_query}</em></h3>', unsafe_allow_html=True)
    with col_filter:
        available_categories = ["All Categories"] + sorted(list(set(system.categories)))
        selected_category = st.selectbox("Category Filter:", available_categories)
    
    if query:
        with st.spinner(f"Running search with {model_choice}..."):
            total_corpus_docs = len(system.documents)
            
            # Execute search with active algorithm across full corpus
            bm25_rank_dict, sem_rank_dict = None, None
            if model_choice == "Hybrid (RRF)":
                bm25_indices, _ = system.search_bm25(query, top_n=total_corpus_docs)
                sem_indices, _ = system.search_semantic(query, top_n=total_corpus_docs)
                bm25_rank_dict = {doc_idx: r + 1 for r, doc_idx in enumerate(bm25_indices)}
                sem_rank_dict = {doc_idx: r + 1 for r, doc_idx in enumerate(sem_indices)}
                all_indices, all_scores = system.search_hybrid_rrf(query, top_n=total_corpus_docs, alpha=hybrid_alpha)
            elif model_choice == "BM25":
                all_indices, all_scores = system.search_bm25(query, top_n=total_corpus_docs)
            elif model_choice == "TF-IDF":
                all_indices, all_scores = system.search_tfidf(query, top_n=total_corpus_docs)
            else:
                all_indices, all_scores = system.search_semantic(query, top_n=total_corpus_docs)
            
            # Feature: Top Recommendation from Each Category (Collapsed by default)
            with st.expander(f"[Category Breakdown] Top Recommendation from Each Category (Ranked by {model_choice})", expanded=False):
                st.caption(f"Single highest-ranking document per category based on active ranking algorithm ({model_choice}):")
                all_cats = sorted(list(set(system.categories)))
                top_per_cat = {}
                for c in all_cats:
                    for i, s in zip(all_indices, all_scores):
                        if system.categories[i].lower() == c.lower():
                            top_per_cat[c] = (i, s)
                            break
                
                cat_cols = st.columns(len(all_cats))
                for col, c in zip(cat_cols, all_cats):
                    with col:
                        if c in top_per_cat:
                            c_idx, c_score = top_per_cat[c]
                            c_raw = system.raw_documents[c_idx]
                            c_snippet = system.query_focused_summary(c_raw, query, top_sentences=1)
                            badge_html = render_category_badge(c)
                            score_label = format_score_badge(c_score, model_choice)
                            
                            st.markdown(badge_html, unsafe_allow_html=True)
                            st.caption(score_label)
                            
                            if highlight_matches:
                                snippet_disp = system.highlight_keywords(c_snippet[:130] + "...", query)
                                st.markdown(f"<div style='font-size:12px; line-height:1.4;'>{snippet_disp}</div>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"<div style='font-size:12px; line-height:1.4;'>{c_snippet[:130]}...</div>", unsafe_allow_html=True)
                            
                            with st.popover("Inspect Full Article"):
                                st.markdown(render_category_badge(c), unsafe_allow_html=True)
                                st.markdown(f"**Score:** `{score_label}`")
                                st.markdown("---")
                                calc_trace_html = render_calculation_breakdown(
                                    c_idx, c_score, model_choice, query, system,
                                    alpha=hybrid_alpha, bm25_rank_dict=bm25_rank_dict, sem_rank_dict=sem_rank_dict
                                )
                                with st.expander("Score Calculation Breakdown", expanded=False):
                                    st.markdown(calc_trace_html, unsafe_allow_html=True)
                                st.markdown("---")
                                if highlight_matches:
                                    full_disp = system.highlight_keywords(c_raw[:1000], query)
                                    st.markdown(f"<div style='font-size:13px; line-height:1.6;'>{full_disp}...</div>", unsafe_allow_html=True)
                                else:
                                    st.write(c_raw[:1000] + "...")
                st.markdown("---")

            # Filter results if user selected a specific category
            if selected_category != "All Categories":
                filtered = [(i, s) for i, s in zip(all_indices, all_scores) if system.categories[i].lower() == selected_category.lower()]
                indices = [f[0] for f in filtered[:top_k]]
                scores = [f[1] for f in filtered[:top_k]]
            else:
                indices = all_indices[:top_k]
                scores = all_scores[:top_k]
            
            if len(indices) == 0:
                st.info(f"No documents found matching category filter '{selected_category}'.")
            else:
                st.caption(f"Displaying top {len(indices)} results. (Filter: **{selected_category}** | Algorithm: **{model_choice}**)")
                for rank, (idx, score) in enumerate(zip(indices, scores), 1):
                    raw_text = system.raw_documents[idx]
                    category_name = system.categories[idx]
                    score_text = format_score_badge(score, model_choice)
                    badge_html = render_category_badge(category_name)
                    
                    with st.expander(f"#{rank} | [{category_name.upper()}] | {score_text}", expanded=False):
                        st.markdown(f"**Category:** {badge_html} &nbsp;&nbsp;|&nbsp;&nbsp; **Relevance Metric:** `{score_text}`", unsafe_allow_html=True)
                        st.markdown("---")
                        
                        if use_summarization:
                            summary = system.query_focused_summary(raw_text, query)
                            st.markdown('**<i class="fa-solid fa-wand-magic-sparkles" style="color: #111827; margin-right: 5px;"></i>Query-Focused Summary:**', unsafe_allow_html=True)
                            if highlight_matches:
                                highlighted_summary = system.highlight_keywords(summary, query)
                                st.markdown(f"<div style='background-color:#f8fafc; border-left:3px solid #2563eb; padding:8px 12px; margin-bottom:10px;'>{highlighted_summary}</div>", unsafe_allow_html=True)
                            else:
                                st.info(summary)
                        
                        st.markdown('**<i class="fa-solid fa-file-lines" style="color: #111827; margin-right: 5px;"></i>Full Text Snippet:**', unsafe_allow_html=True)
                        snippet_text = raw_text[:850] + ("..." if len(raw_text) > 850 else "")
                        if highlight_matches:
                            highlighted_text = system.highlight_keywords(snippet_text, query)
                            st.markdown(f"<div style='font-size:13.5px; line-height:1.6; color:#1f2937;'>{highlighted_text}</div>", unsafe_allow_html=True)
                        else:
                            st.write(snippet_text)
                            
                        # Clean native nested expander for on-demand score calculation trace
                        with st.expander("Score Calculation Breakdown & Mathematical Trace", expanded=False):
                            calc_trace_html = render_calculation_breakdown(
                                idx, score, model_choice, query, system,
                                alpha=hybrid_alpha, bm25_rank_dict=bm25_rank_dict, sem_rank_dict=sem_rank_dict
                            )
                            st.markdown(calc_trace_html, unsafe_allow_html=True)

        # Tab 0 Educational Explainer Expander (NO LATEX MATH)
        with st.expander("[Information Guide] Search Algorithms, Mathematics & Relevance Scores", expanded=False):
            st.markdown("""
            ### <i class="fa-solid fa-circle-info" style="color: #111827; margin-right: 6px;"></i> Understanding the 4 Ranking Algorithms & Relevance Scores
            
            #### 1. Okapi BM25 (Probabilistic Lexical Search)
            * **What it does**: Evaluates exact term matches while penalizing documents that contain redundant keyword spam or excessive document length.
            * **Mathematical Formula**:
              For each query term q present in document D:
              Score_BM25(D, Q) = Sum over q of [ IDF(q) * ( TF(q, D) * (k1 + 1) ) / ( TF(q, D) + k1 * (1 - b + b * (DocLength / AvgDocLength)) ) ]
              Where k1 = 1.5, b = 0.75, and IDF(q) = log( (TotalDocs - DocCount(q) + 0.5) / (DocCount(q) + 0.5) + 1 ).
            * **Score Characteristics**:
              BM25 produces unbounded real numbers (typically between 1.0 and 25.0).
              It is not a probability or cosine angle. Hence, BM25 is displayed in its genuine raw score format rather than a synthetic percentage.
            
            #### 2. TF-IDF with Cosine Similarity (Vector Space Model)
            * **What it does**: Maps documents and queries into high-dimensional sparse bag-of-words vectors and measures the cosine of the angle between them.
            * **Mathematical Formula**:
              Cosine_Sim(q, d) = DotProduct(q, d) / ( Length(q) * Length(d) )
              Where term weights are computed as TF(t, d) * IDF(t).
            * **Score Characteristics & Match Percentage**:
              Because TF-IDF vector weights are non-negative, the cosine similarity lies strictly between 0.0 and 1.0.
              A score of 0.35 means Cosine Similarity = 0.35 (35.0% Match).
              In Information Retrieval, a 4-word query evaluated against a 400-word document typically produces cosine similarities between 0.15 and 0.45 because most non-zero coordinates in the document vector do not exist in the short query.
            
            #### 3. Sentence Transformer (Dense Semantic Embeddings)
            * **What it does**: Encodes documents and queries into dense 384-dimensional continuous vector embeddings using the deep transformer model 'all-MiniLM-L6-v2'.
            * **Mathematical Formula**:
              Cosine_Sim(Embedding_q, Embedding_d) = DotProduct(Embedding_q, Embedding_d) / ( Norm(Embedding_q) * Norm(Embedding_d) )
            * **Score Characteristics & Match Percentage**:
              Embeddings are L2-normalized unit vectors. Cosine similarity ranges between -1.0 and 1.0 (with positive semantic matches lying between 0.0 and 1.0).
              A score of 0.3960 means 39.6% semantic match.
              Even if zero exact words match, dense embeddings capture semantic meaning (e.g., 'artificial intelligence' matches 'machine learning algorithms').
            
            #### 4. Hybrid Search via Reciprocal Rank Fusion (RRF)
            * **What it does**: Combines the precision of exact keyword matching (BM25) with conceptual semantic understanding (Sentence Transformers).
            * **Mathematical Formula**:
              RRF_Score(d) = [ 2 * (1 - alpha) / (k + Rank_BM25(d) + 1) ] + [ 2 * alpha / (k + Rank_Semantic(d) + 1) ]
              Where k = 60 is a smoothing constant, and alpha tunes the balance between BM25 and Semantic.
            * **Why RRF Scores Look Small & Rank Saturation Percentage**:
              The maximum theoretical score achievable by any document (if it achieves Rank 1 in both BM25 and Semantic lists simultaneously) is:
              Max_Possible_Score = 2 / (60 + 1) = 2 / 61 = 0.032787.
              Therefore, an RRF score of 0.0325 is not a low score; it achieves 0.0325 / 0.032787 = 99.1% of the maximum possible fusion rank saturation!
            """, unsafe_allow_html=True)

# TAB 1: 3D Semantic Universe
with tabs[1]:
    st.markdown('<h3 style="margin:0; font-size:20px; color:#111827;"><i class="fa-solid fa-cubes" style="color: #111827; margin-right: 8px;"></i>3D Semantic Universe: Document & Query Geometry</h3>', unsafe_allow_html=True)
    st.caption("Visualizing dense 384-dimensional sentence transformer embeddings reduced to 3 principal spatial components via Principal Component Analysis (PCA).")
    
    if query and system.embeddings_3d is not None:
        query_emb = system.semantic_model.encode([query])
        query_3d = system.pca_model.transform(query_emb)[0]
        
        df_plot = pd.DataFrame({
            'x': system.embeddings_3d[:, 0], 'y': system.embeddings_3d[:, 1], 'z': system.embeddings_3d[:, 2],
            'category': [c.upper() for c in system.categories],
            'text_snippet': [doc[:120] + "..." for doc in system.raw_documents],
            'type': ['Article'] * len(system.documents)
        })
        
        query_row = pd.DataFrame({
            'x': [query_3d[0]], 'y': [query_3d[1]], 'z': [query_3d[2]],
            'category': ['[ACTIVE QUERY]'], 'text_snippet': [query], 'type': ['Search Query']
        })
        df_plot = pd.concat([df_plot, query_row], ignore_index=True)
        
        color_map = {
            'TECH': '#4338ca', 'BUSINESS': '#15803d', 'POLITICS': '#b45309',
            'SPORT': '#0f766e', 'ENTERTAINMENT': '#be185d', '[ACTIVE QUERY]': '#dc2626'
        }
        
        fig = px.scatter_3d(
            df_plot, x='x', y='y', z='z', color='category', symbol='type',
            color_discrete_map=color_map, hover_data=['text_snippet'],
            title="3D Semantic Document Distribution", size_max=10, opacity=0.8
        )
        fig.update_traces(marker=dict(size=4), selector=dict(type='scatter3d', mode='markers'))
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("[Information Guide] How Does the 3D Semantic Map Work?", expanded=False):
        st.markdown("""
        ### <i class="fa-solid fa-circle-info" style="color: #111827; margin-right: 6px;"></i> Mathematics of Dimensionality Reduction (PCA)
        1. **High-Dimensional Embeddings**:
           Every document in the 2,225 BBC corpus is converted by the 'all-MiniLM-L6-v2' transformer into a vector of 384 numbers. Each dimension represents a subtle latent semantic feature.
        2. **Principal Component Analysis (PCA)**:
           Because human visual perception is limited to 3 dimensions, PCA calculates the eigenvectors and eigenvalues of the 384-dimensional covariance matrix.
           It projects all 384 dimensions onto the 3 orthogonal axes that preserve the largest variance in meaning.
        3. **Query Projection**:
           When you enter a search query, it is encoded into 384 dimensions and projected through the exact same PCA transformation matrix into coordinates (x, y, z).
        4. **Geometric Interpretation**:
           Articles sharing semantic themes cluster tightly together. The red '[ACTIVE QUERY]' marker shows where your search query lands conceptually in relation to the document clusters.
        """, unsafe_allow_html=True)

# TAB 2: Performance Evaluation
with tabs[2]:
    st.markdown('<h3 style="margin:0; font-size:20px; color:#111827;"><i class="fa-solid fa-chart-line" style="color: #111827; margin-right: 8px;"></i>Information Retrieval Performance Evaluation</h3>', unsafe_allow_html=True)
    st.caption(f"Benchmarking retrieval algorithms against target category: **{expected_category.upper()}** (Query: *{query}*)")
    
    if query:
        metrics_data = []
        models = [
            ("TF-IDF", system.search_tfidf),
            ("BM25", system.search_bm25),
            ("Semantic", system.search_semantic),
            ("Hybrid RRF", lambda q, top_n: system.search_hybrid_rrf(q, top_n=top_n, alpha=hybrid_alpha))
        ]
        
        pr_curves = []
        retrieved_categories_map = {}
        
        for name, func in models:
            indices, scores = func(query, top_n=top_k)
            retrieved_categories_map[name] = [system.categories[idx].upper() for idx in indices]
            metrics = system.evaluate_metrics(indices, expected_category)
            
            # Format numbers clearly with percentage representations
            p10 = metrics["Precision@10"]
            rec = metrics["Recall"]
            map_s = metrics["MAP"]
            ndcg_s = metrics["nDCG"]
            
            metrics_data.append({
                "Model": name,
                "Precision@10": f"{p10:.4f} ({p10*100:.1f}%)",
                "Recall@K": f"{rec:.4f} ({rec*100:.1f}%)",
                "MAP@K": f"{map_s:.4f} ({map_s*100:.1f}%)",
                "nDCG@K": f"{ndcg_s:.4f} ({ndcg_s*100:.1f}%)",
                "_p10_num": p10,
                "_rec_num": rec,
                "_map_num": map_s,
                "_ndcg_num": ndcg_s
            })
            
            # Calculate PR curve coordinates
            y_true = [1 if system.categories[idx] == expected_category else 0 for idx in indices]
            total_rel = system.categories.count(expected_category)
            precs, recs, hits = [], [], 0
            for i, y in enumerate(y_true):
                if y == 1: hits += 1
                precs.append(hits / (i + 1))
                recs.append(hits / total_rel if total_rel > 0 else 0)
            pr_curves.append((name, precs, recs))
        
        # Display Comparative Table
        display_df = pd.DataFrame(metrics_data).set_index("Model")
        view_cols = ["Precision@10", "Recall@K", "MAP@K", "nDCG@K"]
        st.dataframe(display_df[view_cols], use_container_width=True)
        
        col_chart_pr, col_chart_dist = st.columns([1, 1])
        
        with col_chart_pr:
            st.markdown('#### <i class="fa-solid fa-chart-line" style="color: #111827; margin-right: 6px;"></i>Precision-Recall Curve', unsafe_allow_html=True)
            fig_pr = go.Figure()
            for name, precs, recs in pr_curves:
                fig_pr.add_trace(go.Scatter(x=recs, y=precs, mode='lines+markers', name=name))
            fig_pr.update_layout(
                xaxis_title="Recall", yaxis_title="Precision",
                hovermode="x unified", margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig_pr, use_container_width=True)
            
        with col_chart_dist:
            st.markdown(f'#### <i class="fa-solid fa-chart-pie" style="color: #111827; margin-right: 6px;"></i>Category Distribution in Top {top_k} Results (Hybrid)', unsafe_allow_html=True)
            hybrid_cats = retrieved_categories_map.get("Hybrid RRF", [])
            if hybrid_cats:
                cat_counts = pd.Series(hybrid_cats).value_counts().reset_index()
                cat_counts.columns = ["Category", "Count"]
                cat_counts["Percentage"] = (cat_counts["Count"] / len(hybrid_cats) * 100).round(1)
                
                fig_dist = px.pie(
                    cat_counts, names="Category", values="Count",
                    color="Category",
                    color_discrete_map={
                        'TECH': '#4338ca', 'BUSINESS': '#15803d', 'POLITICS': '#b45309',
                        'SPORT': '#0f766e', 'ENTERTAINMENT': '#be185d'
                    },
                    hole=0.45
                )
                fig_dist.update_traces(textposition='inside', textinfo='percent+label')
                fig_dist.update_layout(margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_dist, use_container_width=True)

    with st.expander("[Information Guide] Metric Calculation Breakdown & Why Values Lie in Their Ranges", expanded=False):
        st.markdown(f"""
        ### <i class="fa-solid fa-circle-info" style="color: #111827; margin-right: 6px;"></i> Step-by-Step Metric Calculations for Information Retrieval
        
        #### 1. Ground Truth Benchmark Definition
        In standard benchmarks (like TREC), human annotators manually label whether each article answers a query.
        Because the BBC dataset does not include query-specific human judgments, this system uses document category as an objective proxy:
        * An article is graded **Relevant (1)** if its true category matches your selected target category: **`{expected_category.upper()}`**.
        * Otherwise, it is graded **Non-Relevant (0)**.
        
        #### 2. Precision@10
        * **Formula**: Precision@10 = (Number of relevant articles among top 10 retrieved) / 10
        * **Range**: Strictly 0.0 to 1.0 (0% to 100%).
        * **Interpretation**: If 4 out of the top 10 articles belong to `{expected_category.upper()}`, Precision@10 = 4 / 10 = 0.4000 (40.0%).
        
        #### 3. Recall@K
        * **Formula**: Recall@K = (Number of relevant articles in retrieved top K) / (Total relevant articles in entire 2,225 corpus)
        * **Why is Recall small? (Crucial for beginners to understand)**:
          The BBC corpus contains approximately 400 to 500 articles for category `{expected_category.upper()}`.
          If your retrieval slider is set to Top {top_k}, even if every single retrieved article was relevant, the maximum possible recall would be:
          Max Possible Recall = {top_k} / {system.categories.count(expected_category)} = {top_k / max(1, system.categories.count(expected_category)):.4f} ({top_k / max(1, system.categories.count(expected_category))*100:.1f}%).
          Retrieving 10 documents cannot physically return 400 documents. Hence, a recall score around 1% to 2% is mathematically normal for small retrieval windows.
        
        #### 4. MAP (Mean Average Precision)
        * **Formula**:
          AP = (1 / min(K, Total_Relevant)) * Sum from i=1 to K of [ Precision@i * Relevance(i) ]
        * **Interpretation**: Evaluates ranking quality. If relevant documents appear at ranks #1, #2, and #3, MAP is significantly higher than if they appear at ranks #8, #9, and #10.
        
        #### 5. nDCG (Normalized Discounted Cumulative Gain)
        * **Formula**: nDCG = DCG / IDCG, where DCG = Sum from i=1 to K of [ Relevance(i) / log2(i + 1) ]
        * **Interpretation**: Position-weighted ranking utility with logarithmic decay. Placing a relevant document at rank #1 provides full utility (1 / log2(2) = 1.0), whereas rank #10 provides discounted utility (1 / log2(11) = 0.289). IDCG represents the score of a perfect ranking where all relevant documents are placed first.
        """, unsafe_allow_html=True)

# TAB 3: ML Classifier & Document Ingestion
with tabs[3]:
    st.markdown('<h3 style="margin:0; font-size:20px; color:#111827;"><i class="fa-solid fa-brain" style="color: #111827; margin-right: 8px;"></i>Machine Learning Classifier & Live Document Ingestion</h3>', unsafe_allow_html=True)
    st.markdown("Train a multi-class Logistic Regression classifier on TF-IDF vectors of the 2,225 BBC corpus. Once trained, you can paste breaking news to predict its category, append it to the corpus, and dynamically re-index all search and 3D models in real-time.")
    
    col_btn_train, col_status = st.columns([1, 2])
    with col_btn_train:
        train_clicked = st.button("Train Classification Model")
    with col_status:
        if system.is_trained:
            st.success("Classifier Status: Trained and active in memory.")
        else:
            st.info("Classifier Status: Not yet trained. Click 'Train Classification Model' to initialize.")
            
    if train_clicked:
        with st.spinner("Fitting Logistic Regression model with 80/20 train/validation split..."):
            accuracy = system.train_classifier()
            st.success(f"Model successfully trained! Validation Accuracy: **{accuracy*100:.2f}%**")
    
    st.markdown("---")
    st.markdown('#### <i class="fa-solid fa-cloud-arrow-up" style="color: #111827; margin-right: 6px;"></i>Submit Breaking News', unsafe_allow_html=True)
    new_article = st.text_area("Paste a new news article below to ingest it dynamically into the search engine:")
    save_to_disk = st.checkbox(
        "Save permanently to disk (updates data/bbc_news.csv and data/embeddings_cache.npy)",
        value=False,
        help="If unchecked, document is stored in RAM for current session only. If checked, persisted to disk across application restarts."
    )
    
    if st.button("Ingest and Index Article"):
        if not system.is_trained:
            st.error("Please click 'Train Classification Model' above first so the system can predict the category!")
        elif not new_article.strip():
            st.error("Please paste non-empty article text.")
        else:
            with st.spinner("Classifying and Re-Indexing Search Engine..."):
                start_t = time.time()
                pred_cat = system.add_live_article(new_article, save_to_disk=save_to_disk)
                end_t = time.time()
                
                st.success(f"Successfully classified and indexed in {end_t - start_t:.3f} seconds!")
                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    st.markdown("**Predicted Category:**")
                    st.markdown(render_category_badge(pred_cat), unsafe_allow_html=True)
                with col_res2:
                    st.markdown("**Total Corpus Size:**")
                    st.metric(label="Articles in Corpus", value=len(system.documents))
                    
                storage_loc = "permanently saved to disk (`data/bbc_news.csv` + `data/embeddings_cache.npy`)" if save_to_disk else "stored in temporary RAM for this session"
                st.info(f"The document is now live in the search index, 3D PCA space, and category recommendations ({storage_loc}). Go to the 'Search Results' tab to query it!")

    with st.expander("[Information Guide] How Does Classification and Real-Time Ingestion Work?", expanded=False):
        st.markdown("""
        ### <i class="fa-solid fa-circle-info" style="color: #111827; margin-right: 6px;"></i> Pipeline Architecture: Real-Time Machine Learning Ingestion
        1. **Feature Representation**:
           The raw text of all 2,225 articles is cleaned, stripped of punctuation, tokenized, Porter-stemmed, and converted into TF-IDF vector features using Scikit-Learn's TfidfVectorizer.
        2. **Supervised Classifier**:
           A multi-class Logistic Regression model is trained using an 80% training / 20% validation split.
           Accuracy is computed as:
           Accuracy = (Number of correct category predictions on validation set) / (Total validation set items)
        3. **Live Article Inference**:
           When new breaking news is submitted:
           - The text is preprocessed and transformed through the fitted TF-IDF matrix.
           - The classifier predicts the probability distribution over the 5 classes: Tech, Business, Politics, Sport, Entertainment.
           - The highest-probability class is assigned as the predicted category.
        4. **Dynamic Search Re-Indexing**:
           Unlike static search engines that require offline rebuilds, this system dynamically updates:
           - Lexical Index: Re-fits TF-IDF and BM25 token corpus with the new document.
           - Semantic Index: Encodes the new text using the Sentence Transformer into a 384-D vector and appends it to the embedding matrix.
           - 3D PCA Projection: Updates the principal component projection so the new document appears in the 3D semantic scatter plot.
        5. **Storage Modes**:
           - **RAM-Only (Default)**: Fast, non-destructive experimentation without altering the baseline dataset.
           - **Disk Persistence**: Appends the text and category to `data/bbc_news.csv` and re-saves `data/embeddings_cache.npy` so the document remains available across computer or server restarts.
        """, unsafe_allow_html=True)
