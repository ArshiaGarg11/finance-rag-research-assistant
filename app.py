"""
app.py
------
Streamlit frontend for the Finance RAG Research Assistant.
Ties together the retriever and generator into a clean, demo-ready UI.

Run with:
    streamlit run app.py
"""

import sys
import time
from pathlib import Path

import streamlit as st

# Make src/ importable
sys.path.insert(0, str(Path(__file__).parent / "src"))

from retriever import FinanceRetriever
from generator import FinanceGenerator

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Finance Research Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Hide Streamlit default toolbar/header */
[data-testid="stToolbar"],
[data-testid="stDecoration"],
header[data-testid="stHeader"] {
    display: none !important;
}

/* Overall background */
[data-testid="stAppViewContainer"] {
    background-color: #0f1117;
}
[data-testid="stSidebar"] {
    background-color: #161b27;
    border-right: 1px solid #1e2d40;
}

/* Main font */
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

/* Answer box */
.answer-box {
    background: #161b27;
    border: 1px solid #1e3a5f;
    border-left: 4px solid #2d7dd2;
    border-radius: 8px;
    padding: 1.5rem 2rem;
    margin: 1rem 0;
    color: #e2e8f0;
    line-height: 1.8;
    font-size: 0.97rem;
}

/* Source badge */
.source-badge {
    display: inline-block;
    background: #1e3a5f;
    color: #60a5fa;
    border: 1px solid #2d5a8e;
    border-radius: 4px;
    padding: 2px 10px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 3px 3px 3px 0;
    font-family: 'Courier New', monospace;
    letter-spacing: 0.03em;
}

/* Company badge colors */
.badge-aapl { background: #1a2f1a; color: #4ade80; border-color: #2d5c2d; }
.badge-luv  { background: #2f1a1a; color: #f87171; border-color: #5c2d2d; }
.badge-jpm  { background: #1a1a2f; color: #a78bfa; border-color: #2d2d5c; }

/* Stats row */
.stat-box {
    background: #161b27;
    border: 1px solid #1e2d40;
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
}
.stat-number {
    font-size: 1.8rem;
    font-weight: 700;
    color: #2d7dd2;
    line-height: 1;
}
.stat-label {
    font-size: 0.75rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
}

/* Query input styling */
[data-testid="stTextArea"] textarea {
    background: #161b27 !important;
    border: 1px solid #1e3a5f !important;
    color: #e2e8f0 !important;
    border-radius: 6px !important;
    font-size: 1rem !important;
}

/* Divider */
.section-divider {
    border: none;
    border-top: 1px solid #1e2d40;
    margin: 1.5rem 0;
}

/* Chunk preview */
.chunk-preview {
    background: #0d1117;
    border: 1px solid #1e2d40;
    border-radius: 6px;
    padding: 0.8rem 1rem;
    font-size: 0.83rem;
    color: #94a3b8;
    font-family: 'Courier New', monospace;
    line-height: 1.6;
    max-height: 120px;
    overflow-y: auto;
}

/* Suggested queries */
.suggestion-chip {
    color: #cbd5e1 !important;
    display: inline-block;
    background: #161b27;
    border: 1px solid #1e3a5f;
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 0.82rem;
    color: #94a3b8;
    cursor: pointer;
    margin: 3px;
}

/* Fix heading colors */
h1, h2, h3, h4, h5 { color: #e2e8f0 !important; }
[data-testid="stMarkdownContainer"] h3 { color: #e2e8f0 !important; }
[data-testid="stMarkdownContainer"] p { color: #cbd5e1 !important; }

/* Streamlit suggestion buttons */
[data-testid="stButton"] > button {
    background: #161b27 !important;
    border: 1px solid #2d5a8e !important;
    color: #cbd5e1 !important;
    border-radius: 8px !important;
    font-size: 0.84rem !important;
    padding: 0.4rem 0.8rem !important;
}
[data-testid="stButton"] > button:hover {
    border-color: #2d7dd2 !important;
    color: #ffffff !important;
    background: #1e2d40 !important;
}

/* Sidebar text */
[data-testid="stSidebar"] label { color: #cbd5e1 !important; }
[data-testid="stSidebar"] p { color: #94a3b8 !important; }
</style>
""", unsafe_allow_html=True)


# ── Cache models (only loads once per session) ─────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_retriever():
    r = FinanceRetriever()
    r.build_index()
    return r

@st.cache_resource(show_spinner=False)
def load_generator():
    return FinanceGenerator()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Finance RAG")
    st.markdown(
        "<p style='color:#64748b;font-size:0.85rem;margin-top:-10px'>"
        "Query any public company\'s SEC filings & earnings calls</p>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    st.markdown("### Filters")
    company_filter = st.selectbox(
        "Company",
        ["All", "AAPL — Apple", "LUV — Southwest Airlines", "JPM — JPMorgan Chase"],
        index=0,
    )
    doc_type_filter = st.selectbox(
        "Document type",
        ["All", "10-K Annual Report", "Earnings Transcript"],
        index=0,
    )
    top_k = st.slider("Sources to retrieve", min_value=3, max_value=10, value=5)

    st.markdown("---")
    st.markdown("### Currently Indexed")
    st.markdown("""
    <div style='font-size:0.82rem;color:#64748b;line-height:2'>
    🍎 <b style='color:#4ade80'>AAPL</b> — 10-K 2024, 2025<br>
    ✈️ <b style='color:#f87171'>LUV</b> — 10-K 2025, 2026<br>
    🏦 <b style='color:#a78bfa'>JPM</b> — 10-K 2025, Q1 2026 Call<br><br>
    <span style='color:#374151;font-size:0.75rem'>+ any SEC filer via EDGAR</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        "<p style='font-size:0.75rem;color:#374151'>"
        "Hybrid retrieval: FAISS + BM25 + RRF<br>"
        "Generation: Llama 3.3-70b via Groq<br>"
        "Embeddings: all-MiniLM-L6-v2</p>",
        unsafe_allow_html=True
    )


# ── Main area ──────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='color:#e2e8f0;font-size:2rem;font-weight:700;margin-bottom:0'>"
    "Finance Research Assistant</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='color:#64748b;margin-top:4px;margin-bottom:2rem'>"
    "Ask questions about SEC filings and earnings calls. "
    "Answers are grounded in source documents with citations.</p>",
    unsafe_allow_html=True
)

# Stats row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="stat-box"><div class="stat-number">1397</div><div class="stat-label">Indexed Chunks</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="stat-box"><div class="stat-number">6</div><div class="stat-label">Documents</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="stat-box"><div class="stat-number">3</div><div class="stat-label">Companies</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="stat-box"><div class="stat-number">2</div><div class="stat-label">Retrieval Methods</div></div>', unsafe_allow_html=True)

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

# Query input
st.markdown("### Ask a question")

# Suggested queries
suggestions = [
    "How does Southwest Airlines hedge fuel costs?",
    "What derivative instruments does Southwest use for hedging?",
    "What are Apple's supply chain risk factors?",
    "What credit risk factors does JPMorgan face?",
    "What are JPMorgan's main sources of market risk?",
]

st.markdown("**Suggested questions:**")
cols = st.columns(len(suggestions))
for i, (col, suggestion) in enumerate(zip(cols, suggestions)):
    with col:
        if st.button(suggestion, key=f"sug_{i}", use_container_width=True):
            st.session_state["query_input"] = suggestion

query = st.text_area(
    label="Your question",
    placeholder="e.g. What is Apple's total revenue and how has it changed year over year?",
    height=80,
    key="query_input",
    label_visibility="collapsed",
)

search_clicked = st.button("🔍  Search & Generate", type="primary", use_container_width=False)

# ── Search & Generate ──────────────────────────────────────────────────────────
if search_clicked and query.strip():

    # Parse filters
    company_map  = {
        "All": None,
        "AAPL — Apple": "AAPL",
        "LUV — Southwest Airlines": "LUV",
        "JPM — JPMorgan": "JPM",
    }
    doc_type_map = {
        "All": None,
        "10-K Annual Report": "10-K",
        "Earnings Transcript": "earnings_transcript",
    }
    company_f  = company_map[company_filter]
    doc_type_f = doc_type_map[doc_type_filter]

    # Load models (cached)
    with st.spinner("Loading indexes..."):
        retriever = load_retriever()
        generator = load_generator()

    # Retrieve
    with st.spinner("Retrieving relevant sections..."):
        t0 = time.time()
        results = retriever.search(
            query,
            top_k=top_k,
            company=company_f,
            doc_type=doc_type_f,
        )
        retrieval_time = time.time() - t0

    if not results:
        st.warning("No relevant chunks found. Try adjusting your filters or rephrasing the question.")
        st.stop()

    # Generate
    with st.spinner("Generating answer with Llama 3.3-70b..."):
        t1 = time.time()
        output = generator.generate(query, results)
        generation_time = time.time() - t1

    # ── Display answer ─────────────────────────────────────────────────────────
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("### Answer")

    st.markdown(
        f'<div class="answer-box">{output["answer"]}</div>',
        unsafe_allow_html=True
    )

    # Performance stats
    perf_col1, perf_col2, perf_col3 = st.columns(3)
    with perf_col1:
        st.caption(f"⚡ Retrieval: {retrieval_time:.2f}s")
    with perf_col2:
        st.caption(f"🤖 Generation: {generation_time:.2f}s")
    with perf_col3:
        st.caption(f"🔢 Tokens used: {output['tokens_used']}")

    # ── Sources cited ──────────────────────────────────────────────────────────
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("### Sources cited")

    if output["sources_used"]:
        for num in output["sources_used"]:
            if num <= len(results):
                chunk = results[num - 1]
                meta  = chunk["metadata"]
                company  = meta.get("company", "?")
                doc_type = meta.get("doc_type", "?")
                period   = meta.get("year", meta.get("quarter", "?"))
                section  = meta.get("section", meta.get("speakers", ""))

                badge_class = f"badge-{company.lower()}"
                label = f"[Source {num}]  {company} {doc_type} ({period})"
                if section:
                    label += f"  ·  {section}"

                with st.expander(label):
                    st.markdown(
                        f'<div class="chunk-preview">{chunk["text"]}</div>',
                        unsafe_allow_html=True
                    )
    else:
        st.info("The model did not cite specific sources in this answer.")

    # ── All retrieved chunks (collapsed by default) ───────────────────────────
    with st.expander(f"📄 All {len(results)} retrieved chunks (retrieval debug)"):
        for i, chunk in enumerate(results, 1):
            meta     = chunk["metadata"]
            company  = meta.get("company", "?")
            doc_type = meta.get("doc_type", "?")
            period   = meta.get("year", meta.get("quarter", "?"))
            section  = meta.get("section", "")
            score    = chunk.get("score", 0)

            st.markdown(
                f"**Chunk {i}** — {company} {doc_type} ({period}) · {section} · "
                f"<span style='color:#64748b'>score: {score:.4f}</span>",
                unsafe_allow_html=True
            )
            st.markdown(
                f'<div class="chunk-preview">{chunk["text"][:300]}...</div>',
                unsafe_allow_html=True
            )
            st.markdown("")

elif search_clicked and not query.strip():
    st.warning("Please enter a question first.")

# ── Empty state ────────────────────────────────────────────────────────────────
else:
    st.markdown("""
    <div style='text-align:center;padding:3rem 0;color:#374151'>
        <div style='font-size:3rem;margin-bottom:1rem'>📊</div>
        <div style='font-size:1.1rem;color:#64748b'>
            Enter a question above to search across Apple, Southwest Airlines,<br>
            and JPMorgan SEC filings and earnings calls.
        </div>
    </div>
    """, unsafe_allow_html=True)
