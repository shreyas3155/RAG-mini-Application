"""
TalkPDF - Mini RAG Document Chatbot
Streamlit Application with Modern TalkPDF Dashboard Aesthetics,
Vector Retrieval with NumPy Cosine Similarity & Google Gemini Generation.
"""

import os
import streamlit as st
from pathlib import Path

# Custom .env loader
env_path = Path(".env")
saved_api_key = ""
if env_path.exists():
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("GEMINI_API_KEY="):
            saved_api_key = line.split("=", 1)[1].strip().strip("'\"")
            os.environ["GEMINI_API_KEY"] = saved_api_key

# Page configuration
st.set_page_config(
    page_title="TalkPDF - AI Document Chatbot (RAG)",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Slate & Neon Cyan Glassmorphism)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Main Background */
    .stApp {
        background: radial-gradient(circle at top left, #0f172a, #090d16 80%);
        color: #f1f5f9;
    }

    /* Top Hero Header */
    .hero-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.8));
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(12px);
    }
    
    .hero-title {
        font-size: 26px;
        font-weight: 700;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    
    .hero-subtitle {
        color: #94a3b8;
        font-size: 14px;
        margin-top: 6px;
    }
    
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-right: 8px;
    }
    
    .badge-cyan {
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }
    
    .badge-purple {
        background: rgba(129, 140, 248, 0.15);
        color: #818cf8;
        border: 1px solid rgba(129, 140, 248, 0.3);
    }

    /* Metric Box */
    .stat-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 14px;
        text-align: center;
    }
    
    .stat-value {
        font-size: 22px;
        font-weight: 700;
        color: #38bdf8;
    }
    
    .stat-label {
        font-size: 12px;
        color: #64748b;
        text-transform: uppercase;
    }

    /* Source Citation Accordion / Box */
    .source-box {
        background: rgba(30, 41, 59, 0.5);
        border-left: 3px solid #38bdf8;
        border-radius: 6px;
        padding: 10px 14px;
        margin-top: 8px;
        font-size: 13px;
        color: #cbd5e1;
    }

    /* Streamlit Input & Buttons */
    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 14px rgba(56, 189, 248, 0.3);
    }
</style>
""", unsafe_allow_html=True)


# Lazy imports for RAG modules
try:
    from rag.document_processor import extract_text_from_pdf, chunk_document
    from rag.vector_store import VectorStoreManager
    from rag.rag_chain import RAGPipeline
    MODULES_LOADED = True
except Exception as e:
    MODULES_LOADED = False
    IMPORT_ERROR = str(e)


# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0
if "api_key" not in st.session_state:
    st.session_state["api_key"] = saved_api_key


# Cached VectorStore
@st.cache_resource(show_spinner=False)
def get_vector_store():
    return VectorStoreManager(persist_directory="./vector_store_data", collection_name="pdf_docs")


def get_rag_pipeline(api_key: str, model_name: str = "gemini-1.5-flash"):
    store = get_vector_store()
    return RAGPipeline(vector_store=store, api_key=api_key, model_name=model_name)


# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("### ⚡ **Control Panel**")
    
    # API Key Input
    user_api_key = st.text_input(
        "Google Gemini API Key",
        value=st.session_state.get("api_key", saved_api_key),
        type="password",
        help="Free key from https://aistudio.google.com/app/apikey"
    )
    if user_api_key:
        st.session_state["api_key"] = user_api_key
        st.caption("✅ Gemini API Key connected")
    else:
        st.caption("⚠️ Gemini API key required.")
        st.markdown("[👉 Get Free Gemini Key](https://aistudio.google.com/app/apikey)")

    model_options = {
        "Gemini 3 Flash (Recommended / Ultra Fast)": "gemini-3-flash",
        "Gemini 3.1 Flash-Lite (High RPM & Low Latency)": "gemini-3.1-flash-lite",
        "Gemini 3.8 Flash (Advanced Reasoning)": "gemini-3.8-flash",
        "Gemini 2.5 Flash": "gemini-2.5-flash",
        "Gemini 2.0 Flash": "gemini-2.0-flash",
        "Gemini 1.5 Flash (Classic High Capacity)": "gemini-1.5-flash",
    }
    selected_label = st.selectbox(
        "LLM Generation Model",
        options=list(model_options.keys()),
        index=0,
        help="Free API access with standard rate limits (10–15 RPM). Automatic failover kicks in if any model experiences high demand (503)."
    )
    selected_model = model_options[selected_label]
    st.caption("⚡ Free Tier: 10–15 RPM protection + auto failover")

    st.markdown("---")
    
    # Document Upload Section
    st.markdown("### 📁 **Upload Documents**")
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload notes, resume, textbooks, or documentation."
    )

    col1, col2 = st.columns(2)
    chunk_size = col1.number_input("Chunk Size", min_value=200, max_value=2000, value=700, step=100)
    chunk_overlap = col2.number_input("Overlap", min_value=0, max_value=400, value=120, step=20)
    top_k = st.slider("Top Chunks (k)", min_value=1, max_value=8, value=3)

    if uploaded_files and st.button("🚀 Process & Embed PDFs", use_container_width=True):
        if not user_api_key:
            st.error("Please provide your Google Gemini API Key first to generate embeddings!")
        elif not MODULES_LOADED:
            st.error(f"Cannot process: {IMPORT_ERROR}")
        else:
            vector_store = get_vector_store()
            new_chunks_count = 0
            
            with st.spinner("Extracting text, chunking, and embedding with Gemini..."):
                for uploaded_file in uploaded_files:
                    if uploaded_file.name not in st.session_state.processed_files:
                        pages_data = extract_text_from_pdf(uploaded_file, filename=uploaded_file.name)
                        chunks = chunk_document(pages_data, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                        added = vector_store.add_chunks(chunks, api_key=user_api_key)
                        new_chunks_count += added
                        st.session_state.processed_files.append(uploaded_file.name)
                
                st.session_state.total_chunks = vector_store.count()
                st.success(f"Indexed {new_chunks_count} chunks into Vector Database!")

    # Indexed Documents Summary
    if st.session_state.processed_files:
        st.markdown("#### 📚 **Indexed Files**")
        for f in st.session_state.processed_files:
            st.caption(f"• `{f}`")

    st.markdown("---")
    
    # Reset Buttons
    if st.button("🗑️ Reset Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.button("⚠️ Clear Vector Database", use_container_width=True):
        if MODULES_LOADED:
            get_vector_store().clear()
            st.session_state.processed_files = []
            st.session_state.total_chunks = 0
            st.session_state.messages = []
            st.warning("Vector database cleared.")
            st.rerun()


# ==========================================
# MAIN INTERFACE
# ==========================================

# Hero Banner
st.markdown("""
<div class="hero-card">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1 class="hero-title">TalkPDF — AI Document Intelligence</h1>
            <p class="hero-subtitle">Production-grade Retrieval-Augmented Generation (RAG) powered by Gemini Embeddings & Gemini 3 Flash / 3.1 Flash-Lite</p>
            <div style="margin-top: 10px;">
                <span class="badge badge-cyan">RAG Architecture</span>
                <span class="badge badge-purple">Dense Vector Embeddings</span>
                <span class="badge badge-cyan">Gemini 3 Flash</span>
                <span class="badge badge-purple">Gemini 3.1 Flash-Lite</span>
                <span class="badge badge-cyan">10–15 RPM Safe</span>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Top Stat Metrics
db_count = get_vector_store().count() if MODULES_LOADED else 0
m1, m2, m3, m4 = st.columns(4)
m1.markdown(f'<div class="stat-card"><div class="stat-value">{len(st.session_state.processed_files)}</div><div class="stat-label">PDFs Loaded</div></div>', unsafe_allow_html=True)
m2.markdown(f'<div class="stat-card"><div class="stat-value">{db_count}</div><div class="stat-label">Indexed Chunks</div></div>', unsafe_allow_html=True)
m3.markdown(f'<div class="stat-card"><div class="stat-value">{chunk_size}</div><div class="stat-label">Chunk Size (chars)</div></div>', unsafe_allow_html=True)
m4.markdown(f'<div class="stat-card"><div class="stat-value">{top_k}</div><div class="stat-label">Top-K Retrieved</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Quick Questions Pills
st.markdown("##### 💡 Suggested Questions")
pill_cols = st.columns(3)
sample_prompt = None

if pill_cols[0].button("📌 Summarize the main topics of this document", use_container_width=True):
    sample_prompt = "Summarize the main topics and key points discussed in this document."
if pill_cols[1].button("🔍 What are the key technical concepts explained?", use_container_width=True):
    sample_prompt = "What are the key technical concepts, terms, and explanations found in the document?"
if pill_cols[2].button("❓ Give me 3 important interview questions from this", use_container_width=True):
    sample_prompt = "Generate 3 important technical questions and their answers based strictly on this document."

# Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Display Sources / Citations if available
        if "sources" in msg and msg["sources"]:
            with st.expander(f"🔍 View {len(msg['sources'])} Grounding Sources from Document"):
                for idx, src in enumerate(msg["sources"], 1):
                    meta = src.get("metadata", {})
                    st.markdown(f"**Source {idx}:** `{meta.get('source', 'Unknown')}` (Page {meta.get('page', '?')}) | Similarity Score: `{src.get('score', 0):.4f}`")
                    st.markdown(f"_{src.get('text', '')}_")
                    st.divider()

# Handle User Input
user_query = st.chat_input("Ask any question about your uploaded document...") or sample_prompt

if user_query:
    # Append user question
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # RAG Response Generation
    with st.chat_message("assistant"):
        api_key = st.session_state.get("api_key", saved_api_key)
        
        if not MODULES_LOADED:
            st.error(f"Module import issue: {IMPORT_ERROR}")
        elif db_count == 0:
            msg = "⚠️ Please upload and index a PDF file in the sidebar first before asking questions!"
            st.warning(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})
        else:
            with st.spinner("Retrieving relevant document chunks and generating answer..."):
                pipeline = get_rag_pipeline(api_key, model_name=selected_model)
                response = pipeline.ask(question=user_query, top_k=top_k)

                st.markdown(response["answer"])

                if response.get("sources"):
                    with st.expander(f"🔍 View {len(response['sources'])} Grounding Sources"):
                        for idx, src in enumerate(response["sources"], 1):
                            meta = src.get("metadata", {})
                            st.markdown(f"**Source {idx}:** `{meta.get('source', 'Unknown')}` (Page {meta.get('page', '?')}) | Similarity Score: `{src.get('score', 0):.4f}`")
                            st.markdown(f"_{src.get('text', '')}_")
                            st.divider()

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response["answer"],
                    "sources": response.get("sources", [])
                })
