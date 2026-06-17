import os

import streamlit as st

from ingest import load_and_index_pdfs
from rag_pipeline import (
    generate_paper_brief,
    get_answer_with_sources,
    get_indexed_paper_count,
)

# ─── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="Research Paper Intelligence Assistant",
    page_icon="📚",
    layout="wide",
)

# ─── Custom CSS ─────────────────────────────────────────────
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a5276;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .answer-box {
        background-color: #eaf4fb;
        border-left: 5px solid #1a5276;
        padding: 1.2rem;
        border-radius: 6px;
        margin: 1rem 0;
    }
    .source-tag {
        background-color: #1a5276;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .page-tag {
        background-color: #27ae60;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 6px;
    }
    /* Paper brief section cards */
    .brief-card {
        border-left: 5px solid #1a5276;
        padding: 0.5rem 0 0.5rem 0.75rem;
        margin-bottom: 0.25rem;
    }
    .brief-card-novelty   { border-left-color: #1a5276; }
    .brief-card-findings  { border-left-color: #27ae60; }
    .brief-card-limits    { border-left-color: #e67e22; }
</style>
""",
    unsafe_allow_html=True,
)

# ─── Session state defaults ───────────────────────────────────
if "indexed" not in st.session_state:
    st.session_state["indexed"] = False
if "filenames" not in st.session_state:
    st.session_state["filenames"] = []

# ─── Header ─────────────────────────────────────────────────
st.markdown(
    '<p class="main-header">📚 Research Paper Intelligence Assistant</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="sub-header">Upload any academic paper — get instant '
    "structured analysis, cited answers, and cross-paper comparison.</p>",
    unsafe_allow_html=True,
)
st.divider()

# ─── Sidebar: PDF Upload ─────────────────────────────────────
with st.sidebar:
    st.header("📂 Upload Research Papers")
    st.caption("Supports multiple PDFs from any academic domain")

    uploaded_files = st.file_uploader(
        "Drop PDFs here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        if st.button("🔄 Index Documents", use_container_width=True, type="primary"):
            os.makedirs("data", exist_ok=True)

            for f in os.listdir("data"):
                os.remove(os.path.join("data", f))

            for uploaded_file in uploaded_files:
                file_path = os.path.join("data", uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

            with st.spinner("Reading and indexing your papers..."):
                vectorstore = load_and_index_pdfs("data/")

            if vectorstore:
                st.success(f"✅ Indexed {len(uploaded_files)} paper(s) successfully!")
                st.session_state["indexed"] = True
                st.session_state["filenames"] = [f.name for f in uploaded_files]
            else:
                st.error("Something went wrong. Please try again.")

    if st.session_state.get("filenames"):
        st.divider()
        st.markdown("**📄 Indexed Papers:**")
        for name in st.session_state["filenames"]:
            st.markdown(f"- {name}")

    if not st.session_state.get("indexed") and os.path.exists("faiss_index"):
        st.session_state["indexed"] = True
        existing = os.listdir("data") if os.path.exists("data") else []
        st.session_state["filenames"] = [f for f in existing if f.endswith(".pdf")]

    st.divider()
    st.markdown("**💡 Example Questions:**")
    st.caption("What is the main research problem?")
    st.caption("What methods or models were used?")
    st.caption("What are the key findings and results?")
    st.caption("What limitations do the authors acknowledge?")
    st.caption("What future work do the authors recommend?")


# ─── Helper: render answer + sources ─────────────────────────
def _render_answer(result: dict):
    """Display an answer box and expandable source citations."""
    st.markdown("### 💬 Answer")
    st.markdown(
        f'<div class="answer-box">{result["answer"]}</div>',
        unsafe_allow_html=True,
    )

    if result["sources"]:
        st.markdown("### 📎 Sources")
        st.caption("Exact sections of your papers used to generate this answer.")
        for i, source in enumerate(result["sources"], 1):
            with st.expander(
                f"Source {i} — {source['file']} | Page {source['page']}",
                expanded=(i == 1),
            ):
                st.markdown(
                    f'<span class="source-tag">📄 {source["file"]}</span>'
                    f'<span class="page-tag">Page {source["page"]}</span>',
                    unsafe_allow_html=True,
                )
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"*{source['snippet']}*")


# ─── Tab layout ───────────────────────────────────────────────
paper_count = len(st.session_state.get("filenames", [])) or get_indexed_paper_count()
tab_labels = ["💬 Ask Questions", "📋 Paper Brief"]
if paper_count >= 2:
    tab_labels.append("⚖️ Compare Papers")

tabs = st.tabs(tab_labels)
tab_ask = tabs[0]
tab_brief = tabs[1]
tab_compare = tabs[2] if paper_count >= 2 else None

# ═══════════════════════════════════════════════════════════════
# TAB 1 — Ask Questions (existing functionality)
# ═══════════════════════════════════════════════════════════════
with tab_ask:
    if not st.session_state.get("indexed"):
        st.info(
            "👈 Start by uploading your research PDFs in the sidebar, "
            "then ask any question about them."
        )
    else:
        col1, col2 = st.columns([3, 1])
        with col1:
            question = st.text_input(
                "Ask a question about your uploaded papers:",
                placeholder="e.g. What models were used and what accuracy was achieved?",
                key="question_input",
            )
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            ask_button = st.button("🔍 Ask", use_container_width=True, type="primary")

        if ask_button and question:
            with st.spinner("Searching through your papers..."):
                result = get_answer_with_sources(question)
            _render_answer(result)

        elif not ask_button:
            st.markdown("### 🔬 Ready to answer questions")
            st.markdown(
                "Your papers are indexed. Type a question above and hit **Ask**."
            )
            st.markdown("**Try one of these:**")
            sample_questions = [
                "What is the main research problem addressed?",
                "What methods or models were used?",
                "What are the key findings and results?",
                "What limitations do the authors acknowledge?",
            ]
            cols = st.columns(2)
            for i, q in enumerate(sample_questions):
                with cols[i % 2]:
                    if st.button(q, use_container_width=True, key=f"sample_{i}"):
                        st.session_state["question_input"] = q
                        st.rerun()

# ═══════════════════════════════════════════════════════════════
# TAB 2 — Paper Brief
# ═══════════════════════════════════════════════════════════════
with tab_brief:
    if not st.session_state.get("indexed"):
        st.info("👈 Upload and index papers in the sidebar to generate a paper brief.")
    else:
        st.markdown(
            "### 📋 Structured Paper Analysis",
            help="Automatically extracts 7 key sections from your indexed papers.",
        )

        if st.button("Generate Paper Brief", type="primary"):
            with st.spinner("Analyzing paper structure..."):
                brief = generate_paper_brief()

            # Section display config: (label, dict key, CSS class for left border)
            sections = [
                ("🔍 Research Problem", "research_problem", "brief-card"),
                ("✨ Novelty", "novelty", "brief-card brief-card-novelty"),
                ("🔬 Methodology", "methodology", "brief-card"),
                ("📊 Key Findings", "key_findings", "brief-card brief-card-findings"),
                ("⚠️ Limitations", "limitations", "brief-card brief-card-limits"),
                ("🔮 Future Work", "future_work", "brief-card"),
                ("🌍 Real World Impact", "real_world_impact", "brief-card"),
            ]

            for label, key, css_class in sections:
                with st.expander(label, expanded=True):
                    st.markdown(
                        f'<div class="{css_class}">{brief.get(key, "Not explicitly mentioned")}</div>',
                        unsafe_allow_html=True,
                    )

# ═══════════════════════════════════════════════════════════════
# TAB 3 — Compare Papers (only when 2+ PDFs indexed)
# ═══════════════════════════════════════════════════════════════
if tab_compare is not None:
    with tab_compare:
        if not st.session_state.get("indexed"):
            st.info("👈 Upload and index at least two papers to use comparison.")
        else:
            st.markdown("### ⚖️ Cross-Paper Comparison")
            st.caption(
                f"Comparing across {paper_count} indexed papers. "
                "Answers will reference which paper says what."
            )

            compare_question = st.text_input(
                "What aspect do you want to compare?",
                placeholder="e.g. Which paper's methodology is more robust?",
                key="compare_input",
            )

            if st.button("⚖️ Compare", type="primary", use_container_width=False):
                if compare_question:
                    prefixed = (
                        f"Comparing across all uploaded papers: {compare_question}. "
                        "Explicitly reference which paper says what."
                    )
                    with st.spinner("Comparing papers..."):
                        result = get_answer_with_sources(prefixed)
                    _render_answer(result)
                else:
                    st.warning("Please enter a comparison question.")
