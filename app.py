"""
app.py
------
Streamlit web interface for FactCheck-RAG.
Provides an interactive UI for fact-checking
COVID-19 claims using the RAG pipeline.
"""

# ============================================================
# IMPORTS
# ============================================================
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
from chain import fact_check, load_llm
from retriever import load_vectorstore


# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="FactCheck-RAG",
    page_icon="🔍",
    layout="centered"
)


# ============================================================
# LOAD COMPONENTS (cached to avoid reloading on every run)
# ============================================================
@st.cache_resource
def load_components():
    """
    Load and cache the vector store and LLM.
    @st.cache_resource ensures these are loaded only once
    and reused across all user interactions.
    """
    vectorstore = load_vectorstore()
    llm = load_llm()
    return vectorstore, llm


# ============================================================
# UI — HEADER
# ============================================================
st.title("🔍 FactCheck-RAG")
st.subheader("COVID-19 Claim Verification powered by RAG & LLM")
st.markdown("""
This tool verifies COVID-19 claims against a database of
**16,771 verified sources** using Retrieval-Augmented Generation (RAG).
""")
st.divider()


# ============================================================
# UI — INPUT
# ============================================================
st.markdown("### Enter a claim to fact-check")

examples = [
    "The COVID vaccine causes infertility",
    "COVID-19 is more deadly than the seasonal flu",
    "Masks completely prevent COVID transmission",
    "Hydroxychloroquine cures COVID-19"
]

selected_example = st.selectbox(
    "Or choose an example claim:",
    [""] + examples
)

user_input = st.text_area(
    "Your claim:",
    value=selected_example,
    height=100,
    placeholder="e.g. The COVID vaccine causes infertility"
)


# ============================================================
# UI — FACT-CHECK BUTTON
# ============================================================
if st.button("🔍 Fact-Check", type="primary", use_container_width=True):

    if not user_input.strip():
        st.warning("Please enter a claim to fact-check.")
    else:
        # Load components
        with st.spinner("Connecting to knowledge base..."):
            vectorstore, llm = load_components()

        # Run RAG pipeline
        with st.spinner("Analyzing claim against verified sources..."):
            result = fact_check(user_input, vectorstore, llm)

        # --------------------------------------------------------
        # UI — DISPLAY RESULT
        # --------------------------------------------------------
        st.divider()
        st.markdown("### Fact-Check Result")

        # Parse verdict from result
        if "VERDICT: TRUE" in result and "PARTIALLY" not in result:
            verdict = "TRUE"
            color = "green"
        elif "VERDICT: FALSE" in result:
            verdict = "FALSE"
            color = "red"
        elif "VERDICT: PARTIALLY TRUE" in result:
            verdict = "PARTIALLY TRUE"
            color = "orange"
        else:
            verdict = "UNCERTAIN"
            color = "gray"

        # Display verdict badge
        st.markdown(
            f"<h2 style='color:{color};'>{verdict}</h2>",
            unsafe_allow_html=True
        )

        # Display full explanation
        st.markdown("#### Explanation")
        explanation = result.split("EXPLANATION:")[-1].strip()
        st.write(explanation)


# ============================================================
# UI — FOOTER
# ============================================================
st.divider()
st.markdown("""
<small>
Built with LangChain · Qdrant Cloud · BGE Embeddings · Groq LLM · Streamlit  
Knowledge base: 16,771 verified COVID-19 sources
</small>
""", unsafe_allow_html=True)