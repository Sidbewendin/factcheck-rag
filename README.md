# FactCheck-RAG

> A Retrieval-Augmented Generation (RAG) system for automated COVID-19 claim verification.

---

## Live demo : factcheck-rag.streamlit.app

https://factcheck-rag-qhnsydq4r3frfbyq3vpbcw.streamlit.app/

## Overview

FactCheck-RAG is an end-to-end fact-checking pipeline that verifies COVID-19 claims
against a curated knowledge base of **16,771 verified sources** using state-of-the-art
NLP techniques.

Given a claim, the system:
1. **Retrieves** the most semantically relevant verified sources from a vector database
2. **Augments** the LLM prompt with these sources
3. **Generates** an argued verdict: `TRUE`, `FALSE`, or `PARTIALLY TRUE`

---

## Architecture
User Claim
↓
[RETRIEVAL] BGE Embeddings + ChromaDB
→ Top-k most similar verified chunks
↓
[AUGMENTATION] Prompt Engineering
→ Claim + Verified Sources
↓
[GENERATION] Groq LLM (Llama 3.1)
→ Verdict + Explanation

---

## Project Structure
factcheck-rag/
├── data/
│   ├── covid_tweets.csv       # COVID-19 tweets dataset
│   ├── covid_articles.csv     # COVID-19 articles dataset
│   └── covid_chunks.csv       # Preprocessed and chunked documents
├── src/
│   ├── ingest.py              # Data loading, chunking and ChromaDB indexing
│   ├── retriever.py           # Vector store loading and similarity search
│   └── chain.py               # RAG pipeline and LLM integration
├── app.py                     # Streamlit web interface
├── requirements.txt           # Project dependencies
└── .env.example               # Environment variables template

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Embeddings | `BAAI/bge-small-en-v1.5` |
| Vector Store | `ChromaDB` |
| LLM | `Llama 3.1 8B` via `Groq API` |
| RAG Framework | `LangChain` |
| Interface | `Streamlit` |
| Language | `Python 3.10+` |

---

## Dataset

The knowledge base combines two COVID-19 datasets:

| Dataset | Source | Size |
|---------|--------|------|
| COVID-19 Tweets | `nanyy1025/covid_fake_news` (HuggingFace) | 6,420 tweets |
| COVID-19 Articles | `fake_new_dataset.xlsx` | 3,119 articles |

**Preprocessing pipeline:**
- Removed URLs, mentions, hashtags, special characters
- Concatenated title + text for articles
- Chunking with size=500 and overlap=50
- Filtered only verified documents (label=1) for the knowledge base
- Final knowledge base: **16,771 verified chunks**

---

## Installation

**1. Clone the repository**
```bash
git clone https://github.com/your-username/factcheck-rag.git
cd factcheck-rag
```

**2. Create and activate virtual environment**
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up environment variables**
```bash
cp .env.example .env
# Add your GROQ_API_KEY to .env
```

**5. Build the vector store**
```bash
python src/ingest.py
```

**6. Launch the app**
```bash
streamlit run app.py
```

---

## Usage

1. Enter a COVID-19 claim in the text field
2. Click **Fact-Check**
3. The system retrieves relevant verified sources and generates a verdict

**Example claims:**
- *"The COVID vaccine causes infertility"* → FALSE
- *"COVID-19 is more deadly than the seasonal flu"* → PARTIALLY TRUE
- *"Masks completely prevent COVID transmission"* → FALSE

---

## Technical Choices

**Why BGE-small over all-MiniLM?**
BGE-small ranks higher on the MTEB benchmark while remaining lightweight
enough for CPU inference — the right cost/performance tradeoff for this project.

**Why only label=1 chunks in the knowledge base?**
Indexing only verified documents ensures the LLM generates verdicts grounded
in trustworthy sources, preventing "garbage in, garbage out" retrieval.

**Why chunk_size=500 with overlap=50?**
BGE-small has a 512-token limit. A 500-character chunk stays well within
this limit. The 50-character overlap preserves context across chunk boundaries.

---

## Author

**Sidbewendin Angelique YAMEOGO**  
PhD in Computer Science — NLP & GenAI  
[LinkedIn](https://linkedin.com/in/yameogo-angelique-ph-d-b5227b158)

---