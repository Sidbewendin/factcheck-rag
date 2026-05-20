"""
ingest.py
---------
Data ingestion pipeline for FactCheck-RAG.
Loads, cleans, and indexes verified COVID-19 information
into a Qdrant Cloud vector store using BGE embeddings.
"""

# IMPORTS

import os
import pandas as pd
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()


# CONFIGURATION

DATA_PATH = "data/covid_chunks.csv"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "factcheck_covid"


# STEP 1 — LOAD DATA

def load_true_chunks(data_path: str) -> pd.DataFrame:
    """
    Load pre-processed chunks from CSV and filter
    only verified (label=1) documents.

    Args:
        data_path: path to the chunks CSV file
    Returns:
        DataFrame containing only verified chunks
    """
    df = pd.read_csv(data_path)

    # Drop rows with missing text
    df = df.dropna(subset=['text'])

    # Keep only verified chunks (label=1) as our knowledge base
    df_true = df[df['label'] == 1].reset_index(drop=True)

    print(f"[INFO] Total chunks loaded      : {len(df)}")
    print(f"[INFO] Verified chunks (label=1): {len(df_true)}")

    return df_true


# STEP 2 — CONVERT TO LANGCHAIN DOCUMENTS

def build_documents(df: pd.DataFrame) -> list[Document]:
    """
    Convert DataFrame rows into LangChain Document objects.

    Args:
        df: DataFrame with 'text', 'label', 'source_idx' columns
    Returns:
        List of LangChain Document objects
    """
    documents = [
        Document(
            page_content=row['text'],
            metadata={
                'label': int(row['label']),           # 1 = verified true
                'source_idx': int(row['source_idx'])  # original document index
            }
        )
        for _, row in df.iterrows()
    ]

    print(f"[INFO] {len(documents)} LangChain documents created")
    return documents


# STEP 3 — LOAD EMBEDDING MODEL

def load_embeddings() -> HuggingFaceEmbeddings:
    """
    Initialize the BGE embedding model.
    normalize_embeddings=True is required for correct
    cosine similarity computation with BGE models.

    Returns:
        HuggingFaceEmbeddings instance
    """
    print(f"[INFO] Loading embedding model: {EMBEDDING_MODEL}")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    print("[INFO] Embedding model loaded successfully")
    return embeddings


# STEP 4 — BUILD VECTOR STORE

def build_vectorstore(
    documents: list[Document],
    embeddings: HuggingFaceEmbeddings
) -> QdrantVectorStore:
    """
    Vectorize documents using BGE and store them in Qdrant Cloud.

    Args:
        documents  : list of LangChain Document objects
        embeddings : BGE embedding model
    Returns:
        QdrantVectorStore instance
    """
    print(f"[INFO] Indexing {len(documents)} documents into Qdrant Cloud...")

    vectorstore = QdrantVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        collection_name=COLLECTION_NAME
    )

    print(f"[INFO] Indexing complete — {len(documents)} vectors stored")
    print(f"[INFO] Collection name: '{COLLECTION_NAME}'")

    return vectorstore


# MAIN — RUN FULL INGESTION PIPELINE

if __name__ == "__main__":
    print("\n" + "="*55)
    print("   FACTCHECK-RAG — DATA INGESTION PIPELINE")
    print("="*55 + "\n")

    # Step 1 — Load verified chunks
    df_true = load_true_chunks(DATA_PATH)

    # Step 2 — Convert to LangChain documents
    documents = build_documents(df_true)

    # Step 3 — Load embedding model
    embeddings = load_embeddings()

    # Step 4 — Build and persist vector store in Qdrant Cloud
    vectorstore = build_vectorstore(documents, embeddings)

    print("\n[SUCCESS] Ingestion pipeline completed successfully!")