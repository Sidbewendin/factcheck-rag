"""
ingest.py
---------
Data ingestion pipeline for FactCheck-RAG.
Loads, cleans, and indexes verified COVID-19 information
into a ChromaDB vector store using BGE embeddings.
"""

# IMPORTS
import os
import pandas as pd
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# CONFIGURATION

DATA_PATH = "data/covid_chunks.csv"
CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


# LOAD DATA
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
    
    print(f"[INFO] Total chunks loaded     : {len(df)}")
    print(f"[INFO] Verified chunks (label=1): {len(df_true)}")
    
    return df_true


# CONVERT TO LANGCHAIN DOCUMENTS

def build_documents(df: pd.DataFrame) -> list[Document]:
    """
    Convert DataFrame rows into LangChain Document objects.
    Each document contains the chunk text and its metadata.
    
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


# LOAD EMBEDDING MODEL

def load_embeddings() -> HuggingFaceEmbeddings:
    """
    Initialize the BGE embedding model.
    BGE-small is chosen for its balance between performance
    and speed on CPU environments.
    normalize_embeddings=True is required for BGE models
    to ensure correct cosine similarity computation.
    
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


# BUILD VECTOR STORE

def build_vectorstore(
    documents: list[Document],
    embeddings: HuggingFaceEmbeddings,
    persist_dir: str
) -> Chroma:
    """
    Vectorize documents using BGE and store them in ChromaDB.
    The vector store is persisted on disk for reuse.
    
    Args:
        documents  : list of LangChain Document objects
        embeddings : BGE embedding model
        persist_dir: directory where ChromaDB will be saved
    Returns:
        Chroma vector store instance
    """
    print(f"[INFO] Indexing {len(documents)} documents into ChromaDB...")
    
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    
    print(f"[INFO] Indexing complete — {vectorstore._collection.count()} vectors stored")
    print(f"[INFO] Vector store saved to '{persist_dir}/'")
    
    return vectorstore


# MAIN — RUN FULL INGESTION PIPELINE

if __name__ == "__main__":
    print("\n" + "="*55)
    print("   FACTCHECK-RAG — DATA INGESTION PIPELINE")
    print("="*55 + "\n")
    
    # Load verified chunks
    df_true = load_true_chunks(DATA_PATH)
    
    # Convert to LangChain documents
    documents = build_documents(df_true)
    
    # Load embedding model
    embeddings = load_embeddings()
    
    # Build and persist vector store
    vectorstore = build_vectorstore(documents, embeddings, CHROMA_DIR)
    
    print("\n[SUCCESS] Ingestion pipeline completed successfully!")