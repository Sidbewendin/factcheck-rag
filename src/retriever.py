"""
retriever.py
------------
Retrieval module for FactCheck-RAG.
Loads the Qdrant Cloud vector store and retrieves
the most relevant verified chunks for a given query.
"""

# IMPORTS

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

load_dotenv()


# CONFIGURATION

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "factcheck_covid"



# LOAD VECTOR STORE

def load_vectorstore() -> QdrantVectorStore:
    """
    Connect to Qdrant Cloud and load the vector store.
    Credentials are loaded from environment variables.

    Returns:
        QdrantVectorStore instance
    """
    print("[INFO] Connecting to Qdrant Cloud...")

    # Load embedding model — must match the one used during ingestion
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # Connect to Qdrant Cloud
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )

    # Load existing collection
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings
    )

    print(f"[INFO] Connected to collection '{COLLECTION_NAME}'")
    return vectorstore


# RETRIEVE RELEVANT CHUNKS

def retrieve(query: str, vectorstore: QdrantVectorStore, k: int = 5) -> list[Document]:
    """
    Retrieve the k most semantically similar chunks
    to the given query from the Qdrant vector store.

    Args:
        query       : the claim or question to fact-check
        vectorstore : QdrantVectorStore instance
        k           : number of chunks to retrieve (default: 5)
    Returns:
        List of the k most relevant Document objects
    """
    print(f"[INFO] Retrieving top {k} chunks for query: '{query}'")

    # Compute query embedding and find nearest neighbors in Qdrant
    docs = vectorstore.similarity_search(query, k=k)

    print(f"[INFO] {len(docs)} relevant chunks retrieved")
    return docs


# MAIN — TEST RETRIEVAL

if __name__ == "__main__":
    print("\n" + "="*55)
    print("   FACTCHECK-RAG — RETRIEVAL MODULE TEST")
    print("="*55 + "\n")

    # Load vector store
    vectorstore = load_vectorstore()

    # Test retrieval
    query = "Does the COVID vaccine cause infertility?"
    docs = retrieve(query, vectorstore)

    print(f"\nTop {len(docs)} results for: '{query}'\n")
    for i, doc in enumerate(docs):
        print(f"--- Result {i+1} ---")
        print(f"Text    : {doc.page_content[:200]}")
        print(f"Metadata: {doc.metadata}\n")