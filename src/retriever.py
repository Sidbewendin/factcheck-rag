"""
retriever.py
------------
Retrieval module for FactCheck-RAG.
Loads the ChromaDB vector store and retrieves
the most relevant verified chunks for a given query.
"""

# IMPORTS

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


# CONFIGURATION

CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# LOAD VECTOR STORE

def load_vectorstore() -> Chroma:
    """
    Load the persisted ChromaDB vector store from disk.
    The vector store must have been created by ingest.py first.

    Returns:
        Chroma vector store instance
    """
    print(f"[INFO] Loading vector store from '{CHROMA_DIR}/'...")

    # Load embedding model — must match the one used during ingestion
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # Load existing ChromaDB from disk
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )

    print(f"[INFO] Vector store loaded — {vectorstore._collection.count()} vectors available")
    return vectorstore

# RETRIEVE RELEVANT CHUNKS

def retrieve(query: str, vectorstore: Chroma, k: int = 5) -> list[Document]:
    """
    Retrieve the k most semantically similar chunks
    to the given query from the vector store.

    Args:
        query       : the claim or question to fact-check
        vectorstore : ChromaDB vector store instance
        k           : number of chunks to retrieve (default: 5)
    Returns:
        List of the k most relevant Document objects
    """
    print(f"[INFO] Retrieving top {k} chunks for query: '{query}'")

    # Compute query embedding and find nearest neighbors in ChromaDB
    docs = vectorstore.similarity_search(query, k=k)

    print(f"[INFO] {len(docs)} relevant chunks retrieved")
    return docs


# TEST RETRIEVAL

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