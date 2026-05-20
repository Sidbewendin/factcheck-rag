"""
chain.py
--------
RAG chain module for FactCheck-RAG
"""

# IMPORTS

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from retriever import load_vectorstore, retrieve

# CONFIGURATION

load_dotenv()
LLM_MODEL = "llama-3.1-8b-instant"
DEFAULT_K = 5

# LOAD LLM

def load_llm() -> ChatGroq:
    """
    Initialize the Groq LLM.
    temperature=0 ensures deterministic responses,
    which is critical for fact-checking accuracy.

    Returns:
        ChatGroq LLM instance
    """
    print(f"[INFO] Loading LLM: {LLM_MODEL}")

    llm = ChatGroq(
        model=LLM_MODEL,
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0  # deterministic output for fact-checking
    )

    print("[INFO] LLM loaded successfully")
    return llm

# BUILD PROMPT

def build_prompt(query: str, docs: list) -> str:
    """
    Build the augmented prompt by combining the user claim
    with the retrieved verified sources.

    Args:
        query : the claim to fact-check
        docs  : list of retrieved Document objects
    Returns:
        Complete prompt string for the LLM
    """
    # Concatenate retrieved chunks as numbered sources
    context = "\n\n".join([
        f"Source {i+1}: {doc.page_content}"
        for i, doc in enumerate(docs)
    ])

    prompt = f"""You are a fact-checking assistant specialized in COVID-19 information.
The sources provided below are verified and trustworthy.
Based on these sources, determine if the claim is TRUE, FALSE, or PARTIALLY TRUE.
Always justify your verdict with specific references to the sources.

CLAIM: {query}

VERIFIED SOURCES:
{context}

Provide your verdict in this format:
VERDICT: [TRUE/FALSE/PARTIALLY TRUE]
EXPLANATION: [your explanation based on the sources]
"""
    return prompt

# FULL RAG PIPELINE

def fact_check(query: str, vectorstore: Chroma, llm: ChatGroq, k: int = DEFAULT_K) -> str:
    """
    Full RAG pipeline for fact-checking a claim.
    1. Retrieve relevant verified chunks from ChromaDB
    2. Build augmented prompt with retrieved sources
    3. Generate verdict using Groq LLM

    Args:
        query       : the claim to fact-check
        vectorstore : ChromaDB vector store instance
        llm         : ChatGroq LLM instance
        k           : number of chunks to retrieve
    Returns:
        String containing the verdict and explanation
    """
    # Step 1 — Retrieve relevant chunks
    docs = retrieve(query, vectorstore, k=k)

    # Step 2 — Build augmented prompt
    prompt = build_prompt(query, docs)

    # Step 3 — Generate verdict
    response = llm.invoke(prompt)

    return response.content


# TEST FULL RAG PIPELINE

if __name__ == "__main__":
    print("\n" + "="*55)
    print("   FACTCHECK-RAG — RAG CHAIN TEST")
    print("="*55 + "\n")

    # Load components
    vectorstore = load_vectorstore()
    llm = load_llm()

    # Test claims
    test_claims = [
        "The COVID vaccine causes infertility",
        "COVID-19 is more deadly than the seasonal flu",
        "Masks completely prevent COVID transmission"
    ]

    for claim in test_claims:
        print(f"\nCLAIM: {claim}")
        print("-" * 55)
        result = fact_check(claim, vectorstore, llm)
        print(result)
        print("=" * 55)