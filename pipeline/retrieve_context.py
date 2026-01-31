# pipeline/retrieve_context.py
"""
Context retrieval module using RAG (Retrieval-Augmented Generation).

Retrieves relevant knowledge base documents for trope identification.
"""

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Lazy loading - only initialize when first called
_index = None
_retriever = None


def _initialize_retriever():
    """Initialize the retriever lazily (only when first needed)."""
    global _index, _retriever
    
    if _retriever is not None:
        return _retriever
    
    # Local embedding model (no API key, no 429)
    print("Loading embedding model and building index...")
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Load KB
    documents = SimpleDirectoryReader("kb").load_data()
    
    # Build vector index
    _index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
    # Increase top_k to get more relevant context
    _retriever = _index.as_retriever(similarity_top_k=4)
    
    print("✓ Index ready!")
    return _retriever


def retrieve_context(query: str):
    """Retrieve relevant knowledge base documents using RAG (Retrieval-Augmented Generation).

    Parameters
    ----------
    query
        Query text to search for in knowledge base.

    Returns
    -------
    list
        List of retrieved document objects from knowledge base, ranked by relevance.
    """
    retriever = _initialize_retriever()
    return retriever.retrieve(query)
