# MetaRAG-LC Documentation

## What is MetaRAG-LC?

MetaRAG-LC is a retrieval-augmented generation system that improves RAG retrieval precision through intelligent metadata engineering.

### Problem It Solves

Traditional RAG systems suffer from:
- **Embedding Pollution**: Irrelevant metadata contaminates embeddings
- **Query-Chunk Misalignment**: Chunks don't align with how users phrase queries
- **Lack of Structured Context**: Chunks lack semantic signals
- **No Explicit Metadata Control**: Everything gets embedded blindly

### The Solution

MetaRAG-LC engineers the representation layer by:
1. Generating LLM-based semantic metadata
2. Controlling what metadata enters embeddings
3. Separating offline indexing from online retrieval
4. Making representation intentional

## Architecture

### Offline Pipeline
- Document Ingestion
- Semantic Chunking
- LLM Metadata Augmentation
- Metadata Control
- Embedding
- Vector Storage

### Online Pipeline
- Query Encoding
- Metadata Filtering
- Similarity Search
- Re-ranking
- LLM Response Generation

## Stack

- **Framework**: LangChain
- **Embeddings**: MiniLM
- **Vector Store**: Chroma
- **LLM**: Ollama + Mistral
