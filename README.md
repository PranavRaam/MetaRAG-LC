# MetaRAG-LC: Engineering Retrieval Through Metadata

MetaRAG-LC solves the fundamental problem in RAG systems: **retrieval quality degradation due to weak chunk representation**.

Traditional RAG systems embed raw text chunks, which suffer from:
- **Embedding Pollution**: Irrelevant metadata contaminates embeddings
- **Query-Chunk Misalignment**: Queries are phrased differently than how chunks are stored
- **Lack of Structured Context**: Chunks don't contain semantic signals
- **No Explicit Metadata Control**: Everything gets embedded blindly

## The Solution

MetaRAG-LC improves retrieval precision by **engineering the representation layer**:

1. **LLM-Generated Semantic Metadata**: Each chunk gets a title, example questions, and semantic summary
2. **Controlled Embedding Strategy**: Decide what metadata enters embeddings vs. filters vs. LLM context
3. **Metadata-Aware Retrieval**: Filter and rank based on semantic signals, not raw text similarity
4. **Clean Architecture**: Separate offline indexing from online retrieval

## Architecture

### Offline Indexing Pipeline (Compute-Heavy, Run Once)
```
Document Sources → Ingestion → Semantic Chunking → 
LLM Metadata Augmentation → Metadata Control → 
Embedding Generation → Vector Store
```

### Online Retrieval Pipeline (Real-Time)
```
User Query → Query Encoding → Filter Building → 
Similarity Search → Re-ranking → LLM Response
```

## Stack

- **Framework**: LangChain
- **Embeddings**: MiniLM (sentence-transformers)
- **Vector Store**: Chroma
- **LLM**: Ollama + Mistral 7B

## Development Phases

### Phase 1: Offline Indexing Pipeline
- **Step 1** ✅ Document Ingestion
- **Step 2** Semantic Chunking
- **Step 3** LLM Metadata Augmentation
- **Step 4** Metadata Control
- **Step 5** Embedding & Storage

### Phase 2: Online Retrieval Pipeline
- Query Encoding & Filtering
- Retrieval & Ranking
- LLM Response Generation

### Phase 3: Optimization & Evaluation
- Hybrid retrieval
- Re-ranking strategies
- Quality metrics

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Step 1: Document Ingestion
```bash
python test_step1_ingestion.py
```

This loads documents and extracts metadata:
```
docs/
├── example1.md
└── example2.txt
```

Each document gets:
- `source`: filename
- `file_path`: absolute path
- `file_type`: pdf | txt | md
- `document_id`: unique identifier
- `page_count`: (for PDFs)

## Project Structure

```
metarag_lc/
├── ingestion/        # Document loading & extraction
├── chunking/         # Semantic text splitting
├── metadata/         # LLM augmentation & control
├── embedding/        # Vector generation
├── storage/          # Chroma vector store
├── retrieval/        # Search & ranking
├── generation/       # LLM response
└── pipeline/         # Orchestration (offline/online)
```

## Core Concept

MetaRAG-LC is not about bigger LLMs or better prompts.

It's about **intentional representation engineering**: deciding precisely what signals go into embeddings, how chunks are filtered, and how metadata influences retrieval.

This is how production RAG systems are built.

## Testing

```bash
# Test Step 1 (ingestion)
python test_step1_ingestion.py

# (More tests coming as we build phases)
```

## Author

Built with engineering discipline. No shortcuts.
