#!/usr/bin/env python3
"""
MetaRAG-LC Complete Workflow Runner

Edit the values in this file to run the full pipeline end-to-end.
"""

import time
from pathlib import Path

from config import (
    DOCS_DIR,
    PERSIST_DIR,
    COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MAX_CHUNKS,
    MAX_CONTENT_LENGTH,
    AUGMENT_MODEL,
    AUGMENT_TEMPERATURE,
    EMBED_MODEL,
    EMBED_BATCH_SIZE,
    EMBED_DEVICE,
    RETRIEVAL_TOP_K,
    RESPONDER_MODEL,
    RESPONDER_TEMPERATURE,
)

from metarag_lc.ingestion.loader import DocumentLoader
from metarag_lc.chunking.semantic_chunker import SemanticChunker
from metarag_lc.metadata.augmentor import MetadataAugmentor
from metarag_lc.metadata.controller import MetadataController, MetadataConfig
from metarag_lc.embedding.embedder import EmbeddingEngine
from metarag_lc.storage.vector_store import VectorStore
from metarag_lc.retrieval.query_encoder import QueryEncoder
from metarag_lc.retrieval.retriever import Retriever
from metarag_lc.generation.responder import Responder


# ------------------------
# USER SETTINGS (EDIT HERE)
# ------------------------
USER_QUERY = "What is CLaRa?"
RUN_OFFLINE_INDEXING = True
RUN_ONLINE_QUERY = True
RESET_COLLECTION = True

# Set to True to skip LLM metadata augmentation (faster, but less semantic)
SKIP_AUGMENTATION = True  # Change to False if Ollama + Mistral are running

# Optional: Override docs directory (defaults to ./docs)
DOCS_PATH_OVERRIDE = None  # Example: Path("D:/MetaRAG-LC/docs")


def run_offline_indexing(docs_path: Path) -> None:
    start_time = time.time()
    
    print("=" * 70)
    print("METARAG-LC OFFLINE INDEXING")
    print("=" * 70)
    print()

    # STEP 1: Document Ingestion
    print("STEP 1: Document Ingestion")
    print("-" * 70)
    loader = DocumentLoader()
    documents = loader.load_directory(str(docs_path))
    print(f"✓ Loaded: {len(documents)} documents")
    print()

    # STEP 2: Semantic Chunking
    print("STEP 2: Semantic Chunking")
    print("-" * 70)
    chunker = SemanticChunker(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    all_chunks = chunker.chunk_documents(documents)
    print(f"✓ Total chunks created: {len(all_chunks)}")

    num_chunks = min(MAX_CHUNKS, len(all_chunks))
    test_chunks = all_chunks[:num_chunks]
    print(f"✓ Using first {num_chunks} chunks for indexing")
    print()

    # STEP 3: LLM Metadata Augmentation
    print("STEP 3: LLM Metadata Augmentation")
    print("-" * 70)
    
    if SKIP_AUGMENTATION:
        print("⚠ SKIPPING augmentation (SKIP_AUGMENTATION=True)")
        print("  Using raw chunks without title/questions/summary")
        augmented_chunks = test_chunks
        # Add minimal metadata for compatibility
        for chunk in augmented_chunks:
            chunk.metadata.setdefault('title', 'Untitled')
            chunk.metadata.setdefault('questions', [])
            chunk.metadata.setdefault('summary', '')
            chunk.metadata['metadata_augmented'] = False
        print(f"✓ Skipped: {num_chunks} chunks (using raw content)")
    else:
        print(f"Augmenting {num_chunks} chunks with {AUGMENT_MODEL}...")
        print("  (This may take several minutes - ensure Ollama is running)")
        augmentor = MetadataAugmentor(model=AUGMENT_MODEL, temperature=AUGMENT_TEMPERATURE)
        augmented_chunks = augmentor.augment_chunks(test_chunks)
        aug_stats = augmentor.get_augmentation_stats(augmented_chunks)
        print(f"✓ Augmented: {aug_stats['augmented_chunks']}/{num_chunks}")
    print()

    # STEP 4: Metadata Control
    print("STEP 4: Metadata Control")
    print("-" * 70)
    config = MetadataConfig(max_content_length=MAX_CONTENT_LENGTH)
    controller = MetadataController(config=config)
    controlled_chunks = controller.process_chunks(augmented_chunks)

    ctrl_stats = controller.get_control_stats(controlled_chunks)
    print(f"✓ Controlled: {ctrl_stats['controlled_chunks']}/{num_chunks}")
    print(f"✓ Avg embedding text length: {ctrl_stats['avg_embedding_text_length']} chars")
    print()

    # STEP 5: Embedding Generation
    print("STEP 5: Embedding Generation")
    print("-" * 70)
    print(f"Embedding {num_chunks} chunks with MiniLM...")
    embedder = EmbeddingEngine(model_name=EMBED_MODEL, device=EMBED_DEVICE)
    embedded_chunks = embedder.embed_chunks(controlled_chunks, batch_size=EMBED_BATCH_SIZE)

    emb_stats = embedder.get_engine_stats(embedded_chunks)
    print(f"✓ Embedded: {emb_stats['embedded_chunks']}/{num_chunks}")
    print(f"✓ Model: {emb_stats['embedding_model']}")
    print(f"✓ Device: {emb_stats['device']}")
    print(f"✓ Dimensions: {emb_stats['embedding_dim']}")
    print(f"✓ Vector norm: {emb_stats['avg_vector_norm']}")
    print()

    # STEP 6: Vector Store (Chroma)
    print("STEP 6: Vector Store (Chroma)")
    print("-" * 70)
    print(f"Persisting {num_chunks} chunks to Chroma...")

    vector_store = VectorStore(
        collection_name=COLLECTION_NAME,
        persist_dir=str(PERSIST_DIR)
    )

    if RESET_COLLECTION:
        vector_store.delete_collection()

    added_count = vector_store.add_chunks(embedded_chunks)

    store_stats = vector_store.get_stats()
    print(f"✓ Added: {added_count} chunks to Chroma")
    print(f"✓ Collection: {store_stats['collection_name']}")
    print(f"✓ Total in store: {store_stats['total_chunks']}")
    print()

    elapsed_time = time.time() - start_time
    print("✓ OFFLINE INDEXING COMPLETE")
    print(f"⏱ Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print()


def run_online_query(user_query: str) -> None:
    total_start = time.time()
    
    print("=" * 70)
    print("METARAG-LC ONLINE RETRIEVAL")
    print("=" * 70)
    print()

    # Initialize components
    vector_store = VectorStore(
        collection_name=COLLECTION_NAME,
        persist_dir=str(PERSIST_DIR)
    )

    stats = vector_store.get_stats()
    if stats.get('total_chunks', 0) == 0:
        print("✗ No chunks indexed. Run offline indexing first.")
        return

    query_encoder = QueryEncoder(device=EMBED_DEVICE)
    retriever = Retriever(vector_store)
    responder = Responder(model=RESPONDER_MODEL, temperature=RESPONDER_TEMPERATURE)

    # Encode query
    print(f"Query: \"{user_query}\"")
    
    encode_start = time.time()
    encoded = query_encoder.encode(user_query)
    encode_time = time.time() - encode_start

    # Retrieve top-k
    retrieval_start = time.time()
    results = retriever.retrieve(encoded['embedding'], k=RETRIEVAL_TOP_K)
    retrieval_time = time.time() - retrieval_start
    print(f"✓ Retrieved {len(results)} chunks (k={RETRIEVAL_TOP_K})")
    print()

    # Show retrieved chunks (titles and sources only)
    for result in results:
        print(f"[{result.position + 1}] {result.chunk_id}")
        print(f"    Title: {result.llm_context.get('title', 'Untitled')}")
        print(f"    Source: {result.llm_context.get('source', 'Unknown')}")
        print()

    # Generate answer
    print("Generating answer with Mistral...")
    generation_start = time.time()
    answer = responder.generate_answer(user_query, results)
    generation_time = time.time() - generation_start
    print()

    print("ANSWER:")
    print("-" * 70)
    print(answer)
    print("-" * 70)
    print()
    
    # Show timing breakdown
    total_time = time.time() - total_start
    print("PERFORMANCE METRICS:")
    print("-" * 70)
    print(f"⏱ Query encoding:     {encode_time:.3f}s")
    print(f"⏱ Retrieval (k={RETRIEVAL_TOP_K}):    {retrieval_time:.3f}s")
    print(f"⏱ Answer generation:  {generation_time:.3f}s")
    print(f"⏱ Total:              {total_time:.3f}s")
    print("-" * 70)
    print()


def main() -> None:
    docs_path = DOCS_PATH_OVERRIDE or DOCS_DIR

    if RUN_OFFLINE_INDEXING:
        run_offline_indexing(docs_path)

    if RUN_ONLINE_QUERY:
        run_online_query(USER_QUERY)


if __name__ == "__main__":
    main()
