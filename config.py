#!/usr/bin/env python3
"""
Central configuration for MetaRAG-LC.

Keep defaults here and override in main.py if needed.
"""

from pathlib import Path

BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "docs"
PERSIST_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "metarag_lc_offline"

# Offline indexing settings
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
MAX_CHUNKS = 50
MAX_CONTENT_LENGTH = 800

AUGMENT_MODEL = "mistral"
AUGMENT_TEMPERATURE = 0.2

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_BATCH_SIZE = 32
EMBED_DEVICE = None  # None = auto-detect

# Online retrieval settings
RETRIEVAL_TOP_K = 4

# LLM response settings
RESPONDER_MODEL = "mistral"
RESPONDER_TEMPERATURE = 0.2
