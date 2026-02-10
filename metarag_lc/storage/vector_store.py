"""
Vector Store Module (Chroma)

Persists embeddings + controlled metadata to Chroma vector database.

Key Design:
- Embeddings are pre-computed (from EmbeddingEngine)
- Chroma stores: ids, embeddings, metadatas, documents
- Metadata is selective (filters only, not verbose)
- No automatic re-embedding (we control embedding)

This is where the indexed knowledge base is created.
"""

from typing import List, Dict, Any, Optional
import logging
import os

import chromadb
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Manages embedding storage and retrieval using Chroma.
    
    Stores:
    - Pre-computed embeddings (384-dim from MiniLM)
    - Filter metadata (source, chunk_id, etc.)
    - Embedding text (for reference)
    """
    
    def __init__(
        self,
        collection_name: str = "metarag_lc",
        persist_dir: str = "./chroma_db",
    ):
        """
        Initialize vector store.
        
        Args:
            collection_name: Name of Chroma collection
            persist_dir: Directory to persist Chroma data
        """
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        
        # Create persist directory if needed
        os.makedirs(persist_dir, exist_ok=True)
        
        logger.info(f"Initializing Chroma vector store")
        logger.info(f"  Collection: {collection_name}")
        logger.info(f"  Persist dir: {persist_dir}")
        
        try:
            # Initialize Chroma with new PersistentClient API
            self.client = chromadb.PersistentClient(path=persist_dir)
            
            # Get or create collection
            # NOTE: We use embedding_function=None because we provide embeddings manually
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=None,  # Manual embeddings only
                metadata={"hnsw:space": "cosine"}  # Cosine distance for normalized vectors
            )
            
            logger.info(f"✓ Chroma collection ready: {collection_name}")
        
        except Exception as e:
            logger.error(f"Failed to initialize Chroma: {str(e)}")
            raise
    
    def add_chunks(self, chunks: List[Document]) -> int:
        """
        Add embedded chunks to vector store.
        
        Args:
            chunks: List of Document objects with embeddings
            
        Returns:
            Number of chunks added
        """
        if not chunks:
            logger.warning("No chunks to add")
            return 0
        
        # Prepare data for Chroma
        ids = []
        embeddings = []
        metadatas = []
        documents = []
        
        for chunk in chunks:
            # Skip if no embedding
            if 'embedding' not in chunk.metadata:
                logger.warning(f"Skipping {chunk.metadata.get('chunk_id')}: no embedding")
                continue
            
            chunk_id = chunk.metadata.get('chunk_id', 'unknown')
            
            # Extract metadata for Chroma (selective, not verbose)
            llm_content = chunk.metadata.get('llm_content')
            if not llm_content:
                llm_content = chunk.metadata.get('original_page_content', '')

            chroma_metadata = {
                'chunk_id': chunk_id,
                'source': chunk.metadata.get('source', 'unknown'),
                'file_type': chunk.metadata.get('file_type', 'unknown'),
                'document_id': chunk.metadata.get('document_id', 'unknown'),
                'chunk_index': str(chunk.metadata.get('chunk_index', 0)),
                'title': chunk.metadata.get('title', 'Untitled')[:100],  # Truncate for storage
                'content': llm_content,
            }
            
            ids.append(chunk_id)
            embeddings.append(chunk.metadata['embedding'])
            metadatas.append(chroma_metadata)
            documents.append(chunk.page_content)  # Embedding text
        
        if not ids:
            logger.warning("No chunks had embeddings to add")
            return 0
        
        # Add to Chroma
        try:
            logger.info(f"Adding {len(ids)} chunks to Chroma...")
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents,
            )
            
            logger.info(f"✓ Added {len(ids)} chunks to collection")
            return len(ids)
        
        except Exception as e:
            logger.error(f"Failed to add chunks: {str(e)}")
            raise
    
    def query_similar(
        self,
        query_embedding: List[float],
        top_k: int = 3,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query similar chunks.
        
        Args:
            query_embedding: Query embedding vector (384-dim)
            top_k: Number of results to return
            filters: Optional metadata filters (Chroma where clause)
            
        Returns:
            List of similar chunks with metadata and distance
        """
        try:
            where = filters if filters else None
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where,
                include=["embeddings", "documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            
            if results and results["ids"] and len(results["ids"]) > 0:
                for i, chunk_id in enumerate(results["ids"][0]):
                    formatted_results.append({
                        'chunk_id': chunk_id,
                        'document': results["documents"][0][i],
                        'metadata': results["metadatas"][0][i],
                        'distance': results["distances"][0][i],
                        'similarity': 1 - results["distances"][0][i],  # Convert to similarity
                    })
            
            return formatted_results
        
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get collection statistics.
        
        Returns:
            Dictionary with collection stats
        """
        try:
            count = self.collection.count()
            
            return {
                'collection_name': self.collection_name,
                'total_chunks': count,
                'persist_dir': self.persist_dir,
            }
        
        except Exception as e:
            logger.error(f"Failed to get stats: {str(e)}")
            return {}
    
    def delete_collection(self) -> None:
        """
        Delete the collection and recreate it (for testing/reset).
        
        Use with caution!
        """
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Deleted collection {self.collection_name}")
            
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=None,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"✓ Collection {self.collection_name} reset")
        except Exception as e:
            logger.error(f"Failed to reset collection: {str(e)}")
            raise
    
    def persist(self) -> None:
        """
        Explicitly persist the collection to disk.
        
        Note: PersistentClient auto-persists, but calling this ensures
        any pending operations are flushed.
        """
        try:
            logger.info("Persisting collection...")
            # PersistentClient automatically persists
            # This is a no-op but keeps the interface consistent
            logger.info("✓ Collection persisted")
        except Exception as e:
            logger.error(f"Failed to persist: {str(e)}")
            raise
