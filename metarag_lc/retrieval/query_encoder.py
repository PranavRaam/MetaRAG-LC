"""
Query Encoder Module (Phase 2 - Step 1)

Encodes user queries using the SAME embedding model as indexing.

Critical: Query embedding must use SAME model as document embeddings.
Otherwise, retrieval will fail.

This module:
1. Encodes raw query text
2. Normalizes embedding (L2)
3. Optionally builds metadata filters
4. Returns query representation ready for retrieval
"""

from typing import Optional, Dict, Any, List
import logging

from metarag_lc.embedding.embedder import EmbeddingEngine

logger = logging.getLogger(__name__)


class QueryEncoder:
    """
    Encodes user queries into embeddings for retrieval.
    
    Uses sentenceTransformers/all-MiniLM-L6-v2 (same as document indexing).
    """
    
    def __init__(self, device: str = None):
        """
        Initialize query encoder with embedding model.
        
        Args:
            device: "cuda", "cpu", or None (auto-detect)
        """
        logger.info("Initializing QueryEncoder")
        
        # Use same embedding engine as documents
        self.embedder = EmbeddingEngine(device=device)
        
        logger.info(f"✓ QueryEncoder ready")
        logger.info(f"  Model: {self.embedder.model_name}")
        logger.info(f"  Device: {self.embedder.device}")
    
    def encode(
        self,
        query: str,
        metadata_filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Encode a user query.
        
        Args:
            query: Raw query text
            metadata_filters: Optional filters for Chroma search
                             Example: {"source": "example.pdf"}
            
        Returns:
            Dictionary with:
            - embedding: normalized query embedding
            - original_query: the original query text
            - filters: metadata filters if provided
        """
        logger.debug(f"Encoding query: {query[:100]}...")
        
        # Embed the query
        embedding = self.embedder.embed_text(query)
        
        # Return query representation
        result = {
            'original_query': query,
            'embedding': embedding.tolist(),  # Convert numpy to list for JSON
            'embedding_dim': len(embedding),
            'embedding_normalized': True,
        }
        
        # Add filters if provided
        if metadata_filters:
            result['filters'] = metadata_filters
            logger.debug(f"Added filters: {metadata_filters}")
        
        return result
    
    def build_filter(
        self,
        source: Optional[str] = None,
        file_type: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build a metadata filter for Chroma.
        
        Supports:
        - source: filename
        - file_type: pdf, txt, md
        - document_id: unique doc identifier
        
        Args:
            source: Optional source file name
            file_type: Optional file type
            document_id: Optional document ID
            
        Returns:
            Filter dict for Chroma where clause
        """
        filters = {}
        
        if source:
            filters['source'] = source
        if file_type:
            filters['file_type'] = file_type
        if document_id:
            filters['document_id'] = document_id
        
        if filters:
            logger.debug(f"Built filter: {filters}")
        
        return filters if filters else None
    
    def encode_with_filter(
        self,
        query: str,
        source: Optional[str] = None,
        file_type: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Convenience method: encode query + build filter in one call.
        
        Args:
            query: Raw query text
            source: Optional source filter
            file_type: Optional file type filter
            document_id: Optional document ID filter
            
        Returns:
            Query representation with embedding and filters
        """
        filters = self.build_filter(source, file_type, document_id)
        return self.encode(query, metadata_filters=filters)
