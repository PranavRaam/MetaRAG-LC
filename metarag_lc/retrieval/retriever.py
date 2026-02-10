#!/usr/bin/env python3
"""
Phase 2 — Step 2: Retriever & Ranking Engine

This module handles pure retrieval:
  1. Accept query embedding
  2. Query vector store
  3. Return ranked results with metadata

NO GENERATION HERE — Just retrieval.
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass


@dataclass
class RetrievalResult:
    """
    Single retrieval result with all necessary context.
    
    Attributes:
        chunk_id: Unique chunk identifier
        similarity_score: Cosine similarity to query (0-1, higher = more relevant)
        embedding_text: Full text that was embedded (for reference)
        content: Clean chunk content for LLM context
        llm_context: LLM context metadata {title, source, chunk_id}
        filter_metadata: Filter metadata {source, file_type, chunk_id, document_id}
        position: Rank position in results (0 = most similar)
    """
    chunk_id: str
    similarity_score: float
    embedding_text: str
    content: str
    llm_context: Dict[str, Any]
    filter_metadata: Dict[str, Any]
    position: int


class Retriever:
    """
    Pure retriever: retrieves and ranks results from indexed chunks.
    
    Design principle:
      - Retrieval only, no generation
      - Returns structured results for downstream LLM processing
      - Supports metadata filtering
    """
    
    def __init__(self, vector_store):
        """
        Initialize retriever with indexed vector store.
        
        Args:
            vector_store: VectorStore instance connected to Chroma collection
        """
        self.vector_store = vector_store
        
    def retrieve(
        self,
        query_embedding: List[float],
        filters: Optional[Dict[str, Any]] = None,
        k: int = 5
    ) -> List[RetrievalResult]:
        """
        Retrieve top-k most similar chunks.
        
        Args:
            query_embedding: 384-dimensional embedding from QueryEncoder
            filters: Optional Chroma where clause from QueryEncoder.build_filter()
            k: Number of results to return (default 5)
        
        Returns:
            List of RetrievalResult objects, sorted by similarity (high → low)
        
        Example:
            >>> results = retriever.retrieve(query_embedding, k=5)
            >>> for result in results:
            ...     print(f"{result.chunk_id}: {result.similarity_score:.3f}")
        """
        
        # Query Chroma
        raw_results = self.vector_store.query_similar(
            query_embedding,
            top_k=k,
            filters=filters
        )
        
        # Convert to structured RetrievalResult objects
        retrieval_results = []
        
        for position, result in enumerate(raw_results):
            # Extract metadata
            metadata = result.get('metadata', {})
            
            # Build LLM context (only keys needed for generation)
            llm_context = {
                'title': metadata.get('title', 'Untitled'),
                'source': metadata.get('source', 'Unknown'),
                'chunk_id': metadata.get('chunk_id', result['chunk_id'])
            }
            
            # Keep all filter metadata for reference
            filter_metadata = {
                'source': metadata.get('source', 'Unknown'),
                'file_type': metadata.get('file_type', 'unknown'),
                'chunk_id': metadata.get('chunk_id', result['chunk_id']),
                'document_id': metadata.get('document_id', 'Unknown')
            }
            
            # Create result object
            retrieval_result = RetrievalResult(
                chunk_id=result['chunk_id'],
                similarity_score=result['similarity'],
                embedding_text=result['document'],
                content=metadata.get('content', ''),
                llm_context=llm_context,
                filter_metadata=filter_metadata,
                position=position
            )
            
            retrieval_results.append(retrieval_result)
        
        return retrieval_results
    
    def retrieve_with_scores(
        self,
        query_embedding: List[float],
        filters: Optional[Dict[str, Any]] = None,
        k: int = 5
    ) -> Dict[str, Any]:
        """
        Retrieve results with detailed scoring information.
        
        Useful for debugging ranking and understanding why results were returned.
        
        Args:
            query_embedding: Query embedding vector
            filters: Optional metadata filters
            k: Number of results
        
        Returns:
            Dictionary with:
                - results: List of RetrievalResult objects
                - total_retrieved: Number of results returned
                - avg_similarity: Average similarity score
                - max_similarity: Highest similarity score
        """
        
        results = self.retrieve(query_embedding, filters=filters, k=k)
        
        if not results:
            return {
                'results': [],
                'total_retrieved': 0,
                'avg_similarity': 0.0,
                'max_similarity': 0.0
            }
        
        similarities = [r.similarity_score for r in results]
        
        return {
            'results': results,
            'total_retrieved': len(results),
            'avg_similarity': sum(similarities) / len(similarities),
            'max_similarity': max(similarities)
        }
    
    def retrieve_batch(
        self,
        query_embeddings: List[List[float]],
        filters: Optional[Dict[str, Any]] = None,
        k: int = 5
    ) -> List[List[RetrievalResult]]:
        """
        Retrieve results for multiple queries.
        
        Args:
            query_embeddings: List of embedding vectors
            filters: Optional metadata filters (applied to all)
            k: Number of results per query
        
        Returns:
            List of result lists (one per query)
        """
        
        batch_results = []
        
        for query_embedding in query_embeddings:
            results = self.retrieve(query_embedding, filters=filters, k=k)
            batch_results.append(results)
        
        return batch_results
    
    def format_results_for_display(
        self,
        results: List[RetrievalResult],
        max_text_length: int = 200
    ) -> str:
        """
        Format retrieval results for display/debugging.
        
        Args:
            results: List of RetrievalResult objects
            max_text_length: Maximum characters to show from embedding_text
        
        Returns:
            Formatted string representation
        """
        
        if not results:
            return "No results retrieved."
        
        lines = []
        lines.append(f"Retrieved {len(results)} results:")
        lines.append("")
        
        for result in results:
            lines.append(f"[{result.position + 1}] {result.chunk_id}")
            lines.append(f"    Similarity: {result.similarity_score:.4f}")
            lines.append(f"    Title: {result.llm_context.get('title', 'N/A')}")
            lines.append(f"    Source: {result.llm_context.get('source', 'N/A')}")
            
            # Truncate embedding text for display
            text_preview = result.embedding_text[:max_text_length]
            if len(result.embedding_text) > max_text_length:
                text_preview += "..."
            lines.append(f"    Text: {text_preview}")
            lines.append("")
        
        return "\n".join(lines)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the retriever and indexed collection.
        
        Returns:
            Dictionary with collection stats
        """
        
        return self.vector_store.get_stats()
