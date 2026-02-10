"""
Embedding Engine Module

Generates vector embeddings for controlled metadata text.

Uses:
- Model: sentence-transformers/all-MiniLM-L6-v2
- Input: Formatted embedding text from MetadataController
- Output: Normalized embeddings

Key Points:
- Embeddings are generated from engineered text (title + questions + summary + content)
- Not from raw text
- This is why retrieval precision improves
"""

from typing import List, Dict, Any
import logging
import numpy as np

from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """
    Generates and manages embeddings using sentence-transformers.
    
    Uses all-MiniLM-L6-v2 model:
    - Lightweight (~80MB)
    - Fast
    - Good semantic quality
    - Dimensions: 384
    """
    
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384
    
    def __init__(self, model_name: str = MODEL_NAME, device: str = None):
        """
        Initialize embedding engine.
        
        Args:
            model_name: HuggingFace model identifier
            device: "cuda", "cpu", or None (auto-detect)
        """
        self.model_name = model_name
        
        # Auto-detect device if not specified
        if device is None:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        
        self.device = device
        
        try:
            logger.info(f"Loading embedding model: {model_name}")
            logger.info(f"Using device: {device}")
            
            self.model = SentenceTransformer(model_name, device=device)
            logger.info(f"✓ Model loaded successfully")
            logger.info(f"✓ Device: {self.model.device}")
            logger.info(f"✓ Embedding dimension: {self.EMBEDDING_DIM}")
        except Exception as e:
            logger.error(f"Failed to load model on {device}: {str(e)}")
            
            # Fallback to CPU if GPU fails
            if device != "cpu":
                logger.warning(f"Falling back to CPU")
                try:
                    self.device = "cpu"
                    self.model = SentenceTransformer(model_name, device="cpu")
                    logger.info(f"✓ Model loaded on CPU (fallback)")
                except Exception as e2:
                    logger.error(f"Failed to load model on CPU: {str(e2)}")
                    raise
            else:
                raise
    
    def embed_chunks(
        self, 
        chunks: List[Document],
        batch_size: int = 32
    ) -> List[Document]:
        """
        Embed multiple chunks in batches.
        
        Args:
            chunks: List of Document objects with page_content as embedding text
            batch_size: Number of texts to embed at once (for memory efficiency)
            
        Returns:
            List of Document objects with embeddings added to metadata
        """
        embedded_chunks = []
        
        # Extract texts to embed
        texts = [chunk.page_content for chunk in chunks]
        
        logger.info(f"Embedding {len(texts)} chunks in batches of {batch_size}...")
        
        # Embed with batching
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalization
        )
        
        # Attach embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk.metadata['embedding'] = embedding.tolist()  # Store as list for JSON serialization
            chunk.metadata['embedding_model'] = self.model_name
            chunk.metadata['embedding_dim'] = self.EMBEDDING_DIM
            chunk.metadata['embedding_normalized'] = True
            
            embedded_chunks.append(chunk)
        
        logger.info(f"✓ Embedded {len(embedded_chunks)} chunks")
        return embedded_chunks
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single text string.
        
        Used for query encoding.
        
        Args:
            text: Text to embed
            
        Returns:
            Normalized embedding vector (numpy array)
        """
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding
    
    def get_engine_stats(self, chunks: List[Document]) -> Dict[str, Any]:
        """
        Get statistics about embeddings.
        
        Args:
            chunks: List of embedded Document objects
            
        Returns:
            Dictionary with embedding statistics
        """
        embedded_count = sum(
            1 for c in chunks if 'embedding' in c.metadata
        )
        
        # Calculate some basic stats if embeddings exist
        if embedded_count > 0:
            embeddings = np.array([
                c.metadata['embedding'] for c in chunks if 'embedding' in c.metadata
            ])
            
            # Check normalization (should be ~1.0 for L2 normalized vectors)
            norms = np.linalg.norm(embeddings, axis=1)
            avg_norm = float(np.mean(norms))
            
            return {
                'total_chunks': len(chunks),
                'embedded_chunks': embedded_count,
                'embedding_rate': f"{100 * embedded_count / len(chunks):.1f}%" if chunks else "0%",
                'embedding_model': self.model_name,
                'embedding_dim': self.EMBEDDING_DIM,
                'device': str(self.model.device),
                'avg_vector_norm': f"{avg_norm:.4f}",  # Should be ~1.0 for normalized
            }
        
        return {
            'total_chunks': len(chunks),
            'embedded_chunks': 0,
            'embedding_rate': "0%",
            'embedding_model': self.model_name,
            'embedding_dim': self.EMBEDDING_DIM,
            'device': str(self.model.device),
        }
    
    def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        For normalized embeddings, this is simply dot product.
        
        Args:
            embedding1: First embedding (as list)
            embedding2: Second embedding (as list)
            
        Returns:
            Similarity score between -1 and 1
        """
        emb1 = np.array(embedding1)
        emb2 = np.array(embedding2)
        
        # Dot product for normalized vectors = cosine similarity
        similarity = np.dot(emb1, emb2)
        
        return float(similarity)
