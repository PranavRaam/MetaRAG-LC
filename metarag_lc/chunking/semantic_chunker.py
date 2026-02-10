"""
Semantic Chunking Module

Splits documents into semantically meaningful chunks using RecursiveCharacterTextSplitter.

Config:
- chunk_size: 800 (balance between fragment vs dilution)
- chunk_overlap: 100 (preserve context across chunks)

Each chunk gets:
- Unique ID (chunk_id)
- Source reference
- Metadata preserved from document
"""

from typing import List, Dict, Any
import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class SemanticChunker:
    """
    Splits documents into semantically meaningful chunks.
    
    Uses RecursiveCharacterTextSplitter which preserves semantic boundaries
    (sentences, paragraphs) rather than naive word/token splitting.
    """
    
    # Configuration
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 100
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        """
        Initialize chunker with configuration.
        
        Args:
            chunk_size: Target size for each chunk (characters)
            chunk_overlap: Characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",      # Paragraph boundaries (highest priority)
                "\n",        # Line breaks
                " ",         # Word boundaries
                ""           # Character boundaries (fallback)
            ],
            length_function=len,
        )
        
        logger.info(
            f"SemanticChunker initialized: "
            f"chunk_size={chunk_size}, overlap={chunk_overlap}"
        )
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into semantic chunks.
        
        Args:
            documents: List of LangChain Document objects from ingestion
            
        Returns:
            List of chunked Document objects with unique IDs and metadata
        """
        chunked_docs = []
        
        for doc in documents:
            chunks = self.chunk_document(doc)
            chunked_docs.extend(chunks)
        
        logger.info(f"Split {len(documents)} documents into {len(chunked_docs)} chunks")
        return chunked_docs
    
    def chunk_document(self, document: Document) -> List[Document]:
        """
        Split a single document into semantic chunks.
        
        Args:
            document: A LangChain Document object
            
        Returns:
            List of Document objects representing chunks
        """
        # Split the document content
        splits = self.splitter.split_documents([document])
        
        # Add unique chunk IDs to each split
        chunks = []
        for i, chunk in enumerate(splits):
            # Preserve original metadata
            chunk_metadata = dict(chunk.metadata)
            
            # Add chunking metadata
            source = chunk_metadata.get('source', 'unknown')
            doc_id = chunk_metadata.get('document_id', 'unknown')
            
            # Create unique chunk ID
            chunk_metadata['chunk_id'] = f"{doc_id}_chunk{i}"
            chunk_metadata['chunk_index'] = i
            chunk_metadata['total_chunks'] = len(splits)
            chunk_metadata['chunk_size'] = len(chunk.page_content)
            
            # Create new document with enriched metadata
            chunk_doc = Document(
                page_content=chunk.page_content,
                metadata=chunk_metadata
            )
            
            chunks.append(chunk_doc)
        
        logger.debug(
            f"Chunked document '{source}' into {len(chunks)} chunks "
            f"(avg size: {sum(c.metadata['chunk_size'] for c in chunks) // len(chunks)})"
        )
        
        return chunks
    
    def get_chunk_stats(self, chunks: List[Document]) -> Dict[str, Any]:
        """
        Get statistics about the chunks.
        
        Args:
            chunks: List of chunked Document objects
            
        Returns:
            Dictionary with chunk statistics
        """
        if not chunks:
            return {
                'total_chunks': 0,
                'avg_chunk_size': 0,
                'min_chunk_size': 0,
                'max_chunk_size': 0,
                'by_source': {}
            }
        
        sizes = [len(c.page_content) for c in chunks]
        by_source = {}
        
        for chunk in chunks:
            source = chunk.metadata.get('source', 'unknown')
            by_source[source] = by_source.get(source, 0) + 1
        
        return {
            'total_chunks': len(chunks),
            'avg_chunk_size': sum(sizes) // len(sizes),
            'min_chunk_size': min(sizes),
            'max_chunk_size': max(sizes),
            'by_source': by_source,
        }
