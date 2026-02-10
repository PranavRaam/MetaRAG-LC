"""
Metadata Controller Module

Controls what metadata goes where:
- EMBED_KEYS: Which fields enter the embedding
- FILTER_KEYS: Which fields are used for metadata filtering
- LLM_CONTEXT_KEYS: Which fields are injected into LLM context

This is the core innovation of MetaRAG-LC: intentional representation engineering.

Instead of embedding everything blindly, we:
1. Select semantic fields for embedding
2. Separate filtering metadata
3. Separate context metadata

Result:
- Clean embedding signal (no pollution)
- Queryable filters
- Traceable context
"""

from typing import List, Dict, Any, Set
import logging

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class MetadataConfig:
    """
    Configuration for metadata routing.
    
    Defines which metadata fields go into:
    - Embeddings
    - Filters  
    - LLM Context
    """
    
    def __init__(
        self,
        embed_keys: Set[str] = None,
        filter_keys: Set[str] = None,
        llm_context_keys: Set[str] = None,
        max_content_length: int = 800,
    ):
        """
        Initialize metadata configuration.
        
        Args:
            embed_keys: Fields to include in embedding text
            filter_keys: Fields to use for filtering
            llm_context_keys: Fields to inject into LLM context
            max_content_length: Max characters for chunk content in embedding text
                               (to prevent dilution of semantic signals)
        """
        # Default configuration
        self.embed_keys = embed_keys or {
            'title',
            'questions',
            'summary',
            'page_content',  # The actual chunk content
        }
        
        self.filter_keys = filter_keys or {
            'source',
            'file_type',
            'file_path',
            'chunk_id',
            'document_id',
        }
        
        self.llm_context_keys = llm_context_keys or {
            'title',
            'source',
            'chunk_id',
        }
        
        self.max_content_length = max_content_length
        
        logger.info(
            f"MetadataConfig initialized:\n"
            f"  EMBED_KEYS: {self.embed_keys}\n"
            f"  FILTER_KEYS: {self.filter_keys}\n"
            f"  LLM_CONTEXT_KEYS: {self.llm_context_keys}\n"
            f"  MAX_CONTENT_LENGTH: {max_content_length} chars"
        )
    
    def validate(self) -> bool:
        """
        Validate that configuration is sensible.
        
        Returns:
            True if valid
        """
        # page_content should be in embed_keys
        if 'page_content' not in self.embed_keys:
            logger.warning("'page_content' not in EMBED_KEYS, embedding will lack main content")
        
        # No key should be in all three
        overlap = self.embed_keys & self.filter_keys & self.llm_context_keys
        if overlap:
            logger.debug(f"Keys in all three categories: {overlap}")
        
        return True


class MetadataController:
    """
    Controls metadata routing and embedding text construction.
    
    Takes augmented chunks and:
    1. Constructs embedding text from selected fields
    2. Preserves filter metadata
    3. Marks LLM context fields
    """
    
    def __init__(self, config: MetadataConfig = None):
        """
        Initialize controller with routing config.
        
        Args:
            config: MetadataConfig object (uses defaults if None)
        """
        self.config = config or MetadataConfig()
        self.config.validate()
    
    def process_chunks(self, chunks: List[Document]) -> List[Document]:
        """
        Process multiple chunks through metadata control.
        
        Args:
            chunks: List of augmented Document objects
            
        Returns:
            List of Document objects with controlled metadata
        """
        processed_chunks = []
        
        for chunk in chunks:
            processed_chunk = self.process_chunk(chunk)
            processed_chunks.append(processed_chunk)
        
        logger.info(f"Processed {len(processed_chunks)} chunks through metadata control")
        return processed_chunks
    
    def process_chunk(self, chunk: Document) -> Document:
        """
        Process a single chunk through metadata control.
        
        Args:
            chunk: Document object with augmented metadata
            
        Returns:
            Document with controlled embedding text and metadata
        """
        if 'original_page_content' not in chunk.metadata:
            chunk.metadata['original_page_content'] = chunk.page_content

        # Step 1: Construct embedding text from selected fields
        embedding_text = self._construct_embedding_text(chunk)
        
        # Step 2: Preserve filter metadata
        filter_metadata = self._extract_filter_metadata(chunk)
        
        # Step 3: Mark LLM context fields
        llm_context_metadata = self._extract_llm_context_metadata(chunk)
        
        # Step 4: Construct new enriched metadata
        enriched_metadata = dict(chunk.metadata)
        enriched_metadata['embedding_text'] = embedding_text
        enriched_metadata['embedding_text_length'] = len(embedding_text)
        enriched_metadata['filter_metadata'] = filter_metadata
        enriched_metadata['llm_context_metadata'] = llm_context_metadata
        enriched_metadata['llm_content'] = enriched_metadata.get('original_page_content', '')
        enriched_metadata['metadata_controlled'] = True
        
        # Create new document with embedding text as page_content
        controlled_chunk = Document(
            page_content=embedding_text,
            metadata=enriched_metadata
        )
        
        return controlled_chunk
    
    def _construct_embedding_text(self, chunk: Document) -> str:
        """
        Construct the text that will be embedded.
        
        Only includes fields in EMBED_KEYS.
        Truncates content to prevent embedding signal dilution.
        
        Format:
        Title: ...
        
        Questions:
        - ...
        - ...
        
        Summary: ...
        
        Content:
        ...
        
        Args:
            chunk: Document object
            
        Returns:
            Formatted embedding text (optimized length)
        """
        parts = []
        metadata = chunk.metadata
        
        # Add TITLE
        if 'title' in self.config.embed_keys:
            title = metadata.get('title', '')
            if title:
                parts.append(f"Title: {title}")
        
        # Add QUESTIONS
        if 'questions' in self.config.embed_keys:
            questions = metadata.get('questions', [])
            if questions:
                q_text = "Questions:\n"
                for q in questions:
                    q_text += f"  - {q}\n"
                parts.append(q_text.strip())
        
        # Add SUMMARY
        if 'summary' in self.config.embed_keys:
            summary = metadata.get('summary', '')
            if summary:
                parts.append(f"Summary: {summary}")
        
        # Add CONTENT (from page_content)
        # OPTIMIZATION: Truncate to prevent signal dilution
        if 'page_content' in self.config.embed_keys:
            # Use original page_content stored in metadata or chunk's original content
            original_content = metadata.get('original_page_content', chunk.page_content)
            
            # Truncate content to max_content_length
            if original_content:
                if len(original_content) > self.config.max_content_length:
                    truncated_content = original_content[:self.config.max_content_length]
                    # Try to truncate at word boundary
                    last_space = truncated_content.rfind(' ')
                    if last_space > self.config.max_content_length * 0.8:
                        truncated_content = truncated_content[:last_space]
                    truncated_content += " [truncated]"
                    content_to_add = truncated_content
                else:
                    content_to_add = original_content
                
                parts.append(f"Content:\n{content_to_add}")
        
        # Join with double newlines for clarity
        embedding_text = "\n\n".join(parts)
        
        logger.debug(
            f"Constructed embedding text for {metadata.get('chunk_id')}: "
            f"{len(embedding_text)} chars (max_content: {self.config.max_content_length})"
        )
        
        return embedding_text
    
    def _extract_filter_metadata(self, chunk: Document) -> Dict[str, Any]:
        """
        Extract metadata fields used for filtering.
        
        Args:
            chunk: Document object
            
        Returns:
            Dictionary with filter metadata only
        """
        filter_meta = {}
        
        for key in self.config.filter_keys:
            if key in chunk.metadata:
                filter_meta[key] = chunk.metadata[key]
        
        return filter_meta
    
    def _extract_llm_context_metadata(self, chunk: Document) -> Dict[str, Any]:
        """
        Extract metadata fields to include in LLM context.
        
        These fields will be injected when constructing prompts.
        
        Args:
            chunk: Document object
            
        Returns:
            Dictionary with LLM context metadata only
        """
        context_meta = {}
        
        for key in self.config.llm_context_keys:
            if key in chunk.metadata:
                context_meta[key] = chunk.metadata[key]
        
        return context_meta
    
    def get_control_stats(self, chunks: List[Document]) -> Dict[str, Any]:
        """
        Get statistics about metadata control.
        
        Args:
            chunks: List of processed Document objects
            
        Returns:
            Dictionary with control statistics
        """
        controlled_count = sum(
            1 for c in chunks if c.metadata.get('metadata_controlled', False)
        )
        
        avg_embedding_text_length = 0
        if chunks:
            total_length = sum(
                c.metadata.get('embedding_text_length', 0) for c in chunks
            )
            avg_embedding_text_length = total_length // len(chunks)
        
        return {
            'total_chunks': len(chunks),
            'controlled_chunks': controlled_count,
            'control_rate': f"{100 * controlled_count / len(chunks):.1f}%" if chunks else "0%",
            'avg_embedding_text_length': avg_embedding_text_length,
            'embed_keys': list(self.config.embed_keys),
            'filter_keys': list(self.config.filter_keys),
            'llm_context_keys': list(self.config.llm_context_keys),
        }
    
    def print_example_chunk(self, chunk: Document) -> None:
        """
        Pretty-print an example of controlled chunk structure.
        
        Args:
            chunk: Processed Document object
        """
        print()
        print("=" * 70)
        print("CONTROLLED CHUNK STRUCTURE")
        print("=" * 70)
        print()
        
        print("1. EMBEDDING TEXT (what gets embedded):")
        print("   [optimized: title + questions + summary + truncated content]")
        print("-" * 70)
        embedding_text = chunk.page_content
        print(embedding_text)
        print()
        print(f"   Length: {len(embedding_text)} chars")
        print()
        
        print("2. FILTER METADATA (for retrieval filtering):")
        print("-" * 70)
        filter_meta = chunk.metadata.get('filter_metadata', {})
        for key, value in filter_meta.items():
            print(f"  {key}: {value}")
        print()
        
        print("3. LLM CONTEXT METADATA (for prompt injection):")
        print("-" * 70)
        llm_meta = chunk.metadata.get('llm_context_metadata', {})
        for key, value in llm_meta.items():
            print(f"  {key}: {value}")
        print()
        
        print("4. ALL METADATA (preserved in Document):")
        print("-" * 70)
        important_keys = [
            'chunk_id', 'source', 'file_type', 'metadata_controlled',
            'metadata_augmented', 'chunk_index', 'title', 'questions',
            'summary', 'embedding_text_length'
        ]
        for key in important_keys:
            if key in chunk.metadata:
                value = chunk.metadata[key]
                if isinstance(value, list):
                    print(f"  {key}: [list of {len(value)} items]")
                elif isinstance(value, str) and len(value) > 50:
                    print(f"  {key}: {value[:50]}...")
                else:
                    print(f"  {key}: {value}")
        print()
