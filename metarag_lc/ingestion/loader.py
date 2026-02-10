"""
Document Ingestion Module

Loads documents from various formats (PDF, TXT, MD) using LangChain loaders.
Extracts:
- Raw text content
- Structural metadata (source file, path, page number)

This establishes document identity and traceability before chunking.
"""

from pathlib import Path
from typing import List, Dict, Any
import logging

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader
)
from langchain_core.documents import Document as LangChainDocument

logger = logging.getLogger(__name__)


class DocumentLoader:
    """
    Loads documents from disk using LangChain loaders.
    
    Supports: PDF, TXT, Markdown
    Returns LangChain Document objects with metadata.
    """
    
    SUPPORTED_FORMATS = {'.pdf', '.txt', '.md', '.markdown'}
    
    def __init__(self):
        self.documents: List[LangChainDocument] = []
    
    def load_directory(self, directory_path: str) -> List[LangChainDocument]:
        """
        Load all supported documents from a directory.
        
        Args:
            directory_path: Path to directory containing documents
            
        Returns:
            List of LangChain Document objects with content and metadata
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        if not directory_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory_path}")
        
        self.documents = []
        
        # Recursively find all supported files
        for file_path in directory_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_FORMATS:
                try:
                    docs = self.load_file(str(file_path))
                    if docs:
                        self.documents.extend(docs)
                        logger.info(f"Loaded: {file_path.name} ({len(docs)} docs)")
                except Exception as e:
                    logger.error(f"Failed to load {file_path}: {str(e)}")
        
        logger.info(f"Successfully loaded {len(self.documents)} document chunks")
        return self.documents
    
    def load_file(self, file_path: str) -> List[LangChainDocument]:
        """
        Load a single file using appropriate LangChain loader.
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of LangChain Document objects (PDFs return 1 doc per page)
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_type = file_path.suffix.lower()
        
        if file_type == '.pdf':
            return self._load_pdf(file_path)
        elif file_type == '.txt':
            return self._load_txt(file_path)
        elif file_type in {'.md', '.markdown'}:
            return self._load_markdown(file_path)
        else:
            logger.warning(f"Unsupported file type: {file_type}")
            return []
    
    def _load_pdf(self, file_path: Path) -> List[LangChainDocument]:
        """
        Load PDF using LangChain PyPDFLoader.
        
        Returns one Document per page with page metadata.
        """
        try:
            loader = PyPDFLoader(str(file_path))
            docs = loader.load()
            
            # Enrich metadata
            for doc in docs:
                doc.metadata['source'] = file_path.name
                doc.metadata['file_path'] = str(file_path.absolute())
                doc.metadata['file_type'] = 'pdf'
                page_num = doc.metadata.get('page', 0)
                doc.metadata['document_id'] = f"pdf_{file_path.stem}_page{page_num}"
            
            return docs
        
        except Exception as e:
            logger.error(f"Error reading PDF {file_path.name}: {str(e)}")
            return []
    
    def _load_txt(self, file_path: Path) -> List[LangChainDocument]:
        """Load plain text file using LangChain TextLoader."""
        try:
            loader = TextLoader(str(file_path), encoding='utf-8')
            docs = loader.load()
            
            # Enrich metadata
            for doc in docs:
                doc.metadata['source'] = file_path.name
                doc.metadata['file_path'] = str(file_path.absolute())
                doc.metadata['file_type'] = 'txt'
                doc.metadata['document_id'] = f"txt_{file_path.stem}"
            
            return docs
        
        except UnicodeDecodeError:
            logger.warning(f"UTF-8 decode error, trying latin-1: {file_path.name}")
            loader = TextLoader(str(file_path), encoding='latin-1')
            docs = loader.load()
            
            for doc in docs:
                doc.metadata['source'] = file_path.name
                doc.metadata['file_path'] = str(file_path.absolute())
                doc.metadata['file_type'] = 'txt'
                doc.metadata['document_id'] = f"txt_{file_path.stem}"
            
            return docs
        
        except Exception as e:
            logger.error(f"Error reading TXT {file_path.name}: {str(e)}")
            return []
    
    def _load_markdown(self, file_path: Path) -> List[LangChainDocument]:
        """Load Markdown file using TextLoader (simpler approach)."""
        try:
            loader = TextLoader(str(file_path), encoding='utf-8')
            docs = loader.load()
            
            # Enrich metadata
            for doc in docs:
                doc.metadata['source'] = file_path.name
                doc.metadata['file_path'] = str(file_path.absolute())
                doc.metadata['file_type'] = 'md'
                doc.metadata['document_id'] = f"md_{file_path.stem}"
            
            return docs
        
        except Exception as e:
            logger.error(f"Error reading Markdown {file_path.name}: {str(e)}")
            return []
    
    def get_documents(self) -> List[LangChainDocument]:
        """Return currently loaded documents."""
        return self.documents
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of loaded documents."""
        if not self.documents:
            return {
                'total_documents': 0,
                'total_characters': 0,
                'by_type': {}
            }
        
        by_type = {}
        total_chars = 0
        
        for doc in self.documents:
            file_type = doc.metadata.get('file_type', 'unknown')
            by_type[file_type] = by_type.get(file_type, 0) + 1
            total_chars += len(doc.page_content)
        
        return {
            'total_documents': len(self.documents),
            'total_characters': total_chars,
            'by_type': by_type,
        }
