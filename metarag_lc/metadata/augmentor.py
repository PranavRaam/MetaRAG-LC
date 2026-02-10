"""
Metadata Augmentation Module (Metadata Intelligence Engine)

Generates semantic metadata for chunks using Mistral via Ollama.

For each chunk generates:
- Title: Descriptive title
- Questions: 3 example questions the chunk answers
- Summary: Short semantic summary

Why?
Because we want to embed these signals:
  chunk_title + questions + summary + content
Instead of just raw content.

This aligns chunk representation with how users query.
"""

from typing import List, Dict, Any
import logging
import json

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class MetadataAugmentor:
    """
    Augments chunks with LLM-generated semantic metadata.
    
    Uses Mistral via Ollama to generate:
    - Title
    - Example Questions (3)
    - Summary
    """
    
    # Ollama configuration
    OLLAMA_MODEL = "mistral"
    OLLAMA_BASE_URL = "http://localhost:11434"
    TEMPERATURE = 0.2  # Low temperature for consistency, not creativity
    
    def __init__(self, model: str = OLLAMA_MODEL, temperature: float = TEMPERATURE):
        """
        Initialize augmentor with Ollama connection.
        
        Args:
            model: Ollama model name (default: mistral)
            temperature: Temperature for generation (default: 0.2 for consistency)
        """
        self.model = model
        self.temperature = temperature
        
        # Initialize Ollama LLM
        try:
            self.llm = OllamaLLM(
                model=model,
                base_url=self.OLLAMA_BASE_URL,
                temperature=temperature,
            )
            logger.info(f"Connected to Ollama model: {model} (temp={temperature})")
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {str(e)}")
            raise
        
        # Define the structured prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["chunk_content"],
            template="""You are an information extraction system.

Given a document chunk, you MUST generate ALL of the following fields.
You are NOT allowed to return empty fields.
You are NOT allowed to use placeholders like "N/A", "None", or "Untitled".

If information is unclear, generate the BEST possible approximation based on the content.

Return your answer in EXACTLY the following format:

TITLE: <one concise descriptive title>

SUMMARY: <2–3 sentence factual summary>

QUESTIONS:
1. <question one>
2. <question two>
3. <question three>

Do NOT include any extra text.
Do NOT change the format.
Do NOT add explanations.

DOCUMENT:
<<<
{chunk_content}
>>>"""
        )
    
    def augment_chunks(self, chunks: List[Document]) -> List[Document]:
        """
        Augment multiple chunks with metadata.
        
        Args:
            chunks: List of Document objects to augment
            
        Returns:
            List of Documents with added metadata fields
        """
        augmented_chunks = []
        
        for i, chunk in enumerate(chunks, 1):
            try:
                augmented_chunk = self.augment_chunk(chunk)
                augmented_chunks.append(augmented_chunk)
                
                # Progress indicator every 5 chunks
                if i % 5 == 0:
                    logger.info(f"Progress: [{i}/{len(chunks)}] chunks augmented")
                else:
                    logger.debug(f"[{i}/{len(chunks)}] Augmented: {chunk.metadata['chunk_id']}")
            
            except Exception as e:
                logger.error(
                    f"Failed to augment chunk {chunk.metadata.get('chunk_id')}: {str(e)}"
                )
                # Return chunk with minimal metadata if augmentation fails
                augmented_chunks.append(chunk)
        
        return augmented_chunks
    
    def augment_chunks_sample(self, chunks: List[Document], sample_size: int = 10) -> List[Document]:
        """
        Augment a sample of chunks (for testing/demo).
        
        Useful for:
        - Quick testing before full run
        - Limited hardware (16GB RAM, etc.)
        - Verifying metadata quality before scaling
        
        Args:
            chunks: List of Document objects
            sample_size: Number of chunks to sample and augment
            
        Returns:
            List of all Documents, but only sample_size have augmented metadata
        """
        if sample_size >= len(chunks):
            logger.warning(f"Sample size {sample_size} >= total chunks {len(chunks)}")
            return self.augment_chunks(chunks)
        
        # Sample evenly distributed across all chunks
        import math
        step = len(chunks) // sample_size
        sampled_indices = [i * step for i in range(sample_size)]
        
        augmented_chunks = []
        
        for i, chunk in enumerate(chunks):
            if i in sampled_indices:
                try:
                    augmented_chunk = self.augment_chunk(chunk)
                    augmented_chunks.append(augmented_chunk)
                    sample_num = sampled_indices.index(i) + 1
                    logger.info(f"[SAMPLE {sample_num}/{sample_size}] Augmented chunk {i}")
                except Exception as e:
                    logger.error(f"Failed to augment chunk {i}: {str(e)}")
                    augmented_chunks.append(chunk)
            else:
                # Keep chunks without augmentation
                augmented_chunks.append(chunk)
        
        logger.info(f"Augmented {sample_size}/{len(chunks)} sampled chunks")
        return augmented_chunks
    
    def augment_chunk(self, chunk: Document) -> Document:
        """
        Augment a single chunk with LLM-generated metadata.
        
        Args:
            chunk: Document object to augment
            
        Returns:
            Document with added metadata fields (or unchanged if augmentation fails)
        """
        try:
            # Prepare prompt
            prompt = self.prompt_template.format(chunk_content=chunk.page_content)
            
            # Call Mistral
            logger.debug(f"Calling Mistral for chunk: {chunk.metadata.get('chunk_id')}")
            response = self.llm.invoke(prompt)
            
            logger.debug(f"Raw response:\n{response}\n")
            
            # Parse response with STRICT validation
            metadata = self._parse_response(response)
            
            # Enrich chunk metadata
            chunk.metadata['title'] = metadata.get('title', '')
            chunk.metadata['questions'] = metadata.get('questions', [])
            chunk.metadata['summary'] = metadata.get('summary', '')
            chunk.metadata['metadata_augmented'] = True
            
            return chunk
        
        except ValueError as e:
            # Validation failed - log and skip this chunk
            chunk_id = chunk.metadata.get('chunk_id', 'unknown')
            logger.error(f"Validation failed for {chunk_id}:\n{str(e)}")
            logger.error(f"Skipping augmentation for this chunk")
            
            # Mark as failed but don't crash
            chunk.metadata['metadata_augmented'] = False
            chunk.metadata['augmentation_error'] = str(e)
            
            return chunk
        
        except Exception as e:
            # Other errors
            chunk_id = chunk.metadata.get('chunk_id', 'unknown')
            logger.error(f"LLM call failed for {chunk_id}: {str(e)}")
            
            chunk.metadata['metadata_augmented'] = False
            chunk.metadata['augmentation_error'] = str(e)
            
            return chunk
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse structured response from Mistral.
        
        STRICT parsing with validation.
        
        Expected format:
        TITLE: [title]
        
        SUMMARY: [summary]
        
        QUESTIONS:
        1. [q1]
        2. [q2]
        3. [q3]
        
        Args:
            response: Raw response from LLM
            
        Returns:
            Dictionary with parsed fields
            
        Raises:
            ValueError if validation fails
        """
        result = {
            'title': '',
            'questions': [],
            'summary': ''
        }
        
        try:
            lines = response.strip().split('\n')
            
            current_section = None
            current_content = []
            
            for line in lines:
                line_stripped = line.strip()
                
                if line_stripped.startswith('TITLE:'):
                    current_section = 'title'
                    # Extract content after TITLE:
                    after_header = line_stripped[6:].strip()
                    if after_header:
                        current_content = [after_header]
                    else:
                        current_content = []
                
                elif line_stripped.startswith('SUMMARY:'):
                    if current_section == 'title':
                        result['title'] = ' '.join(current_content).strip()
                    current_section = 'summary'
                    # Extract content after SUMMARY:
                    after_header = line_stripped[8:].strip()
                    if after_header:
                        current_content = [after_header]
                    else:
                        current_content = []
                
                elif line_stripped.startswith('QUESTIONS:'):
                    if current_section == 'summary':
                        result['summary'] = ' '.join(current_content).strip()
                    current_section = 'questions'
                    current_content = []
                
                elif line_stripped and current_section:
                    current_content.append(line_stripped)
            
            # Handle last section (QUESTIONS)
            if current_section == 'questions':
                result['questions'] = self._parse_questions(current_content)
            elif current_section == 'summary':
                result['summary'] = ' '.join(current_content).strip()
            elif current_section == 'title':
                result['title'] = ' '.join(current_content).strip()
            
            # STRICT VALIDATION
            self._validate_parsed_result(result)
            
            return result
        
        except ValueError as e:
            # Re-raise validation errors
            raise
        except Exception as e:
            logger.error(f"Error parsing response: {str(e)}")
            logger.debug(f"Response was:\n{response}")
            raise ValueError(f"Failed to parse response: {str(e)}")
    
    def _validate_parsed_result(self, result: Dict[str, Any]) -> None:
        """
        Validate that all fields are present and non-empty.
        
        STRICT validation - fails loudly if:
        - TITLE is empty or "Untitled"
        - SUMMARY is empty or contains "N/A"
        - QUESTIONS count != 3
        
        Args:
            result: Parsed result dictionary
            
        Raises:
            ValueError if validation fails
        """
        errors = []
        
        # Check TITLE
        title = result.get('title', '').strip()
        if not title or title.lower() in ['untitled', 'n/a', 'none', '']:
            errors.append(f"TITLE is empty or invalid: '{title}'")
        
        # Check SUMMARY
        summary = result.get('summary', '').strip()
        if not summary or 'n/a' in summary.lower() or summary.lower() == 'none':
            errors.append(f"SUMMARY is empty or invalid: '{summary}'")
        if len(summary) < 10:  # Too short
            errors.append(f"SUMMARY too short ({len(summary)} chars): '{summary}'")
        
        # Check QUESTIONS
        questions = result.get('questions', [])
        if len(questions) != 3:
            errors.append(f"QUESTIONS count is {len(questions)}, expected 3: {questions}")
        
        for i, q in enumerate(questions, 1):
            q_clean = q.strip().lower()
            if not q or q_clean in ['n/a', 'none', '']:
                errors.append(f"Question {i} is empty or invalid: '{q}'")
        
        if errors:
            error_msg = '\n'.join(errors)
            raise ValueError(f"Validation failed:\n{error_msg}")
    
    
    def _parse_questions(self, lines: List[str]) -> List[str]:
        """
        Parse numbered questions from response lines.
        
        Args:
            lines: List of lines containing questions
            
        Returns:
            List of question strings
        """
        questions = []
        
        for line in lines:
            # Remove numbering (1., 2., 3., etc.)
            line = line.strip()
            if line and line[0].isdigit() and '.' in line:
                # Remove "1. ", "2. ", etc.
                question = line.split('.', 1)[1].strip()
                if question:
                    questions.append(question)
            elif line and not line[0].isdigit():
                # Unnumbered line, could be continuation
                if questions:
                    questions[-1] += ' ' + line
        
        return questions[:3]  # Return max 3 questions
    
    def get_augmentation_stats(self, chunks: List[Document]) -> Dict[str, Any]:
        """
        Get statistics about augmented chunks.
        
        Args:
            chunks: List of Document objects
            
        Returns:
            Dictionary with augmentation statistics
        """
        augmented_count = sum(
            1 for c in chunks if c.metadata.get('metadata_augmented', False)
        )
        
        return {
            'total_chunks': len(chunks),
            'augmented_chunks': augmented_count,
            'augmentation_rate': f"{100 * augmented_count / len(chunks):.1f}%" if chunks else "0%",
        }
