#!/usr/bin/env python3
"""
Phase 2 — Step 3: LLM Response Layer

Responsibilities:
  1. Accept user query + retrieved chunks
  2. Format chunks for prompt context
  3. Build prompt with hallucination safeguards
  4. Call Mistral
  5. Return generated answer

Key Design Principle:
  - Use ONLY provided context
  - Clearly state if information not in context
  - No hallucination allowed
"""

from typing import List, Dict, Any
from langchain_community.chat_models import ChatOllama


class Responder:
    """
    LLM response generation layer.
    
    Converts retrieved chunks into factual answers using Mistral.
    Designed to prevent hallucination through strict prompt engineering.
    """
    
    PROMPT_TEMPLATE = """You are a factual assistant.

Use ONLY the provided context to answer the question.
If the answer is not contained in the context, say:
"I don't have enough information in the provided context."

Context:
----------------
{formatted_chunks}
----------------

Question:
{user_query}

Answer:"""

    def __init__(self, model: str = "mistral", temperature: float = 0.2):
        """
        Initialize responder with Mistral LLM.
        
        Args:
            model: Ollama model name (default: mistral)
            temperature: Sampling temperature (0.2 = deterministic/factual)
        """
        
        self.model_name = model
        self.temperature = temperature
        
        # Initialize ChatOllama
        self.llm = ChatOllama(
            model=model,
            temperature=temperature
        )
    
    def generate_answer(
        self,
        user_query: str,
        retrieved_chunks: List[Any]  # List of RetrievalResult objects
    ) -> str:
        """
        Generate answer from retrieved chunks.
        
        Args:
            user_query: User's question
            retrieved_chunks: List of RetrievalResult objects from Retriever
        
        Returns:
            Generated answer string
        
        Example:
            >>> answer = responder.generate_answer(
            ...     "What is CLaRa?",
            ...     retrieval_results
            ... )
            >>> print(answer)
            "CLaRa is a method that..."
        """
        
        # Format chunks for context
        formatted_context = self._format_chunks(retrieved_chunks)
        
        # Build complete prompt
        prompt_text = self._build_prompt(user_query, formatted_context)
        
        # Call Mistral
        response = self.llm.invoke(prompt_text)
        
        # Extract and return answer
        answer = response.content.strip()
        
        return answer
    
    def _format_chunks(self, retrieved_chunks: List[Any]) -> str:
        """
        Format retrieved chunks into clean context.
        
        Args:
            retrieved_chunks: List of RetrievalResult objects
        
        Returns:
            Formatted context string
        
        Format:
            [Chunk 1]
            Title: ...
            Source: ...
            Content:
            ...
            
            [Chunk 2]
            Title: ...
            Source: ...
            Content:
            ...
        """
        
        if not retrieved_chunks:
            return "No context provided."
        
        formatted_parts = []
        
        for i, chunk in enumerate(retrieved_chunks, 1):
            # Extract clean metadata
            title = chunk.llm_context.get('title', 'Untitled')
            source = chunk.llm_context.get('source', 'Unknown')
            
            content = chunk.content
            
            # Build chunk block
            chunk_block = f"""[Chunk {i}]
Title: {title}
Source: {source}
Content:
{content}"""
            
            formatted_parts.append(chunk_block)
        
        # Join with blank lines
        return "\n\n".join(formatted_parts)
    
    def _build_prompt(self, user_query: str, formatted_context: str) -> str:
        """
        Build complete prompt for Mistral.
        
        Args:
            user_query: User's question
            formatted_context: Formatted chunk context
        
        Returns:
            Complete prompt string
        """
        
        prompt = self.PROMPT_TEMPLATE.format(
            formatted_chunks=formatted_context,
            user_query=user_query
        )
        
        return prompt
    
    def generate_answer_with_metadata(
        self,
        user_query: str,
        retrieved_chunks: List[Any],
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Generate answer and include metadata about sources.
        
        Useful for showing user which documents the answer came from.
        
        Args:
            user_query: User's question
            retrieved_chunks: Retrieved chunks with metadata
            include_sources: Whether to include source list
        
        Returns:
            Dictionary with:
                - answer: Generated answer text
                - sources: List of {title, source, chunk_id} if include_sources=True
                - chunks_used: Number of chunks in context
        """
        
        # Generate answer
        answer = self.generate_answer(user_query, retrieved_chunks)
        
        # Compile sources
        sources = []
        if include_sources:
            seen = set()
            for chunk in retrieved_chunks:
                source_key = (
                    chunk.llm_context.get('title'),
                    chunk.llm_context.get('source')
                )
                if source_key not in seen:
                    sources.append({
                        'title': chunk.llm_context.get('title', 'Untitled'),
                        'source': chunk.llm_context.get('source', 'Unknown'),
                        'chunk_id': chunk.chunk_id
                    })
                    seen.add(source_key)
        
        return {
            'answer': answer,
            'sources': sources,
            'chunks_used': len(retrieved_chunks)
        }
    
    def generate_batch_answers(
        self,
        queries: List[str],
        retrieved_chunks_list: List[List[Any]]
    ) -> List[str]:
        """
        Generate answers for multiple queries.
        
        Args:
            queries: List of user queries
            retrieved_chunks_list: List of retrieved chunk lists (one per query)
        
        Returns:
            List of generated answers
        """
        
        answers = []
        
        for query, chunks in zip(queries, retrieved_chunks_list):
            answer = self.generate_answer(query, chunks)
            answers.append(answer)
        
        return answers
    
    def get_config(self) -> Dict[str, Any]:
        """Get responder configuration."""
        
        return {
            'model': self.model_name,
            'temperature': self.temperature,
            'prompt_template': 'Factual assistant using only provided context'
        }
