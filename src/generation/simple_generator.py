"""Simple Generator - Basic RAG generation for Phase 0."""

import asyncio
import os
from dataclasses import dataclass

import anthropic
import structlog

from ..retrieval.simple_retriever import RetrievalResult, SimpleRetriever

logger = structlog.get_logger()


@dataclass
class GenerationResult:
    """Result from generation."""
    query: str
    response: str
    retrieval: RetrievalResult
    model: str
    input_tokens: int
    output_tokens: int


class SimpleGenerator:
    """Simple RAG generator (Phase 0).

    Takes a query, retrieves context, and generates an answer.
    No citation tracking or validation.
    """

    def __init__(
        self,
        retriever: SimpleRetriever,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        max_context_tokens: int = 8000,
    ):
        self.retriever = retriever
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

        self.model = model
        self.max_context_tokens = max_context_tokens
        self.client = anthropic.Anthropic(api_key=self.api_key)

    async def generate(
        self,
        query: str,
        top_k: int = 10,
    ) -> GenerationResult:
        """Generate an answer for a query.

        Args:
            query: User's question
            top_k: Number of chunks to retrieve

        Returns:
            GenerationResult with answer and metadata
        """
        # Retrieve relevant chunks
        retrieval = await self.retriever.retrieve(query=query, top_k=top_k)

        # Format context
        context = self.retriever.format_context(
            results=retrieval.results,
            max_tokens=self.max_context_tokens,
        )

        # Build prompt
        prompt = self._build_prompt(query, context)

        # Generate response
        response = await asyncio.to_thread(
            self.client.messages.create,
            model=self.model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )

        answer = response.content[0].text

        logger.info(
            "generated_response",
            query=query[:50],
            context_chunks=len(retrieval.results),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

        return GenerationResult(
            query=query,
            response=answer,
            retrieval=retrieval,
            model=self.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    def _build_prompt(self, query: str, context: str) -> str:
        """Build the generation prompt."""
        return f"""You are an expert on Ethereum protocol development. Answer the question based on the provided context from Ethereum Improvement Proposals (EIPs).

If the context doesn't contain enough information to answer the question, say so clearly. Don't make up information.

<context>
{context}
</context>

<question>
{query}
</question>

Provide a clear, accurate answer based on the EIP documentation above."""
