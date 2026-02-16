"""Simple Generator - Basic RAG generation for Phase 0."""

from dataclasses import dataclass

import structlog

from ..config import DEFAULT_MODEL
from ..retrieval.simple_retriever import RetrievalResult, SimpleRetriever
from .completion import call_llm

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
        model: str = DEFAULT_MODEL,
        max_context_tokens: int = 8000,
        max_tokens: int = 2048,
    ):
        self.retriever = retriever
        self.model = model
        self.max_context_tokens = max_context_tokens
        self.max_tokens = max_tokens

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

        result = await call_llm(self.model, [{"role": "user", "content": prompt}], self.max_tokens)

        logger.info(
            "generated_response",
            query=query[:50],
            context_chunks=len(retrieval.results),
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )

        return GenerationResult(
            query=query,
            response=result.text,
            retrieval=retrieval,
            model=self.model,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
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
