"""Cited Generator - RAG generation with citations for Phase 1."""

import asyncio
import os
import re
import uuid
from dataclasses import dataclass

import anthropic
import structlog

from ..evidence.evidence_ledger import EvidenceLedger
from ..evidence.evidence_span import EvidenceSpan
from ..retrieval.simple_retriever import RetrievalResult, SimpleRetriever

logger = structlog.get_logger()


@dataclass
class CitedGenerationResult:
    """Result from generation with citations."""
    query: str
    response: str
    response_with_citations: str
    evidence_ledger: EvidenceLedger
    retrieval: RetrievalResult
    model: str
    input_tokens: int
    output_tokens: int


class CitedGenerator:
    """RAG generator with citation tracking (Phase 1).

    Generates responses and tracks which evidence supports each claim.
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
    ) -> CitedGenerationResult:
        """Generate an answer with citation tracking.

        Args:
            query: User's question
            top_k: Number of chunks to retrieve

        Returns:
            CitedGenerationResult with answer, citations, and evidence ledger
        """
        # Retrieve relevant chunks
        retrieval = await self.retriever.retrieve(query=query, top_k=top_k)

        # Format context with citation markers
        context, citation_map = self._format_context_with_markers(retrieval)

        # Build prompt that encourages citation use
        prompt = self._build_prompt(query, context)

        # Generate response
        response = await asyncio.to_thread(
            self.client.messages.create,
            model=self.model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )

        raw_response = response.content[0].text

        # Parse citations and build evidence ledger
        evidence_ledger = self._build_evidence_ledger(
            query=query,
            response=raw_response,
            citation_map=citation_map,
            retrieval=retrieval,
        )

        # Format response with inline citations
        response_with_citations = evidence_ledger.format_response_with_citations()

        logger.info(
            "generated_cited_response",
            query=query[:50],
            context_chunks=len(retrieval.results),
            claims=len(evidence_ledger.claims),
            coverage=evidence_ledger.compute_coverage(),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

        return CitedGenerationResult(
            query=query,
            response=raw_response,
            response_with_citations=response_with_citations,
            evidence_ledger=evidence_ledger,
            retrieval=retrieval,
            model=self.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    def _format_context_with_markers(
        self,
        retrieval: RetrievalResult,
    ) -> tuple[str, dict]:
        """Format context with citation markers and build citation map."""
        context_parts = []
        citation_map = {}  # marker -> search result

        for i, result in enumerate(retrieval.results):
            chunk = result.chunk
            marker = f"[{i + 1}]"
            citation_map[marker] = result

            # Format with clear marker
            doc_id = chunk.document_id.upper()
            section = f" - {chunk.section_path}" if chunk.section_path else ""

            context_parts.append(
                f"{marker} {doc_id}{section}\n{chunk.content}"
            )

        context = "\n\n---\n\n".join(context_parts)
        return context, citation_map

    def _build_prompt(self, query: str, context: str) -> str:
        """Build the generation prompt that encourages citations."""
        return f"""You are an expert on Ethereum protocol development. Answer the question based on the provided context from Ethereum Improvement Proposals (EIPs).

IMPORTANT INSTRUCTIONS:
1. Base your answer ONLY on the provided context
2. When making factual claims, cite the source using the marker (e.g., [1], [2])
3. If the context doesn't contain enough information, say so clearly
4. Don't make up information not found in the context

<context>
{context}
</context>

<question>
{query}
</question>

Provide a clear, accurate answer with citations to the relevant sources."""

    def _build_evidence_ledger(
        self,
        query: str,
        response: str,
        citation_map: dict,
        retrieval: RetrievalResult,
    ) -> EvidenceLedger:
        """Build evidence ledger from response with citations."""
        response_id = str(uuid.uuid4())[:8]

        ledger = EvidenceLedger(
            response_id=response_id,
            query=query,
            response_text=response,
        )

        # Extract sentences and their citations
        sentences = self._extract_sentences_with_citations(response)

        for i, (sentence, citations) in enumerate(sentences):
            if not sentence.strip():
                continue

            # Find sentence position in response
            start_offset = response.find(sentence)
            end_offset = start_offset + len(sentence) if start_offset >= 0 else 0

            # Determine claim type
            claim_type = "factual" if citations else "interpretive"

            claim_id = f"claim-{i + 1}"
            ledger.add_claim(
                claim_id=claim_id,
                claim_text=sentence,
                claim_type=claim_type,
                start_offset=start_offset,
                end_offset=end_offset,
            )

            # Add evidence for each citation
            for citation_marker in citations:
                if citation_marker in citation_map:
                    result = citation_map[citation_marker]
                    chunk = result.chunk

                    evidence = EvidenceSpan.from_chunk(
                        document_id=chunk.document_id,
                        chunk_id=chunk.chunk_id,
                        content=chunk.content,
                        section_path=chunk.section_path,
                        git_commit=None,  # Would come from storage
                    )
                    ledger.add_evidence(claim_id, evidence)

        return ledger

    def _extract_sentences_with_citations(
        self,
        response: str,
    ) -> list[tuple[str, list[str]]]:
        """Extract sentences and their citation markers.

        Returns list of (sentence, [citation_markers])
        """
        # Pattern for citation markers like [1], [2], [1,2], [1, 2]
        citation_pattern = re.compile(r'\[(\d+(?:\s*,\s*\d+)*)\]')

        # Split into sentences (roughly)
        sentence_pattern = re.compile(r'([^.!?]+[.!?]+)')
        sentences = sentence_pattern.findall(response)

        # If no sentence punctuation found, treat whole response as one
        if not sentences:
            sentences = [response]

        results = []
        for sentence in sentences:
            # Find all citations in this sentence
            matches = citation_pattern.findall(sentence)
            citations = []
            for match in matches:
                # Handle multiple citations like [1, 2]
                numbers = re.findall(r'\d+', match)
                for num in numbers:
                    citations.append(f"[{num}]")

            # Clean citation markers from sentence for storage
            clean_sentence = citation_pattern.sub('', sentence).strip()

            results.append((clean_sentence, citations))

        return results
