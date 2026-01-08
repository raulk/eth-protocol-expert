"""Synthesis Generator - Combine sub-answers into final response (Phase 5)."""

import os
import re
import uuid
from dataclasses import dataclass
from typing import ClassVar

import anthropic
import structlog

from ..evidence.evidence_ledger import EvidenceLedger
from ..evidence.evidence_span import EvidenceSpan
from ..retrieval.budget_manager import BudgetManager
from ..retrieval.staged_retriever import StagedRetrievalResult, StagedRetriever
from ..routing.query_classifier import ClassificationResult, QueryClassifier
from ..routing.query_decomposer import DecompositionResult, QueryDecomposer, SynthesisStrategy

logger = structlog.get_logger()


@dataclass
class SynthesisResult:
    """Result from synthesis generation."""
    query: str
    response: str
    response_with_citations: str
    classification: ClassificationResult
    decomposition: DecompositionResult | None
    staged_retrieval: StagedRetrievalResult | None
    evidence_ledger: EvidenceLedger
    model: str
    input_tokens: int
    output_tokens: int


class SynthesisGenerator:
    """Generate responses by synthesizing answers to sub-questions.

    Handles the full query decomposition pipeline:
    1. Classify query complexity
    2. Decompose complex queries into sub-questions
    3. Retrieve for each sub-question
    4. Synthesize final answer

    For simple queries, falls through to standard retrieval.
    """

    SYNTHESIS_PROMPTS: ClassVar[dict[SynthesisStrategy, str]] = {
        SynthesisStrategy.COMPARISON: """You are comparing multiple Ethereum concepts. Based on the information gathered for each sub-question:

1. Identify the key dimensions of comparison
2. For each dimension, explain how the concepts differ
3. Highlight similarities where they exist
4. Use specific details from the sources to support your comparison

Structure your response with clear comparison points.""",

        SynthesisStrategy.TIMELINE: """You are constructing a timeline or evolution of Ethereum concepts. Based on the information gathered:

1. Present events in chronological order
2. Explain causal relationships between events
3. Note key milestones and their significance
4. Show how concepts built on or replaced earlier ones

Structure your response chronologically.""",

        SynthesisStrategy.EXPLANATION: """You are explaining an Ethereum concept. Based on the information gathered:

1. Start with a clear, concise overview
2. Explain the technical details progressively
3. Address the motivation and rationale
4. Connect to related concepts where relevant

Structure your response for clarity.""",

        SynthesisStrategy.AGGREGATION: """You are aggregating information about multiple related Ethereum concepts. Based on the information gathered:

1. Present each concept clearly
2. Show how they relate to each other
3. Identify common themes or patterns
4. Provide a synthesized overview

Structure your response to show both individual concepts and their connections.""",
    }

    def __init__(
        self,
        staged_retriever: StagedRetriever,
        query_classifier: QueryClassifier,
        query_decomposer: QueryDecomposer,
        budget_manager: BudgetManager,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        max_context_tokens: int = 8000,
    ):
        self.staged_retriever = staged_retriever
        self.query_classifier = query_classifier
        self.query_decomposer = query_decomposer
        self.budget_manager = budget_manager

        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

        self.model = model
        self.max_context_tokens = max_context_tokens
        self.client = anthropic.Anthropic(api_key=self.api_key)

    async def generate(self, query: str) -> SynthesisResult:
        """Generate a response, using decomposition for complex queries.

        Args:
            query: User's question

        Returns:
            SynthesisResult with response and all intermediate results
        """
        # Step 1: Classify the query
        classification = await self.query_classifier.classify(query)

        logger.info(
            "query_classified",
            query=query[:50],
            query_type=classification.query_type.value,
            needs_decomposition=classification.needs_decomposition,
        )

        # Step 2: Decompose if needed
        decomposition = None
        staged_retrieval = None

        if classification.needs_decomposition:
            decomposition = await self.query_decomposer.decompose(query)

            logger.info(
                "query_decomposed",
                query=query[:50],
                num_sub_questions=len(decomposition.sub_questions),
                strategy=decomposition.synthesis_strategy.value,
            )

            # Step 3: Staged retrieval
            staged_retrieval = await self.staged_retriever.retrieve_staged(decomposition)

            # Step 4: Synthesize
            return await self._synthesize_response(
                query=query,
                classification=classification,
                decomposition=decomposition,
                staged_retrieval=staged_retrieval,
            )

        else:
            # Simple query: single retrieval + generation
            return await self._simple_generate(query, classification)

    async def _synthesize_response(
        self,
        query: str,
        classification: ClassificationResult,
        decomposition: DecompositionResult,
        staged_retrieval: StagedRetrievalResult,
    ) -> SynthesisResult:
        """Synthesize response from staged retrieval results."""
        # Build context with citations
        context, citation_map = self.staged_retriever.format_context_with_citations(
            staged_retrieval
        )

        # Build synthesis prompt
        prompt = self._build_synthesis_prompt(
            query=query,
            decomposition=decomposition,
            context=context,
        )

        # Generate response
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )

        raw_response = response.content[0].text

        # Build evidence ledger
        evidence_ledger = self._build_evidence_ledger(
            query=query,
            response=raw_response,
            citation_map=citation_map,
            staged_retrieval=staged_retrieval,
        )

        response_with_citations = evidence_ledger.format_response_with_citations()

        logger.info(
            "synthesis_complete",
            query=query[:50],
            strategy=decomposition.synthesis_strategy.value,
            num_sub_questions=len(decomposition.sub_questions),
            total_chunks=staged_retrieval.total_chunks,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

        return SynthesisResult(
            query=query,
            response=raw_response,
            response_with_citations=response_with_citations,
            classification=classification,
            decomposition=decomposition,
            staged_retrieval=staged_retrieval,
            evidence_ledger=evidence_ledger,
            model=self.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    async def _simple_generate(
        self,
        query: str,
        classification: ClassificationResult,
    ) -> SynthesisResult:
        """Handle simple queries with standard retrieval."""
        from .cited_generator import CitedGenerator

        # Use the base retriever from staged retriever
        base_retriever = self.staged_retriever.retriever

        # Create a cited generator for simple queries
        cited_gen = CitedGenerator(
            retriever=base_retriever,
            api_key=self.api_key,
            model=self.model,
            max_context_tokens=self.max_context_tokens,
        )

        cited_result = await cited_gen.generate(query)

        return SynthesisResult(
            query=query,
            response=cited_result.response,
            response_with_citations=cited_result.response_with_citations,
            classification=classification,
            decomposition=None,
            staged_retrieval=None,
            evidence_ledger=cited_result.evidence_ledger,
            model=self.model,
            input_tokens=cited_result.input_tokens,
            output_tokens=cited_result.output_tokens,
        )

    def _build_synthesis_prompt(
        self,
        query: str,
        decomposition: DecompositionResult,
        context: str,
    ) -> str:
        """Build the synthesis prompt based on strategy."""
        strategy_guidance = self.SYNTHESIS_PROMPTS.get(
            decomposition.synthesis_strategy,
            self.SYNTHESIS_PROMPTS[SynthesisStrategy.EXPLANATION],
        )

        sub_questions_text = "\n".join(
            f"  {i + 1}. {sq.text}"
            for i, sq in enumerate(decomposition.sub_questions)
        )

        return f"""You are an expert on Ethereum protocol development. Answer the question by synthesizing information from the provided context.

ORIGINAL QUESTION: {query}

SUB-QUESTIONS ADDRESSED:
{sub_questions_text}

SYNTHESIS GUIDANCE:
{strategy_guidance}

IMPORTANT INSTRUCTIONS:
1. Base your answer ONLY on the provided context
2. When making factual claims, cite the source using the marker (e.g., [1], [2])
3. If the context doesn't contain enough information, say so clearly
4. Don't make up information not found in the context
5. Synthesize information across sub-questions into a coherent answer

<context>
{context}
</context>

Provide a clear, well-structured answer that synthesizes the information to answer the original question."""

    def _build_evidence_ledger(
        self,
        query: str,
        response: str,
        citation_map: dict,
        staged_retrieval: StagedRetrievalResult,
    ) -> EvidenceLedger:
        """Build evidence ledger from synthesis response."""
        response_id = str(uuid.uuid4())[:8]

        ledger = EvidenceLedger(
            response_id=response_id,
            query=query,
            response_text=response,
        )

        # Extract sentences and citations
        sentences = self._extract_sentences_with_citations(response)

        for i, (sentence, citations) in enumerate(sentences):
            if not sentence.strip():
                continue

            start_offset = response.find(sentence)
            end_offset = start_offset + len(sentence) if start_offset >= 0 else 0

            claim_type = "factual" if citations else "interpretive"
            claim_id = f"claim-{i + 1}"

            ledger.add_claim(
                claim_id=claim_id,
                claim_text=sentence,
                claim_type=claim_type,
                start_offset=start_offset,
                end_offset=end_offset,
            )

            for citation_marker in citations:
                if citation_marker in citation_map:
                    meta = citation_map[citation_marker]

                    # Find the chunk in staged retrieval
                    chunk_content = self._find_chunk_content(
                        staged_retrieval,
                        meta["sub_question_index"],
                        meta["chunk_id"],
                    )

                    evidence = EvidenceSpan.from_chunk(
                        document_id=meta["document_id"],
                        chunk_id=meta["chunk_id"],
                        content=chunk_content or "",
                        section_path=meta["section"],
                        git_commit=None,
                    )
                    ledger.add_evidence(claim_id, evidence)

        return ledger

    def _find_chunk_content(
        self,
        staged_retrieval: StagedRetrievalResult,
        sub_question_index: int,
        chunk_id: str,
    ) -> str | None:
        """Find chunk content in staged retrieval results."""
        for sq_result in staged_retrieval.sub_question_results:
            if sq_result.sub_question.index == sub_question_index:
                for result in sq_result.retrieval.results:
                    if result.chunk.chunk_id == chunk_id:
                        return result.chunk.content
        return None

    def _extract_sentences_with_citations(
        self,
        response: str,
    ) -> list[tuple[str, list[str]]]:
        """Extract sentences and their citation markers."""
        citation_pattern = re.compile(r'\[(\d+(?:\s*,\s*\d+)*)\]')
        sentence_pattern = re.compile(r'([^.!?]+[.!?]+)')

        sentences = sentence_pattern.findall(response)
        if not sentences:
            sentences = [response]

        results = []
        for sentence in sentences:
            matches = citation_pattern.findall(sentence)
            citations = []
            for match in matches:
                numbers = re.findall(r'\d+', match)
                for num in numbers:
                    citations.append(f"[{num}]")

            clean_sentence = citation_pattern.sub('', sentence).strip()
            results.append((clean_sentence, citations))

        return results
