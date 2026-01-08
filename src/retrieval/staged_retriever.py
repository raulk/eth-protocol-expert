"""Staged Retriever - Retrieve for each sub-question (Phase 5)."""

from dataclasses import dataclass

import structlog

from ..routing.query_decomposer import DecompositionResult, SubQuestion
from ..storage.pg_vector_store import SearchResult
from .budget_manager import BudgetAllocation, BudgetManager
from .simple_retriever import RetrievalResult, SimpleRetriever

logger = structlog.get_logger()


@dataclass
class SubQuestionRetrievalResult:
    """Result of retrieval for a single sub-question."""
    sub_question: SubQuestion
    retrieval: RetrievalResult
    allocation: BudgetAllocation
    chunks_retrieved: int
    tokens_retrieved: int
    average_similarity: float


@dataclass
class StagedRetrievalResult:
    """Result from staged retrieval across all sub-questions."""
    original_query: str
    sub_question_results: list[SubQuestionRetrievalResult]
    total_chunks: int
    total_tokens: int
    budget_summary: dict


class StagedRetriever:
    """Retrieve relevant chunks for each sub-question in a decomposed query.

    Implements staged retrieval:
    1. For each sub-question, retrieve relevant chunks
    2. Track budget usage across all retrievals
    3. Return organized results for synthesis

    Works with the BudgetManager to ensure fair allocation
    across sub-questions while staying within limits.
    """

    def __init__(
        self,
        retriever: SimpleRetriever,
        budget_manager: BudgetManager,
    ):
        self.retriever = retriever
        self.budget_manager = budget_manager

    async def retrieve_staged(
        self,
        decomposition: DecompositionResult,
    ) -> StagedRetrievalResult:
        """Retrieve chunks for all sub-questions.

        Args:
            decomposition: The decomposed query with sub-questions

        Returns:
            StagedRetrievalResult with organized retrievals per sub-question
        """
        # Reset budget for new query
        self.budget_manager.reset()
        self.budget_manager.set_sub_question_count(len(decomposition.sub_questions))

        sub_question_results = []
        total_chunks = 0
        total_tokens = 0

        for sub_question in decomposition.sub_questions:
            # Check if budget is exhausted
            if self.budget_manager.is_budget_exhausted():
                logger.warning(
                    "budget_exhausted_early",
                    sub_question_index=sub_question.index,
                    remaining_sqs=len(decomposition.sub_questions) - sub_question.index,
                )
                break

            # Get allocation for this sub-question
            allocation = self.budget_manager.get_allocation(sub_question.index)

            if allocation.max_chunks <= 0:
                logger.warning(
                    "no_budget_for_sub_question",
                    sub_question_index=sub_question.index,
                    reason=allocation.reason,
                )
                continue

            # Retrieve for this sub-question
            result = await self._retrieve_for_sub_question(
                sub_question=sub_question,
                allocation=allocation,
            )

            sub_question_results.append(result)
            total_chunks += result.chunks_retrieved
            total_tokens += result.tokens_retrieved

        budget_summary = self.budget_manager.get_utilization_summary()

        logger.info(
            "staged_retrieval_complete",
            original_query=decomposition.original_query[:50],
            num_sub_questions=len(decomposition.sub_questions),
            sub_questions_served=len(sub_question_results),
            total_chunks=total_chunks,
            total_tokens=total_tokens,
            budget_utilization=budget_summary["chunks_utilization"],
        )

        return StagedRetrievalResult(
            original_query=decomposition.original_query,
            sub_question_results=sub_question_results,
            total_chunks=total_chunks,
            total_tokens=total_tokens,
            budget_summary=budget_summary,
        )

    async def _retrieve_for_sub_question(
        self,
        sub_question: SubQuestion,
        allocation: BudgetAllocation,
    ) -> SubQuestionRetrievalResult:
        """Retrieve chunks for a single sub-question.

        Args:
            sub_question: The sub-question to retrieve for
            allocation: Budget allocation for this retrieval

        Returns:
            SubQuestionRetrievalResult with retrieved chunks
        """
        # Build retrieval query
        query = sub_question.text

        # If we have an entity focus, add it to the query
        if sub_question.entity:
            # Ensure EIP ID is in the query for better retrieval
            if sub_question.entity.upper() not in query.upper():
                query = f"{sub_question.entity}: {query}"

        # Retrieve with budget limit
        retrieval = await self.retriever.retrieve(
            query=query,
            top_k=allocation.max_chunks,
        )

        # Filter by token budget
        filtered_results, actual_tokens = self._filter_by_token_budget(
            results=retrieval.results,
            max_tokens=allocation.max_tokens,
        )

        # Update retrieval with filtered results
        retrieval = RetrievalResult(
            query=query,
            results=filtered_results,
            total_tokens=actual_tokens,
        )

        # Record usage
        chunks_retrieved = len(filtered_results)
        tokens_retrieved = actual_tokens
        self.budget_manager.record_usage(
            sub_question_index=sub_question.index,
            chunks=chunks_retrieved,
            tokens=tokens_retrieved,
        )

        # Calculate average similarity
        avg_similarity = 0.0
        if filtered_results:
            avg_similarity = sum(r.similarity for r in filtered_results) / len(filtered_results)

        logger.debug(
            "sub_question_retrieval",
            sub_question_index=sub_question.index,
            query=query[:50],
            chunks=chunks_retrieved,
            tokens=tokens_retrieved,
            avg_similarity=round(avg_similarity, 3),
        )

        return SubQuestionRetrievalResult(
            sub_question=sub_question,
            retrieval=retrieval,
            allocation=allocation,
            chunks_retrieved=chunks_retrieved,
            tokens_retrieved=tokens_retrieved,
            average_similarity=avg_similarity,
        )

    def _filter_by_token_budget(
        self,
        results: list[SearchResult],
        max_tokens: int,
    ) -> tuple[list[SearchResult], int]:
        """Filter results to stay within token budget.

        Args:
            results: Search results to filter
            max_tokens: Maximum tokens allowed

        Returns:
            Tuple of (filtered_results, total_tokens)
        """
        filtered = []
        total_tokens = 0

        for result in results:
            chunk_tokens = result.chunk.token_count
            if total_tokens + chunk_tokens > max_tokens:
                break
            filtered.append(result)
            total_tokens += chunk_tokens

        return filtered, total_tokens

    def format_context_for_synthesis(
        self,
        staged_result: StagedRetrievalResult,
    ) -> str:
        """Format staged retrieval results into context for synthesis.

        Organizes chunks by sub-question with clear labeling.

        Args:
            staged_result: The staged retrieval results

        Returns:
            Formatted context string for LLM synthesis
        """
        sections = []

        for sq_result in staged_result.sub_question_results:
            sub_q = sq_result.sub_question

            # Header for this sub-question
            header = f"=== Sub-question {sub_q.index + 1}: {sub_q.text} ==="
            if sub_q.entity:
                header += f" [Entity: {sub_q.entity}]"

            chunks = []
            for i, result in enumerate(sq_result.retrieval.results):
                chunk = result.chunk
                doc_id = chunk.document_id.upper()
                section = f" - {chunk.section_path}" if chunk.section_path else ""

                chunks.append(
                    f"[{sub_q.index + 1}.{i + 1}] {doc_id}{section}\n{chunk.content}"
                )

            section_text = header + "\n\n" + "\n\n".join(chunks)
            sections.append(section_text)

        return "\n\n" + ("=" * 60) + "\n\n".join(sections)

    def format_context_with_citations(
        self,
        staged_result: StagedRetrievalResult,
    ) -> tuple[str, dict]:
        """Format context with citation markers for each chunk.

        Returns context string and a map of citation IDs to chunk metadata.

        Args:
            staged_result: The staged retrieval results

        Returns:
            Tuple of (context_string, citation_map)
        """
        sections = []
        citation_map = {}
        citation_counter = 1

        for sq_result in staged_result.sub_question_results:
            sub_q = sq_result.sub_question

            header = f"### Sub-question {sub_q.index + 1}: {sub_q.text}"

            chunks = []
            for result in sq_result.retrieval.results:
                chunk = result.chunk
                citation_id = f"[{citation_counter}]"

                citation_map[citation_id] = {
                    "sub_question_index": sub_q.index,
                    "document_id": chunk.document_id,
                    "section": chunk.section_path or "Full document",
                    "chunk_id": chunk.chunk_id,
                    "similarity": result.similarity,
                }

                doc_id = chunk.document_id.upper()
                section = f" - {chunk.section_path}" if chunk.section_path else ""

                chunks.append(
                    f"{citation_id} {doc_id}{section}\n{chunk.content}"
                )

                citation_counter += 1

            section_text = header + "\n\n" + "\n\n---\n\n".join(chunks)
            sections.append(section_text)

        context = "\n\n" + "=" * 40 + "\n\n".join(sections)
        return context, citation_map

    def get_all_results(
        self,
        staged_result: StagedRetrievalResult,
    ) -> list[SearchResult]:
        """Get flat list of all search results across sub-questions.

        Args:
            staged_result: The staged retrieval results

        Returns:
            List of all SearchResult objects
        """
        all_results = []
        for sq_result in staged_result.sub_question_results:
            all_results.extend(sq_result.retrieval.results)
        return all_results
