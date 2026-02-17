"""Code Retriever - Retrieve code units from parsed client codebases (Phase 10)."""

import asyncio
from dataclasses import dataclass

import structlog

from src.embeddings import Embedder
from src.parsing import CodeUnit, UnitType
from src.storage.pg_vector_store import PgVectorStore

logger = structlog.get_logger()


@dataclass
class CodeSearchResult:
    """A code search result with similarity score."""

    code_unit: CodeUnit
    similarity: float
    chunk_id: str


@dataclass
class CodeRetrievalResult:
    """Result from code-specific retrieval."""

    query: str
    results: list[CodeSearchResult]
    total_results: int


class CodeRetriever:
    """Retrieve code units from indexed client codebases.

    Supports searching for functions, structs, and methods with
    optional filtering by language, repository, and unit type.
    """

    def __init__(
        self,
        embedder: Embedder,
        store: PgVectorStore,
        default_top_k: int = 10,
    ):
        self.embedder = embedder
        self.store = store
        self.default_top_k = default_top_k

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        repository: str | None = None,
        language: str | None = None,
        unit_type: UnitType | None = None,
    ) -> CodeRetrievalResult:
        """Retrieve relevant code units for a query.

        Args:
            query: The search query (e.g., "EIP-1559 base fee calculation")
            top_k: Number of results to return
            repository: Filter by repository name (e.g., "go-ethereum")
            language: Filter by language (e.g., "go", "rust")
            unit_type: Filter by unit type (FUNCTION, STRUCT, METHOD)

        Returns:
            CodeRetrievalResult with ranked code units
        """
        k = top_k or self.default_top_k

        query_embedding = await asyncio.to_thread(self.embedder.embed_query, query)

        source_filter = None
        if repository:
            source_filter = f"code:{repository}"

        search_results = await self.store.search(
            query_embedding=query_embedding,
            limit=k * 2,
            source_filter=source_filter,
        )

        code_results = []
        for sr in search_results:
            metadata = sr.chunk.metadata or {}

            if metadata.get("content_type") != "code":
                continue

            if language and metadata.get("language") != language:
                continue

            if unit_type and metadata.get("unit_type") != unit_type.value:
                continue

            code_unit = CodeUnit(
                name=metadata.get("name", "unknown"),
                unit_type=UnitType(metadata.get("unit_type", "function")),
                content=sr.chunk.content,
                file_path=metadata.get("file_path", ""),
                line_range=(
                    metadata.get("start_line", 0),
                    metadata.get("end_line", 0),
                ),
                dependencies=metadata.get("dependencies", []),
            )

            code_results.append(
                CodeSearchResult(
                    code_unit=code_unit,
                    similarity=sr.similarity,
                    chunk_id=sr.chunk.chunk_id,
                )
            )

            if len(code_results) >= k:
                break

        logger.debug(
            "code_retrieval_complete",
            query=query[:50],
            results=len(code_results),
            repository=repository,
            language=language,
        )

        return CodeRetrievalResult(
            query=query,
            results=code_results,
            total_results=len(code_results),
        )

    async def find_eip_implementations(
        self,
        eip_number: int,
        repository: str | None = None,
        top_k: int = 20,
    ) -> CodeRetrievalResult:
        """Find code that implements a specific EIP.

        Searches for functions and types that mention or implement
        the specified EIP number.
        """
        query = f"EIP-{eip_number} implementation code"
        result = await self.retrieve(
            query=query,
            top_k=top_k,
            repository=repository,
        )

        eip_str = str(eip_number)
        scored_results = []

        for code_result in result.results:
            content_lower = code_result.code_unit.content.lower()
            name_lower = code_result.code_unit.name.lower()

            boost = 0.0

            if f"eip{eip_str}" in content_lower or f"eip-{eip_str}" in content_lower:
                boost += 0.2
            if f"eip{eip_str}" in name_lower or f"eip-{eip_str}" in name_lower:
                boost += 0.1

            boosted_similarity = min(1.0, code_result.similarity + boost)

            scored_results.append(
                CodeSearchResult(
                    code_unit=code_result.code_unit,
                    similarity=boosted_similarity,
                    chunk_id=code_result.chunk_id,
                )
            )

        scored_results.sort(key=lambda x: x.similarity, reverse=True)

        logger.info(
            "found_eip_implementations",
            eip=eip_number,
            count=len(scored_results),
            repository=repository,
        )

        return CodeRetrievalResult(
            query=f"EIP-{eip_number} implementation",
            results=scored_results[:top_k],
            total_results=len(scored_results),
        )

    def format_code_context(
        self,
        results: list[CodeSearchResult],
        include_line_numbers: bool = True,
    ) -> str:
        """Format code search results into a context string."""
        context_parts = []

        for result in results:
            unit = result.code_unit
            header = f"[{unit.unit_type.value.upper()}] {unit.name}"
            if include_line_numbers:
                header += f" ({unit.file_path}:{unit.line_range[0]}-{unit.line_range[1]})"
            else:
                header += f" ({unit.file_path})"

            context_parts.append(f"{header}\n```\n{unit.content}\n```")

        return "\n\n---\n\n".join(context_parts)

    async def search_by_function_name(
        self,
        function_name: str,
        repository: str | None = None,
        language: str | None = None,
    ) -> CodeRetrievalResult:
        """Search for a function by its exact name."""
        results = await self.retrieve(
            query=f"function {function_name}",
            top_k=20,
            repository=repository,
            language=language,
            unit_type=UnitType.FUNCTION,
        )

        exact_matches = [
            r for r in results.results if r.code_unit.name.lower() == function_name.lower()
        ]

        partial_matches = [
            r
            for r in results.results
            if function_name.lower() in r.code_unit.name.lower() and r not in exact_matches
        ]

        final_results = exact_matches + partial_matches

        return CodeRetrievalResult(
            query=function_name,
            results=final_results[:10],
            total_results=len(final_results),
        )
