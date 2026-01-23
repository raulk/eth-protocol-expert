"""Code Embedder - Generate code-aware embeddings using Voyage AI."""

import os
from dataclasses import dataclass

import structlog
import voyageai

from src.chunking.code_chunker import CodeChunk

logger = structlog.get_logger()


@dataclass
class CodeEmbedding:
    """A code chunk with its embedding."""

    chunk_id: str
    embedding: list[float]
    language: str
    semantic_type: str


class CodeEmbedder:
    """Generate embeddings optimized for code using Voyage AI's voyage-code-3 model."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "voyage-code-3",
        batch_size: int = 128,
        output_dimension: int = 1024,
    ):
        self.api_key = api_key or os.environ.get("VOYAGE_API_KEY")
        if not self.api_key:
            raise ValueError("VOYAGE_API_KEY not set")

        self.model = model
        self.batch_size = batch_size
        self.output_dimension = output_dimension
        self.client = voyageai.Client(api_key=self.api_key)
        self.embedding_dim = output_dimension  # voyage-code-3 supports 256/512/1024/2048

    def embed_chunks(self, chunks: list[CodeChunk]) -> list[CodeEmbedding]:
        """Generate embeddings for a list of code chunks."""
        if not chunks:
            return []

        all_embeddings: list[CodeEmbedding] = []
        texts = [self._prepare_code_text(chunk) for chunk in chunks]

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            batch_chunks = chunks[i : i + self.batch_size]

            logger.debug(
                "embedding_code_batch",
                batch_num=i // self.batch_size + 1,
                batch_size=len(batch_texts),
            )

            result = self.client.embed(
                texts=batch_texts,
                model=self.model,
                input_type="document",
                output_dimension=self.output_dimension,
            )

            for chunk, embedding in zip(batch_chunks, result.embeddings, strict=True):
                semantic_type = self._infer_semantic_type(chunk)
                all_embeddings.append(
                    CodeEmbedding(
                        chunk_id=chunk.chunk_id,
                        embedding=embedding,
                        language=chunk.language,
                        semantic_type=semantic_type,
                    )
                )

        logger.info("embedded_code_chunks", count=len(all_embeddings))
        return all_embeddings

    def embed_query(
        self,
        query: str,
        code_context: bool = True,
    ) -> list[float]:
        """Embed a query for code search."""
        if code_context:
            query = self._prepare_code_query(query)

        result = self.client.embed(
            texts=[query],
            model=self.model,
            input_type="query",
            output_dimension=self.output_dimension,
        )
        return result.embeddings[0]

    def embed_code(self, code: str, language: str = "go") -> list[float]:
        """Embed a raw code snippet."""
        text = f"```{language}\n{code}\n```"
        result = self.client.embed(
            texts=[text],
            model=self.model,
            input_type="document",
            output_dimension=self.output_dimension,
        )
        return result.embeddings[0]

    def _prepare_code_text(self, chunk: CodeChunk) -> str:
        parts: list[str] = []

        parts.append(f"File: {chunk.file_path}")
        parts.append(f"Function: {chunk.function_name}")

        if chunk.dependencies:
            deps = ", ".join(chunk.dependencies[:5])
            parts.append(f"Dependencies: {deps}")

        parts.append("")
        parts.append(f"```{chunk.language}")
        parts.append(chunk.content)
        parts.append("```")

        return "\n".join(parts)

    def _prepare_code_query(self, query: str) -> str:
        code_keywords = [
            "function",
            "method",
            "struct",
            "type",
            "interface",
            "implementation",
            "implements",
            "returns",
            "parameter",
            "go",
            "golang",
            "rust",
            "eip",
            "ethereum",
        ]

        query_lower = query.lower()
        has_code_terms = any(kw in query_lower for kw in code_keywords)

        if has_code_terms:
            return query

        return f"Ethereum client code implementation: {query}"

    def _infer_semantic_type(self, chunk: CodeChunk) -> str:
        content_lower = chunk.content.lower()
        name_lower = chunk.function_name.lower()

        if any(kw in name_lower for kw in ["test", "bench", "example"]):
            return "test"

        if any(kw in name_lower for kw in ["new", "create", "init", "build"]):
            return "constructor"

        if any(kw in name_lower for kw in ["get", "fetch", "load", "read", "query"]):
            return "getter"

        if any(kw in name_lower for kw in ["set", "update", "write", "store", "save"]):
            return "setter"

        if any(kw in name_lower for kw in ["validate", "verify", "check", "is_", "has_"]):
            return "validator"

        if any(kw in name_lower for kw in ["process", "handle", "execute", "run", "apply"]):
            return "handler"

        if any(kw in name_lower for kw in ["parse", "decode", "deserialize", "unmarshal"]):
            return "parser"

        if any(kw in name_lower for kw in ["encode", "serialize", "marshal", "format"]):
            return "encoder"

        if "struct" in content_lower or "type " in content_lower:
            return "type_definition"

        return "function"
