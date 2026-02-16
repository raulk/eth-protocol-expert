"""OpenAI-compatible Embedder -- works with any /v1/embeddings endpoint.

Supports vLLM, text-embeddings-inference, llama.cpp, LiteLLM proxy,
LocalAI, Infinity, and any other server exposing the OpenAI embeddings API.
"""

import os
from dataclasses import dataclass

import httpx
import structlog

from ..chunking.fixed_chunker import Chunk

logger = structlog.get_logger()


@dataclass
class EmbeddedChunk:
    """A chunk with its embedding."""

    chunk: Chunk
    embedding: list[float]
    model: str


class OpenAIEmbedder:
    """Generate embeddings via an OpenAI-compatible /v1/embeddings endpoint.

    This lets you point at any local server (vLLM, text-embeddings-inference,
    llama.cpp, etc.) or remote provider that speaks the OpenAI embeddings API.
    """

    def __init__(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        batch_size: int = 64,
        embedding_dim: int = 1024,
    ):
        self.api_base = (
            api_base
            or os.environ.get("EMBEDDING_API_BASE")
            or os.environ.get("OPENAI_API_BASE", "http://localhost:8080/v1")
        ).rstrip("/")
        self.api_key = (
            api_key
            or os.environ.get("EMBEDDING_API_KEY")
            or os.environ.get("OPENAI_API_KEY", "")
        )
        self.model = model or os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self._client = httpx.Client(timeout=120)

    def _call_api(self, texts: list[str]) -> list[list[float]]:
        """Call the /v1/embeddings endpoint."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "input": texts,
        }

        url = f"{self.api_base}/embeddings"
        resp = self._client.post(url, json=payload, headers=headers)
        resp.raise_for_status()

        data = resp.json()
        # Sort by index to preserve input order (spec doesn't guarantee order)
        items = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in items]

    def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        """Embed a list of chunks."""
        if not chunks:
            return []

        all_embedded = []
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]
            texts = [c.content for c in batch]

            logger.debug(
                "embedding_batch",
                batch_num=i // self.batch_size + 1,
                batch_size=len(texts),
            )

            embeddings = self._call_api(texts)

            for chunk, embedding in zip(batch, embeddings, strict=True):
                all_embedded.append(
                    EmbeddedChunk(chunk=chunk, embedding=embedding, model=self.model)
                )

        logger.info("embedded_chunks", count=len(all_embedded))
        return all_embedded

    def embed_query(self, query: str) -> list[float]:
        """Embed a query for search."""
        return self._call_api([query])[0]

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text."""
        return self._call_api([text])[0]
