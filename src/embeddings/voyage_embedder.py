"""Voyage Embedder - Generate embeddings using Voyage AI."""

import os
from dataclasses import dataclass

import structlog
import voyageai

from ..chunking.fixed_chunker import Chunk

logger = structlog.get_logger()


@dataclass
class EmbeddedChunk:
    """A chunk with its embedding."""
    chunk: Chunk
    embedding: list[float]
    model: str


class VoyageEmbedder:
    """Generate embeddings using Voyage AI's voyage-4-large model."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "voyage-4-large",
        batch_size: int = 128,
    ):
        self.api_key = api_key or os.environ.get("VOYAGE_API_KEY")
        if not self.api_key:
            raise ValueError("VOYAGE_API_KEY not set")

        self.model = model
        self.batch_size = batch_size
        self.client = voyageai.Client(api_key=self.api_key)
        self.embedding_dim = 1024  # voyage-4-large default dimension

    def _make_token_batches(self, chunks: list[Chunk]) -> list[list[int]]:
        """Split chunks into batches that fit within Voyage's token limit.

        Uses a rough 4 chars/token estimate. Voyage's limit is 120K tokens
        per batch; we target 100K to leave headroom.
        """
        max_tokens = 100_000
        chars_per_token = 4
        batches: list[list[int]] = []
        current_batch: list[int] = []
        current_tokens = 0

        for i, chunk in enumerate(chunks):
            est_tokens = len(chunk.content) // chars_per_token + 1
            if current_batch and (
                current_tokens + est_tokens > max_tokens
                or len(current_batch) >= self.batch_size
            ):
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
            current_batch.append(i)
            current_tokens += est_tokens

        if current_batch:
            batches.append(current_batch)
        return batches

    def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        """Embed a list of chunks."""
        if not chunks:
            return []

        all_embedded = []
        batches = self._make_token_batches(chunks)

        for batch_num, indices in enumerate(batches, 1):
            batch_texts = [chunks[i].content for i in indices]
            batch_chunks = [chunks[i] for i in indices]

            logger.debug(
                "embedding_batch",
                batch_num=batch_num,
                total_batches=len(batches),
                batch_size=len(batch_texts),
            )

            result = self.client.embed(
                texts=batch_texts,
                model=self.model,
                input_type="document",
            )

            for chunk, embedding in zip(batch_chunks, result.embeddings, strict=True):
                all_embedded.append(EmbeddedChunk(
                    chunk=chunk,
                    embedding=embedding,
                    model=self.model,
                ))

        logger.info("embedded_chunks", count=len(all_embedded))
        return all_embedded

    def embed_query(self, query: str) -> list[float]:
        """Embed a query for search."""
        result = self.client.embed(
            texts=[query],
            model=self.model,
            input_type="query",
        )
        return result.embeddings[0]

    def embed_text(self, text: str, input_type: str = "document") -> list[float]:
        """Embed a single text."""
        result = self.client.embed(
            texts=[text],
            model=self.model,
            input_type=input_type,
        )
        return result.embeddings[0]
