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
    """Generate embeddings using Voyage AI's voyage-3 model."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "voyage-3",
        batch_size: int = 128,
    ):
        self.api_key = api_key or os.environ.get("VOYAGE_API_KEY")
        if not self.api_key:
            raise ValueError("VOYAGE_API_KEY not set")

        self.model = model
        self.batch_size = batch_size
        self.client = voyageai.Client(api_key=self.api_key)
        self.embedding_dim = 1024  # voyage-3 dimension

    def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        """Embed a list of chunks."""
        if not chunks:
            return []

        # Process in batches
        all_embedded = []
        texts = [chunk.content for chunk in chunks]

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_chunks = chunks[i:i + self.batch_size]

            logger.debug(
                "embedding_batch",
                batch_num=i // self.batch_size + 1,
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
