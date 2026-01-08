"""Local Embedder - Generate embeddings using sentence-transformers (no API needed)."""

from dataclasses import dataclass

import structlog

from ..chunking.fixed_chunker import Chunk

logger = structlog.get_logger()


@dataclass
class EmbeddedChunk:
    """A chunk with its embedding."""
    chunk: Chunk
    embedding: list[float]
    model: str


class LocalEmbedder:
    """Generate embeddings using local sentence-transformers model.

    Uses BGE-large which is one of the best open-source embedding models.
    No API key required - runs locally.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        device: str | None = None,
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None
        self._device = device

    def _load_model(self):
        """Lazy load the model."""
        if self._model is not None:
            return

        from sentence_transformers import SentenceTransformer

        logger.info("loading_local_embedding_model", model=self.model_name)
        self._model = SentenceTransformer(self.model_name, device=self._device)
        self.embedding_dim = self._model.get_sentence_embedding_dimension()
        logger.info("model_loaded", dim=self.embedding_dim)

    def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        """Embed a list of chunks."""
        if not chunks:
            return []

        self._load_model()

        texts = [chunk.content for chunk in chunks]

        logger.debug("embedding_chunks", count=len(texts))

        # BGE models benefit from adding instruction prefix for retrieval
        if "bge" in self.model_name.lower():
            # For document encoding, no prefix needed
            pass

        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True,
        )

        result = []
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            result.append(EmbeddedChunk(
                chunk=chunk,
                embedding=embedding.tolist(),
                model=self.model_name,
            ))

        logger.info("embedded_chunks", count=len(result))
        return result

    def embed_query(self, query: str) -> list[float]:
        """Embed a query for search."""
        self._load_model()

        # BGE models use instruction prefix for queries
        if "bge" in self.model_name.lower():
            query = f"Represent this sentence for searching relevant passages: {query}"

        embedding = self._model.encode(
            query,
            normalize_embeddings=True,
        )
        return embedding.tolist()

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text."""
        self._load_model()
        embedding = self._model.encode(text, normalize_embeddings=True)
        return embedding.tolist()
