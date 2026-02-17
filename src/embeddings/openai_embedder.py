"""OpenAI-compatible Embedder -- works with any /v1/embeddings endpoint.

Supports vLLM, text-embeddings-inference, llama.cpp, LiteLLM proxy,
LocalAI, Infinity, and any other server exposing the OpenAI embeddings API.

Includes:
- Automatic sub-chunking: oversized chunks are split into overlapping
  windows so that no data is lost during embedding.
- Matryoshka truncation: if the server returns higher-dimensional vectors
  than `embedding_dim`, truncate and L2-normalize.  This is critical for
  models trained with Matryoshka Representation Learning (e.g. Qwen3-Embedding)
  where you can safely reduce 4096-dim to 1024-dim with minimal quality loss.
- Resilient batching: automatic retry with smaller batches on 413 errors.
"""

import math
import os
from dataclasses import dataclass, replace

import httpx
import structlog

from ..chunking.fixed_chunker import Chunk

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Qwen3-Embedding-8B limits: 32768 tokens max sequence length.
# BGE-m3 limits: 8192 tokens max sequence length.
# Conservative char budget: code-heavy text tokenizes at ~2 chars/token,
# prose at ~4 chars/token.  We target 6000 tokens (~15 000 chars) to leave
# headroom for special tokens and variance.
# ---------------------------------------------------------------------------
MAX_CHUNK_CHARS = 15_000  # ~6 000 tokens at worst-case 2.5 chars/tok
OVERLAP_CHARS = 1_500     # ~500-750 token overlap between windows


@dataclass
class EmbeddedChunk:
    """A chunk with its embedding."""

    chunk: Chunk
    embedding: list[float]
    model: str


def _split_oversized(chunk: Chunk) -> list[Chunk]:
    """Split a single oversized chunk into overlapping sub-chunks.

    Each sub-chunk inherits the parent's metadata (document_id, section_path,
    etc.) with a modified chunk_id and updated token_count.  Overlap ensures
    context is not lost at window boundaries.

    If the chunk fits within MAX_CHUNK_CHARS, it is returned as-is in a
    single-element list.
    """
    text = chunk.content
    if len(text) <= MAX_CHUNK_CHARS:
        return [chunk]

    sub_chunks: list[Chunk] = []
    start = 0
    sub_idx = 0

    while start < len(text):
        end = start + MAX_CHUNK_CHARS
        window = text[start:end]

        sub_chunk = replace(
            chunk,
            chunk_id=f"{chunk.chunk_id}-sub{sub_idx}",
            content=window,
            token_count=len(window) // 3,  # rough estimate
        )
        sub_chunks.append(sub_chunk)

        # Advance by (window - overlap) so successive windows share context
        start += MAX_CHUNK_CHARS - OVERLAP_CHARS
        sub_idx += 1

    logger.info(
        "split_oversized_chunk",
        chunk_id=chunk.chunk_id,
        original_chars=len(text),
        sub_chunks=len(sub_chunks),
        overlap_chars=OVERLAP_CHARS,
    )
    return sub_chunks


def _truncate_and_normalize(
    embedding: list[float], target_dim: int
) -> list[float]:
    """Truncate to target dimensions and L2-normalize.

    Matryoshka Representation Learning trains models so that the first N
    dimensions of a larger embedding form a valid N-dimensional embedding.
    After truncation, L2 normalization restores unit length for cosine
    similarity to work correctly.

    If the embedding is already at or below target_dim, only normalization
    is applied (no truncation).
    """
    vec = embedding[:target_dim]
    # Guard against None values from server
    if any(x is None for x in vec):
        logger.warning("null_values_in_embedding", target_dim=target_dim)
        vec = [0.0 if x is None else x for x in vec]
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


class OpenAIEmbedder:
    """Generate embeddings via an OpenAI-compatible /v1/embeddings endpoint.

    This lets you point at any local server (vLLM, text-embeddings-inference,
    llama.cpp, etc.) or remote provider that speaks the OpenAI embeddings API.

    If the server returns vectors with more dimensions than ``embedding_dim``,
    they are automatically truncated and L2-normalized (Matryoshka truncation).
    """

    def __init__(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        batch_size: int = 8,
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
        self.embedding_dim = int(
            os.environ.get("EMBEDDING_DIM", str(embedding_dim))
        )
        self._client = httpx.Client(timeout=300)

        logger.info(
            "openai_embedder_init",
            api_base=self.api_base,
            model=self.model,
            embedding_dim=self.embedding_dim,
            batch_size=self.batch_size,
        )

    # ------------------------------------------------------------------
    # API interaction
    # ------------------------------------------------------------------

    def _call_api(self, texts: list[str]) -> list[list[float]]:
        """Call the /v1/embeddings endpoint with retry logic.

        Handles:
        - 422: hard-truncate text and retry (oversized input)
        - 413: halve batch size and retry (payload too large)
        """
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "input": texts,
        }

        url = f"{self.api_base}/embeddings"
        resp = self._client.post(url, json=payload, headers=headers)

        # Oversized input: hard-truncate and retry once
        if resp.status_code == 422:
            logger.warning(
                "422_hard_truncating_and_retrying",
                original_max_chars=max(len(t) for t in texts),
            )
            texts = [t[:8000] for t in texts]
            payload["input"] = texts
            resp = self._client.post(url, json=payload, headers=headers)

        # Payload too large: split batch in half and recurse
        if resp.status_code == 413:
            if len(texts) <= 1:
                # Single text is too large -- truncate and retry
                logger.warning(
                    "413_single_text_truncating",
                    chars=len(texts[0]),
                )
                texts = [texts[0][:8000]]
                payload["input"] = texts
                resp = self._client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                items = sorted(data["data"], key=lambda x: x["index"])
                return [item["embedding"] for item in items]

            mid = len(texts) // 2
            logger.warning(
                "413_splitting_batch",
                original_size=len(texts),
                split_into=[mid, len(texts) - mid],
            )
            left = self._call_api(texts[:mid])
            right = self._call_api(texts[mid:])
            return left + right

        resp.raise_for_status()

        data = resp.json()
        # Sort by index to preserve input order (spec doesn't guarantee order)
        items = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in items]

    def _embed_and_truncate(self, texts: list[str]) -> list[list[float]]:
        """Call API and apply Matryoshka truncation + L2 normalization."""
        raw = self._call_api(texts)
        server_dim = len(raw[0]) if raw else 0

        if server_dim > self.embedding_dim:
            if not hasattr(self, "_truncation_logged"):
                logger.info(
                    "matryoshka_truncation",
                    server_dim=server_dim,
                    target_dim=self.embedding_dim,
                )
                self._truncation_logged = True
            return [_truncate_and_normalize(v, self.embedding_dim) for v in raw]

        if server_dim and server_dim < self.embedding_dim:
            if not hasattr(self, "_dim_warning_logged"):
                logger.warning(
                    "embedding_dim_mismatch",
                    server_dim=server_dim,
                    configured_dim=self.embedding_dim,
                    msg="Server returns fewer dims than configured. "
                        "Update EMBEDDING_DIM or switch models.",
                )
                self._dim_warning_logged = True

        return raw

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        """Embed a list of chunks, sub-chunking any that exceed the model limit.

        Oversized chunks are split into overlapping windows so that every byte
        of the original content is represented in at least one embedding.
        """
        if not chunks:
            return []

        # Expand oversized chunks into sub-chunks
        expanded: list[Chunk] = []
        for chunk in chunks:
            expanded.extend(_split_oversized(chunk))

        all_embedded: list[EmbeddedChunk] = []
        for i in range(0, len(expanded), self.batch_size):
            batch = expanded[i : i + self.batch_size]
            texts = [c.content for c in batch]

            logger.debug(
                "embedding_batch",
                batch_num=i // self.batch_size + 1,
                batch_size=len(texts),
            )

            embeddings = self._embed_and_truncate(texts)

            for chunk, embedding in zip(batch, embeddings, strict=True):
                all_embedded.append(
                    EmbeddedChunk(chunk=chunk, embedding=embedding, model=self.model)
                )

        logger.info("embedded_chunks", count=len(all_embedded))
        return all_embedded

    def embed_query(self, query: str) -> list[float]:
        """Embed a query for search."""
        return self._embed_and_truncate([query])[0]

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text."""
        return self._embed_and_truncate([text])[0]
