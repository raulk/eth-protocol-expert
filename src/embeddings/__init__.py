"""Embedding providers -- factory functions select backend from config."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .code_embedder import CodeEmbedder, CodeEmbedding
from .local_embedder import LocalEmbedder
from .openai_embedder import OpenAIEmbedder
from .voyage_embedder import VoyageEmbedder

if TYPE_CHECKING:
    from ..chunking.fixed_chunker import Chunk

# Re-export the canonical EmbeddedChunk from voyage (identical in all three)
from .voyage_embedder import EmbeddedChunk


# ---------------------------------------------------------------------------
# Protocol that all embedders satisfy (duck typing made explicit)
# ---------------------------------------------------------------------------
@runtime_checkable
class Embedder(Protocol):
    """Interface satisfied by VoyageEmbedder, LocalEmbedder, and OpenAIEmbedder."""

    embedding_dim: int

    def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]: ...
    def embed_query(self, query: str) -> list[float]: ...
    def embed_text(self, text: str, *args, **kwargs) -> list[float]: ...


# ---------------------------------------------------------------------------
# Factory -- reads from config (env vars) unless caller overrides
# ---------------------------------------------------------------------------
def create_embedder(
    embedder_type: str | None = None,
    model: str | None = None,
    embedding_dim: int | None = None,
) -> VoyageEmbedder | LocalEmbedder | OpenAIEmbedder:
    """Create an embedder based on configuration.

    All arguments are optional; when omitted they fall back to the
    environment variables defined in src.config (EMBEDDER_TYPE,
    EMBEDDING_MODEL, EMBEDDING_DIM, etc.).
    """
    from ..config import EMBEDDING_API_BASE, EMBEDDING_API_KEY, EMBEDDING_DIM, EMBEDDING_MODEL, EMBEDDER_TYPE

    etype = (embedder_type or EMBEDDER_TYPE).lower()
    emodel = model or EMBEDDING_MODEL
    edim = embedding_dim or EMBEDDING_DIM

    if etype == "voyage":
        return VoyageEmbedder(model=emodel)
    elif etype == "local":
        return LocalEmbedder(model_name=emodel)
    elif etype == "openai":
        return OpenAIEmbedder(
            model=emodel,
            api_base=EMBEDDING_API_BASE or None,
            api_key=EMBEDDING_API_KEY or None,
            embedding_dim=edim,
        )
    else:
        raise ValueError(
            f"Unknown EMBEDDER_TYPE={etype!r}. "
            "Expected 'voyage', 'local', or 'openai'."
        )


def create_code_embedder(
    embedder_type: str | None = None,
    model: str | None = None,
    embedding_dim: int | None = None,
) -> CodeEmbedder | LocalEmbedder | OpenAIEmbedder:
    """Create a code embedder based on configuration.

    For 'voyage' this returns the specialised CodeEmbedder.
    For 'local' and 'openai' it returns a general embedder configured
    with the CODE_EMBEDDING_MODEL -- the code-specific preprocessing
    is handled by CodeChunker before embedding.
    """
    from ..config import (
        CODE_EMBEDDER_TYPE,
        CODE_EMBEDDING_DIM,
        CODE_EMBEDDING_MODEL,
        EMBEDDING_API_BASE,
        EMBEDDING_API_KEY,
    )

    etype = (embedder_type or CODE_EMBEDDER_TYPE).lower()
    emodel = model or CODE_EMBEDDING_MODEL
    edim = embedding_dim or CODE_EMBEDDING_DIM

    if etype == "voyage":
        return CodeEmbedder(model=emodel, output_dimension=edim)
    elif etype == "local":
        return LocalEmbedder(model_name=emodel)
    elif etype == "openai":
        return OpenAIEmbedder(
            model=emodel,
            api_base=EMBEDDING_API_BASE or None,
            api_key=EMBEDDING_API_KEY or None,
            embedding_dim=edim,
        )
    else:
        raise ValueError(
            f"Unknown CODE_EMBEDDER_TYPE={etype!r}. "
            "Expected 'voyage', 'local', or 'openai'."
        )


__all__ = [
    "CodeEmbedder",
    "CodeEmbedding",
    "Embedder",
    "EmbeddedChunk",
    "LocalEmbedder",
    "OpenAIEmbedder",
    "VoyageEmbedder",
    "create_code_embedder",
    "create_embedder",
]
