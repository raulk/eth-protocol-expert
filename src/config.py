"""Central configuration -- all settings driven by environment variables.

Existing Voyage/Anthropic users change nothing; .env is backward-compatible.
Local-inference users set EMBEDDER_TYPE, LLM_MODEL_DEFAULT, etc.

LLM model strings use LiteLLM conventions:
  - claude-*              -> Anthropic  (ANTHROPIC_API_KEY)
  - gemini/*              -> Google     (GEMINI_API_KEY)
  - openrouter/*          -> OpenRouter (OPENROUTER_API_KEY)
  - openai/*              -> OpenAI-compat  (OPENAI_API_KEY + OPENAI_API_BASE)
  - ollama/*              -> Ollama     (OLLAMA_API_BASE, default localhost:11434)
  - ollama_chat/*         -> Ollama     (same, forces /chat endpoint)
  - and many more: https://docs.litellm.ai/docs/providers
"""

import os

# ---------------------------------------------------------------------------
# LLM generation
# ---------------------------------------------------------------------------
# Model strings follow LiteLLM's provider/model convention.  The default
# values preserve the original Anthropic-only behaviour.

DEFAULT_MODEL = os.environ.get("LLM_MODEL_DEFAULT", "claude-opus-4-6")
MODEL_FAST = os.environ.get("LLM_MODEL_FAST", "claude-haiku-4-5-20251001")
MODEL_BALANCED = os.environ.get("LLM_MODEL_BALANCED", "claude-sonnet-4-5-20250929")
MODEL_POWERFUL = os.environ.get("LLM_MODEL_POWERFUL", "claude-opus-4-6")

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
# EMBEDDER_TYPE selects the backend:
#   "voyage"  - Voyage AI API (default, requires VOYAGE_API_KEY)
#   "local"   - sentence-transformers, runs in-process on CPU/GPU
#   "openai"  - any OpenAI-compatible /v1/embeddings endpoint

EMBEDDER_TYPE = os.environ.get("EMBEDDER_TYPE", "voyage")
EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL",
    {
        "voyage": "voyage-4-large",
        "local": "BAAI/bge-large-en-v1.5",
        "openai": "text-embedding-3-small",
    }.get(EMBEDDER_TYPE, "voyage-4-large"),
)
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "1024"))

# For the "openai" embedder type
EMBEDDING_API_BASE = os.environ.get("EMBEDDING_API_BASE", "")
EMBEDDING_API_KEY = os.environ.get("EMBEDDING_API_KEY", "")

# ---------------------------------------------------------------------------
# Code embeddings
# ---------------------------------------------------------------------------
# CODE_EMBEDDER_TYPE defaults to EMBEDDER_TYPE so a single env var switch
# moves everything local.  Override independently if desired.

CODE_EMBEDDER_TYPE = os.environ.get("CODE_EMBEDDER_TYPE", EMBEDDER_TYPE)
CODE_EMBEDDING_MODEL = os.environ.get(
    "CODE_EMBEDDING_MODEL",
    {
        "voyage": "voyage-code-3",
        "local": "BAAI/bge-large-en-v1.5",
        "openai": "text-embedding-3-small",
    }.get(CODE_EMBEDDER_TYPE, "voyage-code-3"),
)
CODE_EMBEDDING_DIM = int(os.environ.get("CODE_EMBEDDING_DIM", str(EMBEDDING_DIM)))

# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------
# RERANKER_TYPE selects the reranking backend:
#   "cohere"  - Cohere API (default, requires COHERE_API_KEY)
#   "none"    - skip reranking entirely
RERANKER_TYPE = os.environ.get("RERANKER_TYPE", "cohere")
RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "rerank-english-v3.0")
