# Ingestion Reference

This document describes every ingestion script, what it ingests, how it chunks,
which embedding model it uses, which database schema it writes to, and the
services/tokens it requires.

All ingesters write into PostgreSQL via `PgVectorStore` unless noted otherwise.

---

## Storage Schema (PostgreSQL + pgvector)

Created/maintained by `PgVectorStore.initialize_schema()` in
`src/storage/pg_vector_store.py`.

### documents table

```
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    document_id VARCHAR(128) UNIQUE NOT NULL,
    eip_number INTEGER,
    title VARCHAR(512),
    status VARCHAR(64),
    type VARCHAR(64),
    category VARCHAR(64),
    author TEXT,
    created_date VARCHAR(32),
    requires INTEGER[],
    git_commit VARCHAR(64),
    raw_content TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    document_type VARCHAR(64) DEFAULT 'eip',
    source VARCHAR(128),
    metadata JSONB DEFAULT '{}'
)
```

Notes:
- EIP-specific ingesters use `store_document()` (populates EIP fields).
- Non-EIP ingesters use `store_generic_document()` (populates document_type, source, metadata).
- `document_id` is the primary stable identifier across all sources.

### chunks table

```
CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    chunk_id VARCHAR(512) UNIQUE NOT NULL,
    document_id VARCHAR(128) NOT NULL,
    content TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    section_path VARCHAR(256),
    embedding vector(1024),
    git_commit VARCHAR(64),
    created_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
)
```

Notes:
- `embedding` is 1024 dims for Voyage `voyage-4-large` by default.
- `section_path` is used for citations and navigation (when available).
- Some ingesters add `metadata.document_type` for chunk-level filtering.

---

## Embedding Models

### Voyage AI (default)
- Class: `src/embeddings/voyage_embedder.py` (`VoyageEmbedder`)
- Model: `voyage-4-large`
- Dimension: 1024
- Token: `VOYAGE_API_KEY`

### Local (optional)
- Class: `src/embeddings/local_embedder.py` (`LocalEmbedder`)
- Model: `BAAI/bge-large-en-v1.5`
- Token: none (downloads model weights locally)
- Used in `ingest_eips.py` and `ingest_acd_transcripts.py` via `--local`
- Note: requires `sentence-transformers` installed and downloads model weights on first run

---

## Chunking Strategies

- `SectionChunker` (`src/chunking/section_chunker.py`)
  - Section-aware markdown chunking
  - Keeps code blocks atomic
  - Default: 512 tokens, 64 overlap

- `FixedChunker` (`src/chunking/fixed_chunker.py`)
  - Fixed-size token windows
  - Default: 512 tokens, 64 overlap

- `ForumChunker` (`src/chunking/forum_chunker.py`)
  - Preserves post boundaries
  - Adds post metadata to chunk content
  - Splits long posts at paragraph boundaries

- `TranscriptChunker` (`src/chunking/transcript_chunker.py`)
  - Speaker-aware chunking for ACD transcripts

- `PaperChunker` (`src/chunking/paper_chunker.py`)
  - Section-aware PDF chunking
  - Preserves equations/figures when possible

- `TextChunker` (`src/chunking/text_chunker.py`)
  - Generic markdown/text chunking
  - Paragraph-aware

- Custom chunkers in scripts
  - `chunk_markdown()` for consensus specs
  - `chunk_python_code()` for execution specs

---

## External Services and Tokens

Required per source (as used in code):

- PostgreSQL + pgvector
  - `DATABASE_URL` (defaults to `postgresql://postgres:postgres@localhost:5432/eth_protocol`)

- Voyage AI embeddings
  - `VOYAGE_API_KEY`

- GitHub API (issues/PRs only)
  - `GITHUB_TOKEN`

- FalkorDB (graph)
  - `FALKORDB_HOST` (default `localhost`)
  - `FALKORDB_PORT` (default `6379`)

No tokens required:
- Git clone sources (public repos)
- arXiv API
- Discourse API (ethresear.ch, ethereum-magicians.org) in current code

---

## Ingesters (by script)

Each entry below lists: source(s), document_id format, document_type,
chunking, embedding model, and required services/tokens.

### `scripts/ingest_eips.py`
- Source: `ethereum/EIPs` repo (git clone)
- document_id: `eip-{number}`
- document_type: `eip` (default in schema)
- Chunking: `SectionChunker` by default (`--use-fixed-chunking` uses `FixedChunker`)
- Embedding: `VoyageEmbedder` (default) or `LocalEmbedder` with `--local`
- Services/tokens: PostgreSQL; optional FalkorDB for graph build; `VOYAGE_API_KEY` if not local
- Extra: builds EIP dependency graph in FalkorDB (`REQUIRES`, `SUPERSEDES`, `REPLACES`)

### `scripts/ingest_ercs.py`
- Source: `ethereum/EIPs` (category ERC) and `ethereum/ERCs`
- document_id: `erc-{number}`
- document_type: `erc`
- Chunking: `SectionChunker` (ERCs parsed as EIPs)
- Embedding: `VoyageEmbedder`
- Services/tokens: PostgreSQL; `VOYAGE_API_KEY`

### `scripts/ingest_rips.py`
- Source: `ethereum/RIPs`
- document_id: `rip-{number}`
- document_type: `rip`
- Chunking: `SectionChunker` (RIPs parsed as EIPs)
- Embedding: `VoyageEmbedder`
- Services/tokens: PostgreSQL; `VOYAGE_API_KEY`

### `scripts/ingest_consensus_specs.py`
- Source: `ethereum/consensus-specs`
- document_id: `consensus-spec-{fork}-{name}`
- document_type: `consensus_spec`
- Chunking: custom `chunk_markdown()` (splits on headings, then paragraphs)
- Embedding: `VoyageEmbedder`
- Services/tokens: PostgreSQL; `VOYAGE_API_KEY`

### `scripts/ingest_execution_specs.py`
- Source: `ethereum/execution-specs`
- document_id: `execution-spec-{fork}-{module_path}`
- document_type: `execution_spec`
- Chunking: `chunk_python_code()` (fixed-size tokens with overlap)
- Embedding: `VoyageEmbedder`
- Services/tokens: PostgreSQL; `VOYAGE_API_KEY`

### `scripts/ingest_devp2p.py`
- Source: `ethereum/devp2p`
- document_id: `devp2p-{spec_name}`
- document_type: `devp2p_spec`
- Chunking: `SectionChunker` via pseudo-ParsedEIP adapter
- Embedding: `VoyageEmbedder`
- Services/tokens: PostgreSQL; `VOYAGE_API_KEY`

### `scripts/ingest_portal_specs.py`
- Source: `ethereum/portal-network-specs`
- document_id: `portal-{spec_name}`
- document_type: `portal_spec`
- Chunking: `SectionChunker` via pseudo-ParsedEIP adapter
- Embedding: `VoyageEmbedder`
- Services/tokens: PostgreSQL; `VOYAGE_API_KEY`

### `scripts/ingest_builder_specs.py`
- Source: `ethereum/builder-specs`
- document_id: `builder-spec-{spec_name}`
- document_type: `builder_spec`
- Chunking: `SectionChunker` via pseudo-ParsedEIP adapter
- Embedding: `VoyageEmbedder`
- Services/tokens: PostgreSQL; `VOYAGE_API_KEY`

### `scripts/ingest_execution_apis.py`
- Source: `ethereum/execution-apis` (OpenAPI -> markdown)
- document_id: `exec-api-{doc_type}-{title}` (sanitized)
- document_type: `execution_api`
- Chunking: `SectionChunker` via pseudo-ParsedEIP adapter
- Embedding: `VoyageEmbedder`
- Services/tokens: PostgreSQL; `VOYAGE_API_KEY`

### `scripts/ingest_beacon_apis.py`
- Source: `ethereum/beacon-APIs` (OpenAPI -> markdown)
- document_id: `beacon-api-{METHOD}-{path}`
- document_type: `beacon_api`
- Chunking: `SectionChunker` via pseudo-ParsedEIP adapter
- Embedding: `VoyageEmbedder`
- Services/tokens: PostgreSQL; `VOYAGE_API_KEY`

### `scripts/ingest_research.py`
- Source: `ethereum/research` (markdown + python)
- document_id: `research-{category}-{name}` (hashed if long)
- document_type: `research`
- Chunking: `SectionChunker` via pseudo-ParsedEIP adapter
- Embedding: `VoyageEmbedder`
- Services/tokens: PostgreSQL; `VOYAGE_API_KEY`

### `scripts/ingest_arxiv.py`
- Source: arXiv API + PDF downloads
- document_id: `arxiv-{arxiv_id}`
- document_type: `arxiv_paper`
- Chunking: `PaperChunker` (section-aware, keeps equations/figures)
- Embedding: `VoyageEmbedder`
- Services/tokens: PostgreSQL; `VOYAGE_API_KEY`; external arXiv HTTP

### `scripts/ingest_acd_transcripts.py`
- Source: `ethereum/pm` (ACD transcripts)
- document_id: `acd-call-{call_number}`
- document_type: `acd_transcript`
- Chunking: `TranscriptChunker` (speaker-aware)
- Embedding: `VoyageEmbedder` or `LocalEmbedder` with `--local`
- Services/tokens: PostgreSQL; `VOYAGE_API_KEY` if not local

### `scripts/ingest_ethresearch.py`
- Source: ethresear.ch cache (must run `scripts/sync_ethresearch.py` first)
- document_id: `ethresearch-topic-{topic_id}`
- document_type: `forum_topic`
- Chunking: `ForumChunker` (post-aware)
- Embedding: `VoyageEmbedder`
- Services/tokens: PostgreSQL; `VOYAGE_API_KEY`; Discourse API used by sync step

### `scripts/ingest_magicians.py`
- Source: ethereum-magicians.org (Discourse API, live)
- document_id: `magicians-topic-{topic_id}`
- document_type: `forum_topic`
- Chunking: `ForumChunker` (post-aware)
- Embedding: `VoyageEmbedder`
- Services/tokens: PostgreSQL; `VOYAGE_API_KEY`; Discourse API (no token in code)

### `scripts/ingest_github_issues.py`
- Source: GitHub Issues + PRs (REST API)
- document_id:
  - Issues: `github-{owner}-{repo}-issue-{number}`
  - PRs: `github-{owner}-{repo}-pr-{number}`
- document_type: `github_issue` or `github_pr`
- Chunking: `TextChunker` (paragraph-aware markdown)
- Embedding: `VoyageEmbedder`
- Services/tokens: PostgreSQL; `VOYAGE_API_KEY`; GitHub API (`GITHUB_TOKEN`)
- Incremental cursor stored in `data/sync_state.json` under `github_issues/{owner}/{repo}`

### `scripts/ingest_client.py`
- Source: Ethereum client codebases (git clone) via Tree-sitter parsing
- document_id: `code:{repo_name}`
- document_type: uses `store_document()` with `type_="codebase"` and `category={language}`
- Chunking: `code_unit_to_chunk()` (function/struct units extracted by Tree-sitter)
- Embedding: `VoyageEmbedder`
- Services/tokens: PostgreSQL; `VOYAGE_API_KEY`;
  optional FalkorDB when linking EIPs to implementations

---

## Orchestrators (not primary ingesters)

- `scripts/ingest_all.py`
  - Runs multiple ingesters in sequence (EIPs, forums, transcripts, arXiv, specs)

- `scripts/continuous_ingestion.py`
  - Incremental sync for git-based sources and forums
  - Calls `scripts/ingest_github_issues.py` for GitHub issues/PRs

- `scripts/ingest_all_clients.sh`
  - Shell wrapper to run `scripts/ingest_client.py` for all configured clients

---

## Quick Reference: Tokens / Env Vars

| Variable | Used By | Purpose |
|---|---|---|
| `DATABASE_URL` | All ingesters | PostgreSQL connection |
| `VOYAGE_API_KEY` | Most ingesters | Voyage embeddings |
| `GITHUB_TOKEN` | `ingest_github_issues.py` | GitHub Issues/PRs API |
| `FALKORDB_HOST`, `FALKORDB_PORT` | `ingest_eips.py`, `ingest_client.py` | Graph storage |
