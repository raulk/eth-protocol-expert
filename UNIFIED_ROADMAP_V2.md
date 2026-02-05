# Unified Roadmap v2 (Impact-Ordered)

Date: 2026-01-23

## Baseline (observed in code)
- Core RAG pipeline with citations + NLI validation + claim decomposition.
- Hybrid retrieval (BM25 + vectors + rerank), agentic mode, ensembles.
- Graph integration with FalkorDB and EIP dependency endpoints.
- Ingestion scripts for EIPs, ERCs, RIPs, specs, forums, arXiv, research, and client codebases.
- Evaluation harness in `tests/eval/`.

## Impact-Ordered Roadmap

### 1) Corpus Freshness + Safety (P0)
**Why high impact**: Stale or unsafe sources degrade answer quality and trust.

**Scope**
- Add RawContentCache to arXiv and Magicians ingestion.
- Implement `sync_magicians.py` (mirror `sync_ethresearch.py`) and update `ingest_magicians.py` to read cache only.
- Expand `scripts/continuous_ingestion.py` to include arXiv + Magicians + remaining sources, with per-source cursors.
- Add forum content sanitization (strip prompt-injection patterns, scripts, HTML artifacts).
- Add trust-tier weighting in retrieval (EIPs/specs > APIs > forums > issues/PRs).

**Success criteria**
- All sources can sync incrementally within 24 hours.
- Cache hit rate > 90% on re-ingest.
- Forum content is sanitized before storage and retrieval.

---

### 2) GitHub Issues/PRs Ingestion (P0/P1)
**Why high impact**: Design rationale, debates, and decisions live in GitHub threads.

**Scope**
- Implement GitHub loader (LlamaIndex readers or native GitHub API).
- Store issues/PRs as generic documents with `document_type` (`github_issue`, `github_pr`).
- Add ingestion script + incremental sync by `updated_at` cursor.
- Add retrieval filters to avoid flooding results (top-N caps, labels, or recency).

**Success criteria**
- Queries about EIP rationale surface issues/PRs alongside specs.
- Incremental sync updates issues/PRs without re-ingesting everything.

---

### 3) Structured Metadata Extraction (P1)
**Why high impact**: Improves precision for parameter, fork, and normative queries.

**Scope**
- EIPs: extract activation blocks, constants, gas schedule changes, normative keywords.
- Consensus/execution specs: extract constants, state transitions, per-fork diffs.
- ACD transcripts: extract decisions, action items, and mapping to forks/EIPs.
- Store in `documents.metadata` and `chunks.metadata`; backfill existing corpus.
- Enable metadata-aware retrieval filters and query routing.

**Success criteria**
- Metadata filters return correct subsets (e.g., fork-specific queries).
- Retrieval recall improves on structured queries (measured via eval harness).

---

### 4) Drift-Aware Incremental Re-Embedding (P1)
**Why impact**: Reduces ingest time/cost and prevents stale chunks.

**Scope**
- Use cache drift detection + git diffs to identify changed docs.
- Add chunk-level upserts and deletion of stale chunks in PgVectorStore.
- Track document hash + last_processed in metadata.

**Success criteria**
- Re-ingest time reduced by >60% on unchanged corpora.
- No orphaned chunks after updates.

---

### 5) Evaluation Automation + Regression Gates (P2)
**Why impact**: Prevents silent quality regressions.

**Scope**
- CI job to run `tests/eval/run_eval.py` with baseline thresholds.
- Store metrics snapshots with timestamps in `data/eval/`.
- Add regression gate for citation accuracy + retrieval recall.

**Success criteria**
- Failing evals block regressions.
- Metrics trend visible over time.

---

### 6) Documentation & Roadmap Consolidation (P3)
**Why impact**: Reduces confusion and keeps execution aligned.

**Scope**
- Deprecate older roadmap docs or clearly mark them as historical.
- Update README feature claims after P0/P1 changes land.

**Success criteria**
- Single authoritative roadmap (this file).
- README reflects real capabilities and counts.

