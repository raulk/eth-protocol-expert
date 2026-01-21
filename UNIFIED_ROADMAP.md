# Unified Roadmap

This roadmap consolidates all existing plans (core phases, extended corpus, cache/continuous ingestion, graph integration, LlamaIndex GitHub ingestion, and SOTA upgrades) into one cohesive sequence with clear dependencies.

## Guiding goals
- Accurate, cited answers across Ethereum protocol sources
- Fast, repeatable ingestion with corpus versioning and drift detection
- Scalable retrieval + verification with measurable quality metrics

---

## Phase 0 — Foundations (Prereqs + Eval Harness)
**Purpose**: Establish system health + measurement before behavior changes.

**Deliverables**
- Database + dependencies verified (`docker compose`, `uv sync`).
- Baseline ingestion and query validation (EIP ingest + simple query).
- Evaluation harness (QA set, multi‑hop set, evidence‑labeled set) with versioning (`v1.0 → v1.1 → v2.0`).
- Metrics capture: retrieval recall@k, citation accuracy, NLI agreement, end‑to‑end correctness.

**Sources**: Existing EIP corpus.

**Depends on**: None.

---

## Phase 1 — Storage & Ingestion Infrastructure
**Purpose**: Enable generic documents, caching, and deterministic corpus builds.

**Deliverables**
- Storage schema updates for generic documents + metadata.
- Raw content cache for API sources (arXiv, forums).
- Corpus versioning + drift detection scaffolding.

**Sources**: EIPs + API sources (cached).

**Depends on**: Phase 0.

---

## Phase 2 — Extended Corpus (Git + Specs)
**Purpose**: Expand breadth of protocol sources via git‑based repositories.

**Deliverables**
- ERCs, RIPs ingestion (reuse EIP parser).
- Markdown spec loaders for devp2p + portal‑network‑specs.
- Execution APIs + Beacon APIs (OpenAPI parsing).
- Builder specs ingestion.

**Sources**: ERCs, RIPs, devp2p, portal specs, execution‑apis, beacon‑apis, builder‑specs.

**Depends on**: Phase 1.

---

## Phase 3 — Forum + Research Corpus
**Purpose**: Add design rationale, discussions, and academic context.

**Deliverables**
- Ethresear.ch + Magicians ingestion with raw markdown.
- arXiv ingestion with PDF extraction + quality scoring.
- research repo ingestion (code + PDFs + design notes).

**Sources**: Ethresear.ch, Magicians, arXiv, ethereum/research.

**Depends on**: Phase 1 + cache layer.

---

## Phase 4 — Continuous Ingestion
**Purpose**: Keep corpus up to date with incremental syncs and webhooks.

**Deliverables**
- Ingestion orchestrator + sync state store.
- Source‑specific syncers (GitHub, arXiv, Discourse).
- Optional webhooks for near‑real‑time updates.

**Sources**: All.

**Depends on**: Phases 1–3.

---

## Phase 5 — Graph Integration (FalkorDB)
**Purpose**: Enable dependency traversal and GraphRAG features.

**Deliverables**
- Fix ingestion pipeline for graph population.
- Docker compose integration + API lifespan init.
- Graph endpoints (dependencies, dependents).
- Graph‑augmented query mode.

**Sources**: EIPs + inferred relationships.

**Depends on**: Phase 1, plus EIP corpus.

---

## Phase 6 — Retrieval Upgrades (Hybrid + Rerank)
**Purpose**: Improve recall and ranking quality.

**Deliverables**
- Hybrid search (BM25 + dense + RRF).
- Query expansion (concept resolver + aliases).
- Cross‑encoder reranker with top‑N cap.
- Diversity control (MMR/section coverage).

**Depends on**: Phase 0 metrics + Phase 1 storage.

---

## Phase 7 — Evidence & Validation Upgrades (SOTA)
**Purpose**: Improve grounding precision and reduce unsupported claims.

**Deliverables**
- Span‑level grounding with accurate offsets.
- Best‑span selection per claim + NLI on minimal spans.
- Claim decomposition (dependency‑aware first; LLM for complex only).
- Citation enforcement + verifier pass.

**Depends on**: Phase 6 retrieval.

---

## Phase 8 — Reasoning & Agentic Retrieval
**Purpose**: Improve multi‑hop reasoning and structured outputs.

**Deliverables**
- Planner‑retriever‑synthesizer loop (upgrade existing agents).
- Adaptive budgets and retrieval modes.
- Graph traversal integration for multi‑hop.
- Structured outputs (timeline, argument map, dependency view).

**Depends on**: Phases 5–7.

---

## Phase 9 — Ensemble & Routing
**Purpose**: Cost‑quality optimization and reliability.

**Deliverables**
- Multi‑model routing with confidence calibration.
- Circuit breakers for low‑evidence cases.
- A/B testing + regression gates.

**Depends on**: Evaluation logs from Phases 6–8.

---

## Phase 10 — Client Codebase Analysis
**Purpose**: Link specs to implementations for deep technical queries.

**Deliverables**
- Tree‑sitter parsing + code chunking.
- Code retrieval + EIP‑to‑code linking.
- Spec‑implementation alignment artifacts.

**Depends on**: Phase 1 storage + Phase 6 retrieval.

---

## Cross‑cutting Workstreams
- **Source enhancements (existing corpus)**: add structured params, normative tags, timelines, consensus extraction, and citation graphs per source.
- **Schema changes**: document + chunk metadata, graph edge labels.
- **Safety**: prompt‑injection stripping + trust‑tier weighting.
- **Observability**: per‑query traces, evidence coverage metrics.

---

## Milestone Gates
- M0: Baseline eval (Phase 0) complete with v1.0 dataset.
- M1: Cache + generic documents + extended corpus ingested.
- M2: Continuous ingestion stable with sync state + metrics.
- M3: Graph integration and graph‑augmented retrieval live.
- M4: Retrieval + evidence upgrades hit quality targets.
- M5: Agentic + ensemble upgrades with regression gates.
- M6: Codebase analysis integrated.

---

## Phase Checklists (No Owners/Estimates)

### Phase 0 — Foundations
- Create `tests/eval/` datasets (`v1.0` seed + multi‑hop).
- Add evaluation scripts for recall@k, citation accuracy, NLI agreement.
- Record baseline metrics to `data/eval/metrics.json`.
- Validate ingestion + query CLI happy‑path.

### Phase 1 — Storage & Ingestion Infrastructure
- Add/verify `documents.metadata` JSONB fields in schema.
- Add `chunks.metadata` JSONB column in schema.
- Implement raw content cache for API sources.
- Add drift detection hooks for corpus changes.

### Phase 2 — Extended Corpus (Git + Specs)
- Implement ERC + RIP loaders and ingestion scripts.
- Implement markdown spec loader + devp2p + portal loaders.
- Implement execution‑apis + beacon‑apis loaders (OpenAPI parsing).
- Implement builder‑specs loader.

### Phase 3 — Forum + Research Corpus
- Implement ethresear.ch + Magicians loaders with raw markdown.
- Implement arXiv fetch + PDF extraction + quality scoring.
- Implement research repo loader (code + docs).

### Phase 4 — Continuous Ingestion
- Create orchestrator + sync state store.
- Implement GitHub, arXiv, Discourse syncers.
- Add webhook endpoints for GitHub (optional).
- Add metrics/alerts for sync health.

### Phase 5 — Graph Integration (FalkorDB)
- Fix ingestion pipeline to populate graph edges.
- Integrate FalkorDB in docker-compose + API lifespan.
- Add graph endpoints for dependencies/dependents.
- Add graph‑augmented query mode.

### Phase 6 — Retrieval Upgrades
- Enable hybrid retrieval (BM25 + dense + RRF).
- Integrate query expansion via concept resolver.
- Add cross‑encoder reranker with top‑N cap.
- Add MMR/section coverage selection.

### Phase 7 — Evidence & Validation Upgrades
- Implement span‑level grounding with offsets.
- Add best‑span selection per claim.
- Update NLI to validate minimal spans.
- Add claim decomposition (dependency‑aware + LLM fallback).
- Enforce citations per factual sentence; add verifier pass.

### Phase 8 — Reasoning & Agentic Retrieval
- Upgrade existing ReAct loop into planner‑retriever‑synthesizer.
- Add adaptive budgeting and retrieval mode selection.
- Integrate graph traversal into multi‑hop reasoning.
- Enable structured outputs (timeline, argument map, dependency view).

### Phase 9 — Ensemble & Routing
- Implement confidence calibration using eval logs.
- Add cost‑quality routing and circuit breakers.
- Add A/B testing + regression gates.

### Phase 10 — Client Codebase Analysis
- Add tree‑sitter parsing + code chunking.
- Add code retrieval and spec‑impl linking.
- Expose code‑aware queries in retrieval layer.

---

## Phase → File Map (Implementation Targets)

### Phase 0
- `tests/` (new eval datasets + scripts)
- `scripts/query_cli.py`
- `scripts/ingest_eips.py`

### Phase 1
- `src/storage/pg_vector_store.py`
- `src/ingestion/cache.py`
- `src/ingestion/*` (metadata population)

### Phase 2
- `src/ingestion/erc_loader.py`
- `src/ingestion/rip_loader.py`
- `src/ingestion/markdown_spec_loader.py`
- `src/ingestion/devp2p_loader.py`
- `src/ingestion/portal_spec_loader.py`
- `src/ingestion/execution_apis_loader.py`
- `src/ingestion/beacon_apis_loader.py`
- `src/ingestion/builder_specs_loader.py`
- `scripts/ingest_ercs.py`
- `scripts/ingest_rips.py`
- `scripts/ingest_devp2p.py`
- `scripts/ingest_portal_specs.py`
- `scripts/ingest_execution_apis.py`
- `scripts/ingest_beacon_apis.py`
- `scripts/ingest_builder_specs.py`

### Phase 3
- `src/ingestion/ethresearch_loader.py`
- `src/ingestion/magicians_loader.py`
- `src/ingestion/arxiv_fetcher.py`
- `src/ingestion/pdf_extractor.py`
- `src/ingestion/research_loader.py`
- `scripts/ingest_ethresearch.py`
- `scripts/ingest_magicians.py`
- `scripts/ingest_arxiv.py`
- `scripts/ingest_research.py`

### Phase 4
- `src/ingestion/orchestrator.py`
- `src/ingestion/metrics.py`
- `scripts/continuous_ingestion.py`
- `src/api/main.py` (webhooks + triggers)

### Phase 5
- `src/graph/falkordb_store.py`
- `src/graph/eip_graph_builder.py`
- `src/graph/dependency_traverser.py`
- `src/api/main.py` (graph endpoints)

### Phase 6
- `src/retrieval/hybrid_retriever.py`
- `src/retrieval/bm25_retriever.py`
- `src/concepts/concept_resolver.py`
- `src/concepts/query_expander.py`
- `src/retrieval/reranker.py` (new)

### Phase 7
- `src/generation/validated_generator.py`
- `src/validation/nli_validator.py`
- `src/validation/claim_decomposer.py`
- `src/evidence/evidence_span.py`

### Phase 8
- `src/agents/react_agent.py`
- `src/agents/retrieval_tool.py`
- `src/agents/reflection.py`
- `src/structured/timeline_builder.py`
- `src/structured/argument_mapper.py`
- `src/structured/dependency_view.py`

### Phase 9
- `src/ensemble/cost_router.py`
- `src/ensemble/confidence_scorer.py`

### Phase 10
- `src/parsing/treesitter_parser.py`
- `src/parsing/code_unit_extractor.py`
- `src/graph/spec_impl_linker.py`
- `src/retrieval/code_retriever.py`

---

## Phase Validation Criteria

### Phase 0
- Eval harness runs end‑to‑end on `v1.0` dataset.
- Baseline metrics captured in `data/eval/metrics.json`.
- EIP ingest + simple query complete without error.
- Validation scripts:
  - `scripts/ingest_eips.py`
  - `scripts/query_cli.py`
  - `scripts/test_e2e.py`

### Phase 1
- Schema migrations applied with no data loss.
- Cache layer successfully reuses API content on second ingest run.
- Drift detection hooks report no false positives on stable corpus.
- Validation scripts:
  - `scripts/ingest_arxiv.py` (run twice to confirm cache hits)
  - `scripts/ingest_ethresearch.py` (run twice to confirm cache hits)
  - `scripts/ingest_magicians.py` (run twice to confirm cache hits)

### Phase 2
- ERC/RIP/spec/API ingestions complete and count matches expected ranges.
- Sample queries return cross‑source results (EIP + spec + API).
- No ingestion errors in logs for git sources.
- Validation scripts:
  - `scripts/ingest_ercs.py`
  - `scripts/ingest_rips.py`
  - `scripts/ingest_devp2p.py`
  - `scripts/ingest_portal_specs.py`
  - `scripts/ingest_execution_apis.py`
  - `scripts/ingest_beacon_apis.py`
  - `scripts/ingest_builder_specs.py`

### Phase 3
- Forum + arXiv + research ingestions complete with cache hits.
- PDF extraction quality scores meet minimum threshold.
- Sample forum queries return raw markdown‑backed citations.
- Validation scripts:
  - `scripts/ingest_ethresearch.py`
  - `scripts/ingest_magicians.py`
  - `scripts/ingest_arxiv.py`
  - `scripts/ingest_research.py`

### Phase 4
- Orchestrator syncs run with state updates in `data/sync_state.json`.
- Incremental syncs only ingest changes since last run.
- Metrics emitted for duration, docs synced, and errors.
- Validation scripts:
  - `scripts/continuous_ingestion.py`

### Phase 5
- Graph DB reachable and populated with dependency edges.
- Dependency endpoints return expected counts.
- Graph‑augmented query mode returns additional relevant context.
- Validation scripts:
  - `scripts/query_cli.py` (graph‑augmented mode when available)

### Phase 6
- Retrieval recall@k improves or stays flat vs baseline.
- Reranker latency within budget (top‑N cap respected).
- Query expansion improves recall on alias tests.
- Validation scripts:
  - `scripts/test_full_rag.py`

### Phase 7
- Citation accuracy improves or stays flat vs baseline.
- Unsupported claim rate <= 10% on eval set.
- Span‑level validation passes edge‑case tests.
- Validation scripts:
  - `scripts/test_full_rag.py`

### Phase 8
- Multi‑hop success rate improves vs baseline.
- Agent loops respect budget limits and exit cleanly.
- Structured outputs produce valid schemas.
- Validation scripts:
  - `scripts/test_full_rag.py`

### Phase 9
- Calibration curves fit with eval logs and reduce overconfidence.
- A/B tests show no regression in accuracy or latency.
- Circuit breakers trigger on low‑evidence cases.
- Validation scripts:
  - `scripts/test_full_rag.py` (baseline comparison)

### Phase 10
- Code ingestion completes for at least one client repo.
- Code retrieval returns relevant functions for EIP queries.
- Spec‑impl links created with traceable evidence.
- Validation scripts:
  - `scripts/test_full_rag.py` (code retrieval path)
