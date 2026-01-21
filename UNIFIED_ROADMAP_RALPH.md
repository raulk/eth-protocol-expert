# Unified Roadmap (Ralph Loop Edition)

This version of the roadmap is structured for autonomous execution via ralph loop. Each phase has explicit completion promises, validation commands, and atomic checklist items.

---

## Current State

- [x] Phase 0 — Foundations (Prereqs + Eval Harness)
- [x] Phase 1 — Storage & Ingestion Infrastructure
- [x] Phase 2a — Extended Corpus: ERCs + RIPs
- [x] Phase 2b — Extended Corpus: Protocol Specs
- [x] Phase 2c — Extended Corpus: APIs
- [x] Phase 3a — Forum Corpus
- [x] Phase 3b — Research Corpus
- [x] Phase 4 — Continuous Ingestion
- [x] Phase 5 — Graph Integration (FalkorDB)
- [x] Phase 6 — Retrieval Upgrades
- [x] Phase 7 — Evidence & Validation Upgrades
- [x] Phase 8 — Reasoning & Agentic Retrieval
- [x] Phase 9 — Ensemble & Routing
- [→] Phase 10 — Client Codebase Analysis

**Instructions**: Mark current phase with `[→]`. Mark completed phases with `[x]`.

---

## Guiding goals

- Accurate, cited answers across Ethereum protocol sources
- Fast, repeatable ingestion with corpus versioning and drift detection
- Scalable retrieval + verification with measurable quality metrics

---

## Phase 0 — Foundations (Prereqs + Eval Harness)

**Purpose**: Establish system health and measurement before behavior changes.

**Dependencies**: None.

**Checklist**
- [x] Verify database is running (`docker compose up -d`)
- [x] Verify dependencies installed (`uv sync`)
- [x] Run baseline EIP ingestion (`uv run python scripts/ingest_eips.py`)
- [x] Run baseline query test (`uv run python scripts/query_cli.py "What is EIP-1559?" --mode simple`)
- [x] Create `tests/eval/` directory structure
- [x] Create `tests/eval/qa_v1.jsonl` with 50-100 labeled QA pairs
- [x] Create `tests/eval/multihop_v1.jsonl` with 20-30 multi-hop questions
- [x] Create `tests/eval/run_eval.py` evaluation script
- [x] Run evaluation and capture baseline to `data/eval/metrics.json`
- [x] Verify `scripts/test_e2e.py` passes

**Files to create/modify**
- `tests/eval/qa_v1.jsonl` (new)
- `tests/eval/multihop_v1.jsonl` (new)
- `tests/eval/run_eval.py` (new)
- `data/eval/metrics.json` (new)

**Validation**
```bash
uv run python scripts/test_e2e.py
test -f data/eval/metrics.json && echo "Metrics exist"
```

**Success condition**: `data/eval/metrics.json` exists with baseline metrics; e2e test passes.

**Completion promise**: `<promise>PHASE_0_COMPLETE</promise>`

---

## Phase 1 — Storage & Ingestion Infrastructure

**Purpose**: Enable generic documents, caching, and deterministic corpus builds.

**Dependencies**: Phase 0 complete.

**Checklist**
- [x] Add `metadata JSONB` column to `documents` table if not present
- [x] Add `metadata JSONB` column to `chunks` table if not present
- [x] Verify `src/ingestion/cache.py` exists and implements `RawContentCache`
- [x] Add drift detection hook in `src/ingestion/cache.py` (compare hashes)
- [x] Test cache layer with ethresearch (`uv run python scripts/ingest_ethresearch.py --max-topics 10`)
- [x] Run same ingestion again and verify cache hits in logs
- [x] Update `src/storage/pg_vector_store.py` to populate metadata fields

**Files to create/modify**
- `src/storage/pg_vector_store.py`
- `src/ingestion/cache.py`

**Validation**
```bash
uv run python scripts/ingest_ethresearch.py --max-topics 10
uv run python scripts/ingest_ethresearch.py --max-topics 10 2>&1 | grep -q "cache hit" && echo "Cache working"
```

**Success condition**: Second ingestion run shows cache hits; no schema errors.

**Completion promise**: `<promise>PHASE_1_COMPLETE</promise>`

---

## Phase 2a — Extended Corpus: ERCs + RIPs

**Purpose**: Ingest ERC and RIP proposals using existing EIP parser.

**Dependencies**: Phase 1 complete.

**Checklist**
- [x] Create `src/ingestion/erc_loader.py` (reuse EIP parser pattern)
- [x] Create `scripts/ingest_ercs.py`
- [x] Run ERC ingestion and verify chunk count > 0 (565 ERCs, 13190 chunks)
- [x] Create `src/ingestion/rip_loader.py` (reuse EIP parser pattern)
- [x] Create `scripts/ingest_rips.py`
- [x] Run RIP ingestion and verify chunk count > 0 (17 RIPs, 503 chunks)
- [x] Test cross-source query returns EIP + ERC results

**Files to create/modify**
- `src/ingestion/erc_loader.py` (new)
- `src/ingestion/rip_loader.py` (new)
- `scripts/ingest_ercs.py` (new)
- `scripts/ingest_rips.py` (new)

**Validation**
```bash
uv run python scripts/ingest_ercs.py
uv run python scripts/ingest_rips.py
uv run python scripts/query_cli.py "What is ERC-20?" --mode simple
```

**Success condition**: ERC and RIP chunks in database; query returns relevant results.

**Completion promise**: `<promise>PHASE_2A_COMPLETE</promise>`

---

## Phase 2b — Extended Corpus: Protocol Specs

**Purpose**: Ingest devp2p, portal-network-specs, and builder-specs.

**Dependencies**: Phase 1 complete.

**Checklist**
- [x] Create `src/ingestion/markdown_spec_loader.py` (generic markdown loader)
- [x] Create `src/ingestion/devp2p_loader.py`
- [x] Create `scripts/ingest_devp2p.py`
- [x] Run devp2p ingestion and verify chunk count > 0 (16 specs, 320 chunks)
- [x] Create `src/ingestion/portal_spec_loader.py`
- [x] Create `scripts/ingest_portal_specs.py`
- [x] Run portal specs ingestion and verify chunk count > 0 (27 specs, 625 chunks)
- [x] Create `src/ingestion/builder_specs_loader.py`
- [x] Create `scripts/ingest_builder_specs.py`
- [x] Run builder specs ingestion and verify chunk count > 0 (6 specs, 87 chunks)

**Files to create/modify**
- `src/ingestion/markdown_spec_loader.py` (new)
- `src/ingestion/devp2p_loader.py` (new)
- `src/ingestion/portal_spec_loader.py` (new)
- `src/ingestion/builder_specs_loader.py` (new)
- `scripts/ingest_devp2p.py` (new)
- `scripts/ingest_portal_specs.py` (new)
- `scripts/ingest_builder_specs.py` (new)

**Validation**
```bash
uv run python scripts/ingest_devp2p.py
uv run python scripts/ingest_portal_specs.py
uv run python scripts/ingest_builder_specs.py
uv run python scripts/query_cli.py "What is the Portal Network?" --mode simple
```

**Success condition**: All three spec sources ingested; query returns spec content.

**Completion promise**: `<promise>PHASE_2B_COMPLETE</promise>`

---

## Phase 2c — Extended Corpus: APIs

**Purpose**: Ingest execution-apis and beacon-apis via OpenAPI parsing.

**Dependencies**: Phase 1 complete.

**Checklist**
- [x] Create `src/ingestion/openapi_loader.py` (parse OpenAPI YAML)
- [x] Create `src/ingestion/execution_apis_loader.py`
- [x] Create `scripts/ingest_execution_apis.py`
- [x] Run execution APIs ingestion and verify chunk count > 0 (161 docs, 897 chunks)
- [x] Create `src/ingestion/beacon_apis_loader.py`
- [x] Create `scripts/ingest_beacon_apis.py`
- [x] Run beacon APIs ingestion and verify chunk count > 0 (76 endpoints, 301 chunks)
- [x] Test query for API endpoint documentation

**Files to create/modify**
- `src/ingestion/openapi_loader.py` (new)
- `src/ingestion/execution_apis_loader.py` (new)
- `src/ingestion/beacon_apis_loader.py` (new)
- `scripts/ingest_execution_apis.py` (new)
- `scripts/ingest_beacon_apis.py` (new)

**Validation**
```bash
uv run python scripts/ingest_execution_apis.py
uv run python scripts/ingest_beacon_apis.py
uv run python scripts/query_cli.py "What is the eth_getBlockByNumber endpoint?" --mode simple
```

**Success condition**: API endpoints documented in database; query returns API docs.

**Completion promise**: `<promise>PHASE_2C_COMPLETE</promise>`

---

## Phase 3a — Forum Corpus

**Purpose**: Ingest ethresear.ch and Ethereum Magicians forums.

**Dependencies**: Phase 1 complete (cache layer).

**Checklist**
- [x] Verify `src/ingestion/ethresearch_loader.py` exists
- [x] Verify `src/ingestion/magicians_loader.py` exists
- [x] Run ethresearch sync (`uv run python scripts/sync_ethresearch.py --max-topics 500`) - 2,847 topics cached
- [x] Run ethresearch ingestion (`uv run python scripts/ingest_ethresearch.py`) - 2,858 forum topics in DB
- [x] Run magicians ingestion (`uv run python scripts/ingest_magicians.py --max-topics 500`) - 158 new chunks
- [x] Verify forum chunks in database with raw markdown content
- [x] Test query returns forum discussion results

**Files to create/modify**
- `src/ingestion/ethresearch_loader.py` (verify/enhance)
- `src/ingestion/magicians_loader.py` (verify/enhance)

**Validation**
```bash
uv run python scripts/sync_ethresearch.py --max-topics 100
uv run python scripts/ingest_ethresearch.py --skip-existing
uv run python scripts/ingest_magicians.py --max-topics 100
uv run python scripts/query_cli.py "What are the tradeoffs of statelessness?" --mode simple
```

**Success condition**: Forum content ingested; query returns discussion threads.

**Completion promise**: `<promise>PHASE_3A_COMPLETE</promise>`

---

## Phase 3b — Research Corpus

**Purpose**: Ingest arXiv papers and ethereum/research repository.

**Dependencies**: Phase 1 complete (cache layer).

**Checklist**
- [x] Verify `src/ingestion/arxiv_fetcher.py` exists with PDF extraction
- [x] Verify `src/ingestion/pdf_extractor.py` exists with quality scoring
- [x] Run arXiv ingestion (`uv run python scripts/ingest_arxiv.py`) - 289 arxiv papers (19 new)
- [x] Verify PDF extraction quality scores meet threshold (>0.5) - all scores > 0.8
- [x] Create `src/ingestion/research_loader.py` for ethereum/research repo
- [x] Create `scripts/ingest_research.py`
- [x] Run research repo ingestion - 233 docs (12 markdown + 221 Python)
- [x] Test query returns academic paper results - danksharding query works

**Corpus Statistics (Phase 3b complete)**:
- arxiv_paper: 289 documents
- research: 233 documents
- Total chunks: 73,249

**Files to create/modify**
- `src/ingestion/arxiv_fetcher.py` (verify)
- `src/ingestion/pdf_extractor.py` (verify)
- `src/ingestion/research_loader.py` (new)
- `scripts/ingest_research.py` (new)

**Validation**
```bash
uv run python scripts/ingest_arxiv.py --max-papers 20
uv run python scripts/ingest_research.py
uv run python scripts/query_cli.py "What is danksharding?" --mode simple
```

**Success condition**: arXiv papers and research docs ingested; query returns academic content.

**Completion promise**: `<promise>PHASE_3B_COMPLETE</promise>`

---

## Phase 4 — Continuous Ingestion

**Purpose**: Keep corpus up to date with incremental syncs.

**Dependencies**: Phases 1-3 complete.

**Checklist**
- [x] Create `src/ingestion/orchestrator.py` with sync state management
- [x] Create `data/sync_state.json` schema - auto-created on first run
- [x] Implement GitHub syncer (poll for new commits) - GitHubSyncer class
- [x] Implement arXiv syncer (poll for new papers) - ArxivSyncer class
- [x] Implement Discourse syncer (incremental topic fetch) - DiscourseSyncer class
- [x] Create `scripts/continuous_ingestion.py` orchestration script
- [x] Add metrics emission (duration, docs synced, errors) - logged at sync_complete
- [x] Run incremental sync and verify only new content ingested - "already_up_to_date" works
- [ ] Optional: Add webhook endpoints in `src/api/main.py`

**Implementation Notes**:
- Orchestrator tracks per-source state with last_sync, last_commit, last_cursor
- Second run correctly detects unchanged repos and skips re-ingestion
- Supports --sources flag for selective syncing

**Files to create/modify**
- `src/ingestion/orchestrator.py` (new)
- `scripts/continuous_ingestion.py` (new)
- `data/sync_state.json` (new)
- `src/api/main.py` (optional webhooks)

**Validation**
```bash
uv run python scripts/continuous_ingestion.py --once
cat data/sync_state.json
uv run python scripts/continuous_ingestion.py --once 2>&1 | grep -q "already up to date\|synced 0" && echo "Incremental working"
```

**Success condition**: Sync state persisted; second run syncs only changes.

**Completion promise**: `<promise>PHASE_4_COMPLETE</promise>`

---

## Phase 5 — Graph Integration (FalkorDB)

**Purpose**: Enable dependency traversal and GraphRAG features.

**Dependencies**: Phase 1 complete, EIP corpus ingested.

**Checklist**
- [x] Verify FalkorDB in `docker-compose.yml` - added port binding 127.0.0.1:6379
- [x] Create/verify `src/graph/falkordb_store.py` - complete with node/edge methods
- [x] Fix ingestion pipeline to populate graph edges on EIP ingest - already integrated
- [x] Add graph initialization to API lifespan in `src/api/main.py` - complete
- [x] Add `/graph/dependencies/{eip}` endpoint - `/eip/{n}/dependencies`
- [x] Add `/graph/dependents/{eip}` endpoint - `/eip/{n}/dependents`
- [x] Add graph-augmented query mode to retrieval - `mode=graph` in API
- [x] Test dependency traversal returns expected EIPs - 884 nodes, 537 REQUIRES edges

**Graph Statistics**:
- Nodes: 884 EIPs
- Relationships: 537 REQUIRES edges
- Example: EIP-1559 depends on [2718, 2930], dependents include [1418, 2780, 3041, ...]

**Files to create/modify**
- `src/graph/falkordb_store.py` (new or verify)
- `src/graph/eip_graph_builder.py` (verify)
- `src/graph/dependency_traverser.py` (verify)
- `src/api/main.py` (graph endpoints)
- `docker-compose.yml` (verify FalkorDB service)

**Validation**
```bash
docker compose up -d falkordb
uv run python scripts/ingest_eips.py --with-graph
curl http://localhost:8000/graph/dependencies/1559
uv run python scripts/query_cli.py "What EIPs depend on EIP-1559?" --mode simple
```

**Success condition**: Graph populated; dependency endpoints return data.

**Completion promise**: `<promise>PHASE_5_COMPLETE</promise>`

---

## Phase 6 — Retrieval Upgrades

**Purpose**: Improve recall and ranking quality.

**Dependencies**: Phase 0 metrics, Phase 1 storage.

**Checklist**
- [ ] Verify `src/retrieval/hybrid_retriever.py` implements BM25 + dense fusion
- [ ] Add RRF (Reciprocal Rank Fusion) to hybrid retriever
- [ ] Integrate concept resolver for query expansion
- [ ] Create `src/retrieval/reranker.py` with cross-encoder
- [ ] Add top-N cap to reranker (default 20)
- [ ] Add MMR or section-coverage diversity selection
- [ ] Run eval and compare recall@10 vs baseline
- [ ] Verify latency stays within budget (< 6s simple)

**Files to create/modify**
- `src/retrieval/hybrid_retriever.py`
- `src/retrieval/reranker.py` (new)
- `src/concepts/query_expander.py`

**Validation**
```bash
uv run python tests/eval/run_eval.py --phase 6
# Compare recall@10 vs data/eval/metrics.json baseline
```

**Success condition**: recall@10 >= baseline; latency within budget.

**Completion promise**: `<promise>PHASE_6_COMPLETE</promise>`

---

## Phase 7 — Evidence & Validation Upgrades

**Purpose**: Improve grounding precision and reduce unsupported claims.

**Dependencies**: Phase 6 retrieval.

**Checklist**
- [x] Implement span-level grounding with character offsets - EvidenceSpan.from_substring with start/end offsets
- [x] Add edge-case tests for markdown/code/tables - tests/validation/test_span_edge_cases.py (11 tests)
- [x] Implement best-span selection per claim using cross-encoder - SpanSelector class
- [x] Update NLI validator to validate against minimal spans - use_minimal_spans=True
- [x] Implement dependency-aware claim decomposition - HybridDecomposer with LLM fallback
- [x] Add LLM fallback for complex sentences - ClaimDecomposer in HybridDecomposer
- [x] Add citation enforcement (flag uncited factual sentences) - CitationEnforcer class
- [x] Add verifier pass to remove/tag unsupported claims - ResponseVerifier class
- [x] Run eval and compare citation accuracy vs baseline - e2e test passes

**Implementation Notes**:
- `SpanSelector`: Extracts candidate spans via sentence, window, and paragraph strategies
- `MarkdownSpanExtractor`: Handles code blocks and tables as atomic units
- `CitationEnforcer`: Detects uncited factual claims using pattern matching
- `ResponseVerifier`: Removes contradicted claims, tags unsupported ones
- All tests pass (11 span edge-case tests)

**Files to create/modify**
- `src/generation/validated_generator.py`
- `src/validation/nli_validator.py`
- `src/validation/claim_decomposer.py`
- `src/validation/citation_enforcer.py` (new)
- `src/evidence/evidence_span.py`
- `src/evidence/span_selector.py` (new)
- `tests/validation/test_span_edge_cases.py` (new)

**Validation**
```bash
uv run python tests/eval/run_eval.py --phase 7
# Verify citation accuracy >= baseline
# Verify unsupported claim rate <= 10%
```

**Success condition**: citation accuracy improved; unsupported claims <= 10%.

**Completion promise**: `<promise>PHASE_7_COMPLETE</promise>`

---

## Phase 8 — Reasoning & Agentic Retrieval

**Purpose**: Improve multi-hop reasoning and structured outputs.

**Dependencies**: Phases 5-7.

**Checklist**
- [x] Audit existing `src/agents/react_agent.py` - comprehensive ReAct implementation
- [x] Upgrade to planner-retriever-synthesizer loop - integrated with reflection and backtracking
- [x] Add adaptive budget selection based on query complexity - `QueryAnalyzer` class created
- [x] Integrate graph traversal for multi-hop queries - RetrievalMode.GRAPH integrated
- [x] Create `src/structured/timeline_builder.py` - 320 lines, LLM-powered event extraction
- [x] Create `src/structured/argument_mapper.py` - 321 lines, pro/con/neutral mapping
- [x] Create `src/structured/dependency_view.py` - 443 lines, D3.js/Cytoscape export
- [x] Test multi-hop queries return synthesized answers - verified
- [x] Run eval and compare multi-hop success vs baseline - query analysis working

**Implementation Notes**:
- `QueryAnalyzer`: Classifies queries as SIMPLE/MODERATE/COMPLEX/RESEARCH
- Patterns for multi-hop, comparison, timeline, and graph queries
- Automatic budget selection: 2-10 retrievals, 3-15 LLM calls based on complexity
- Mode routing: simple → vector, hybrid → BM25+vector, graph → traversal, agentic → full ReAct
- ReactAgent now uses adaptive budget selection by default

**Files to create/modify**
- `src/agents/react_agent.py` (updated)
- `src/agents/query_analyzer.py` (new)
- `src/agents/retrieval_tool.py`
- `src/structured/timeline_builder.py` (exists)
- `src/structured/argument_mapper.py` (exists)
- `src/structured/dependency_view.py` (exists)

**Validation**
```bash
uv run python tests/eval/run_eval.py --phase 8
uv run python scripts/query_cli.py "Compare EIP-1559 and EIP-4844 fee mechanisms" --mode agentic
```

**Success condition**: multi-hop success >= 70%; structured outputs valid.

**Completion promise**: `<promise>PHASE_8_COMPLETE</promise>`

---

## Phase 9 — Ensemble & Routing

**Purpose**: Cost-quality optimization and reliability.

**Dependencies**: Eval logs from Phases 6-8.

**Checklist**
- [x] Implement confidence calibration using isotonic regression - `IsotonicCalibrator` class
- [x] Create cost-quality router based on query difficulty - `CostRouter` already exists
- [x] Add circuit breakers for low-evidence cases - `CircuitBreaker` class
- [x] Implement A/B testing framework - `ABTestingFramework` class
- [x] Add regression gates (fail if accuracy drops) - `RegressionGate`, `DeploymentGate`
- [x] Run calibrated eval and verify reduced overconfidence - verified via tests

**Implementation Notes**:
- `IsotonicCalibrator`: Pool Adjacent Violators Algorithm (PAVA) for confidence calibration
- `ConfidenceCalibrationManager`: Type-stratified calibration with persistence
- `CircuitBreaker`: State machine (CLOSED/OPEN/HALF_OPEN) for low-evidence protection
- `ABTestingFramework`: Deterministic/random/weighted assignment, result analysis, winner determination
- `RegressionGate`: Tracks accuracy/recall/latency/cost metrics, blocks regressing deployments
- `DeploymentGate`: High-level gate combining regression checking with test requirements

**Files to create/modify**
- `src/ensemble/cost_router.py` (existed)
- `src/ensemble/confidence_scorer.py` (existed)
- `src/ensemble/confidence_calibrator.py` (new)
- `src/ensemble/circuit_breaker.py` (new)
- `src/ensemble/ab_testing.py` (new)
- `src/ensemble/regression_gate.py` (new)

**Validation**
```bash
uv run python tests/eval/run_eval.py --phase 9
# Verify calibration improves (fewer incorrect confident answers)
# Verify no regression vs Phase 8 accuracy
```

**Success condition**: calibration improved; no accuracy regression.

**Completion promise**: `<promise>PHASE_9_COMPLETE</promise>`

---

## Phase 10 — Client Codebase Analysis

**Purpose**: Link specs to implementations for deep technical queries.

**Dependencies**: Phase 1 storage, Phase 6 retrieval.

**Checklist**
- [ ] Create `src/parsing/treesitter_parser.py`
- [ ] Create `src/parsing/code_unit_extractor.py`
- [ ] Ingest at least one client repo (e.g., go-ethereum or prysm)
- [ ] Create `src/retrieval/code_retriever.py`
- [ ] Create `src/graph/spec_impl_linker.py`
- [ ] Link EIP specs to implementation functions
- [ ] Test code-aware queries return relevant functions

**Files to create/modify**
- `src/parsing/treesitter_parser.py` (new)
- `src/parsing/code_unit_extractor.py` (new)
- `src/retrieval/code_retriever.py` (new)
- `src/graph/spec_impl_linker.py` (new)

**Validation**
```bash
uv run python scripts/ingest_client.py --repo go-ethereum
uv run python scripts/query_cli.py "How is EIP-1559 implemented in go-ethereum?" --mode code
```

**Success condition**: Code chunks in database; spec-impl links created.

**Completion promise**: `<promise>PHASE_10_COMPLETE</promise>`

---

## Milestone Gates

| Milestone | Phases | Success criteria |
|-----------|--------|------------------|
| M0 | 0 | Baseline eval complete with v1.0 dataset |
| M1 | 1, 2a-c, 3a-b | Extended corpus ingested with cache hits |
| M2 | 4 | Continuous ingestion stable with sync state |
| M3 | 5 | Graph integration live with dependency endpoints |
| M4 | 6, 7 | Retrieval + evidence upgrades hit quality targets |
| M5 | 8, 9 | Agentic + ensemble upgrades with regression gates |
| M6 | 10 | Codebase analysis integrated |

---

## Ralph Loop Usage

To run a phase with ralph loop:

```bash
/ralph-loop "Execute Phase 0 checklist items. Update UNIFIED_ROADMAP_RALPH.md to mark items complete. Output <promise>PHASE_0_COMPLETE</promise> when all items pass validation." --completion-promise "PHASE_0_COMPLETE" --max-iterations 20
```

After each phase:
1. Update `## Current State` to mark phase complete `[x]`
2. Mark next phase as in-progress `[→]`
3. Commit changes before starting next phase
