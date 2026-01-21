# SOTA Upgrade Plan: Accuracy, Reasoning, Precision

This plan upgrades the system to state‑of‑the‑art (SOTA) practices across retrieval, grounding, verification, and reasoning. It is structured in phases so each step yields measurable gains without large regressions.

## Goals (Measurable)
- **Answer precision**: >= 0.80 factual precision on a domain QA set.
- **Citation accuracy**: >= 0.85 claim‑to‑evidence support rate (NLI + human spot‑checks).
- **Retrieval recall**: >= 0.75 recall@10 on labeled evidence set.
- **Reasoning reliability**: >= 0.70 multi‑hop success rate on decomposition tasks.
- **Latency/cost**: keep median query <= 6s (simple) and <= 12s (validated) with budget caps.

## Baseline Observations (Current Gaps)
- Sentence‑level citation extraction is regex‑based and offset‑unsafe.
- Factual claims without citations bypass NLI (unsafe).
- NLI validates against full chunks (noisy evidence).
- Concept expansion exists but is not integrated into retrieval.
- Agent retrieval uses fixed limit/mode regardless of query complexity.
- No domain evaluation harness driving threshold calibration.

---

## Phase 0 — Evaluation Harness (Foundation)
**Objective**: Establish reliable metrics before changing behavior.

**Deliverables**
- `tests/eval/` with:
  - Labeled QA set (EIP facts, protocol parameters, upgrades).
  - Multi‑hop set (dependencies, comparisons, timeline questions).
  - Evidence‑labeled set (gold chunks/spans).
- Concrete labeling targets:
  - 50–100 questions with manually verified evidence chunks.
  - 20–30 multi‑hop questions (dependencies, comparisons, timelines).
  - Bootstrap from existing `tests/` where possible.
  - LLM‑assisted labeling with human verification for scale.
- Evaluation scripts for:
  - retrieval recall@k, MRR
  - citation accuracy (claim support)
  - NLI agreement vs gold
  - end‑to‑end correctness
- Versioned eval datasets:
  - `v1.0` — initial 50‑question seed
  - `v1.1` — expanded with multi‑hop
  - `v2.0` — post‑Phase‑2 with span‑level gold labels

**Metrics**
- Baseline metrics recorded and stored in `data/eval/metrics.json`.

**Risks**
- Label scarcity; mitigate with small, high‑quality seed set + active expansion.

---

## Phase 1 — Retrieval SOTA (Recall + Precision)
**Objective**: Improve recall and ranking quality using modern retrieval methods.

**Deliverables**
- **Multi‑query retrieval**:
  - Alias expansion (existing concept resolver).
  - LLM query rewriting (paraphrase + keyword expansions).
  - Fuse results with RRF.
- **SOTA reranking**:
  - Add a cross‑encoder reranker (e.g., BGE‑reranker or Cohere Rerank v3).
  - Rerank top‑N from hybrid retrieval.
- **Diversity control**:
  - MMR or section‑coverage selection to reduce redundancy.

**Implementation Areas**
- `src/retrieval/hybrid_retriever.py`
- `src/retrieval/simple_retriever.py`
- `src/concepts/*` integration

**Metrics**
- +10–20% recall@10; improved MRR.

**Risks**
- Latency from reranking; mitigate with top‑20 cap, caching, and async rerank.
- Prefer single‑vendor rerank when possible (e.g., `voyage-rerank` if using Voyage embeddings).

---

## Source Enhancements (Existing Corpus)
**Objective**: Enrich current sources with structured metadata and semantics to improve precision.

**EIPs**
- Extract structured params (gas costs, constants, activation/fork, limits) into metadata.
- Tag normative strength (MUST/SHOULD/MAY) per section.
- Build explicit edges: `requires`, `supersedes`, `depends_on`.
- Add “change rationale” summaries per section.

**Consensus specs**
- Extract constants and state transition functions as structured key‑values.
- Build per‑fork diffs (Altair/Bellatrix/Capella/Deneb).
- Tag sections by category (validator, fork choice, rewards).

**Execution specs**
- Extract opcode tables + gas schedule changes into a normalized table.
- Encode preconditions → transitions → postconditions per spec section.
- Tag sections by EVM component (gas, storage, transactions).

**ACD transcripts**
- Extract speakers, decisions, action items, and final status per agenda item.
- Link decisions to forks and EIPs; build timeline index.
- Distinguish consensus vs unresolved discussions.

**arXiv papers**
- Extract abstract + key results + method tags.
- Build citation graph edges to EIPs/specs referenced.
- Extract experiment configs/datasets for reproducibility queries.

**ethresear.ch**
- Identify proposal stages and map to EIP numbers.
- Extract open questions vs consensus points.
- Add per‑thread summaries + key quotes.

**Implementation Areas**
- `src/ingestion/eip_parser.py`
- `src/ingestion/consensus_spec_loader.py`
- `src/ingestion/execution_spec_loader.py`
- `src/ingestion/acd_transcript_loader.py`
- `src/ingestion/arxiv_fetcher.py`
- `src/ingestion/ethresearch_loader.py`
- `src/graph/eip_graph_builder.py`
- `src/graph/citation_graph.py`
- `src/structured/timeline_builder.py`
- `src/structured/argument_mapper.py`

---

## Schema Changes (Metadata)
**Objective**: Add structured fields to store extracted semantics from existing sources.

**Documents**
- `documents.metadata` JSONB:
  - `entities`: normalized entities (EIP numbers, forks, clients, roles)
  - `temporal`: activation blocks, dates, version tags
  - `params`: constants, gas values, limits
  - `normative`: MUST/SHOULD/MAY counts
  - `stage`: proposal stage (idea/draft/final)

**Chunks**
- Add `chunks.metadata` JSONB column:
  - `section_kind`: spec/params/notes/faq/code/table
  - `normative_level`: MUST/SHOULD/MAY
  - `source_section_path`: normalized header path
  - `span_type`: sentence/paragraph/table row

**Graph**
- Add relationship labels:
  - `REQUIRES`, `SUPERSEDES`, `DEPENDS_ON`, `CITES`

**Implementation Areas**
- `src/storage/pg_vector_store.py` (schema + migrations)
- `src/graph/*` (edge labels + loaders)
- `src/ingestion/*` (populate metadata)

---

## Phase 2 — Evidence Precision (Span‑Level Grounding)
**Objective**: Tighten grounding by validating against minimal evidence spans.

**Deliverables**
- Span extraction:
  - Sentence/paragraph segmentation with accurate offsets.
  - Best‑span selection per claim using cross‑encoder scoring.
  - Edge‑case test suite (markdown headers, code blocks, tables, nested lists).
  - Timeboxed fallback to sentence‑level grounding if offsets are unreliable.
- Evidence ledger upgrades:
  - Evidence spans store start/end offsets and source sentence ids.
  - Deduplicate citations by span quality.
- NLI on minimal spans:
  - Validate each atomic claim against top‑k spans; aggregate support.

**Implementation Areas**
- `src/generation/validated_generator.py`
- `src/validation/nli_validator.py`
- `src/evidence/evidence_span.py`

**Metrics**
- +15–25% citation accuracy.

**Risks**
- Span extraction bugs; mitigate with unit tests and offset checks.

---

## Phase 3 — Claim Decomposition + Verification Loop
**Objective**: Ensure claims are atomic, cited, and verified.

**Deliverables**
- Atomic claim splitter:
  - Replace regex splitting with dependency‑aware clause extraction.
  - Reserve LLM extraction for ambiguous or complex sentences.
- Citation enforcement:
  - Reject/flag any factual sentence without citations.
  - Require per‑clause citations in multi‑fact sentences.
- Verification pass:
  - “Critic” pass that checks for unsupported facts; remove or tag.

**Implementation Areas**
- `src/validation/claim_decomposer.py`
- `src/generation/validated_generator.py`

**Metrics**
- Increase factual precision; reduce unsupported claim rate < 10%.

---

## Phase 4 — Reasoning & Agentic Retrieval
**Objective**: Improve multi‑hop reasoning with structured plans and adaptive retrieval.

**Deliverables**
- Planner‑Retriever‑Synthesizer loop:
  - Decompose query → retrieve per sub‑question → synthesize.
- Adaptive retrieval budgeting:
  - Choose mode (hybrid/graph) and k based on query classifier + retrieval quality.
- Reflection augmented by evidence coverage:
  - Use evidence ledger stats to trigger more retrieval.
- Audit and upgrade existing agent code before adding new loops.
- Integrate graph traversal explicitly (dependency edges, inferred relations) for multi‑hop.

**Implementation Areas**
- `src/agents/react_agent.py`
- `src/agents/retrieval_tool.py`
- `src/routing/query_classifier.py`
- `src/graph/eip_graph_builder.py`
- `src/graph/dependency_traverser.py`

**Metrics**
- +15% multi‑hop success; fewer dead‑ends.

---

## Phase 5 — Calibration & Cost‑Quality Routing
**Objective**: Calibrate confidence and optimize cost without quality regression.

**Deliverables**
- Confidence calibration (isotonic regression) using eval logs.
- Dynamic model routing based on query difficulty + retrieval entropy.
- Circuit breakers for hallucination risk (low evidence, low entailment).

**Dependencies**
- Requires eval logs from Phase 0–4 runs to fit calibration curves.

**Implementation Areas**
- `src/ensemble/cost_router.py`
- `src/ensemble/confidence_scorer.py`

**Metrics**
- Stable quality at reduced cost; fewer incorrect “confident” answers.

---

## Phase 6 — Safety & Data Hygiene
**Objective**: Reduce prompt injection risk and corpus drift.

**Deliverables**
- Strict content stripping for forum data.
- Trust‑tier source weighting (EIPs > specs > forum).
- Document update detector to avoid unnecessary re‑embedding.

**Trust‑tier weighting**
- Define one of:
  - Multiplicative factor on similarity scores
  - Additive boost to ranking
  - Tie‑breaker when scores are close

**Implementation Areas**
- `src/ingestion/*`
- `src/retrieval/*`

---

## Tooling & Dependencies (Suggested)
- Reranker: `bge-reranker-large`, `cohere-rerank-v3`, or `voyage-rerank`.
- Sentence segmentation: `spacy` (en_core_web_sm) or `nltk` Punkt.
- Calibration: `sklearn` isotonic regression.
- Eval harness: simple JSONL datasets + pytest or custom scripts.

---

## Fallback Strategy (Default Behavior)
- If evidence coverage is below threshold, default to abstain: “Insufficient evidence in sources.”
- If partial evidence exists, answer with explicit caveats and confidence score.
- Allow override only when user opts into speculative mode.

---

## Experimentation & Regression Control
- A/B testing: shadow mode or feature flags per phase (10% canary).
- Regression gates:
  - Phase 1: retrieval recall must not drop below baseline.
  - Phase 2: citation accuracy must improve or stay flat.
  - Phase 3: median latency must stay within budget.

---

## Rollout Strategy
1. Implement Phase 0, establish baseline.
2. Implement Phase 1 and re‑evaluate.
3. Implement Phase 2 and re‑evaluate.
4. Implement Phase 3 and re‑evaluate.
5. Implement Phase 4 and re‑evaluate.
6. Implement Phase 5/6 as optimization and safety hardening.

---

## Success Criteria
- Retrieval recall@10 >= 0.75
- Citation accuracy >= 0.85
- Unsupported claim rate <= 10%
- Multi‑hop success >= 0.70
- Median latency within budget targets
