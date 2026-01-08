# Ethereum Protocol Intelligence System: Implementation Plan

**Version**: 1.1
**Date**: January 2026
**Philosophy**: Each phase delivers a working system. You can stop at any phase and have something useful. Later phases add intelligence, not just features.

---

## Current Status

| Phase | Name | Status | Notes |
|-------|------|--------|-------|
| 0 | Hello World RAG | ✅ Complete | 878 EIPs, 9,494 chunks, Voyage embeddings |
| 1 | Trustworthy Citations | ✅ Complete | Section-aware chunking, evidence ledger |
| 2 | Citation Validation | ✅ Complete | NLI validator, claim decomposition |
| 3 | Hybrid Search | ✅ Complete | BM25 + vectors, RRF fusion, Cohere reranking |
| 4 | EIP Graph | ✅ Complete | FalkorDB, dependency traversal, graph-augmented retrieval |
| 5 | Query Decomposition | ✅ Complete | Multi-hop reasoning, budget management, synthesis |
| 6 | Forums | ✅ Complete | ethresear.ch, Ethereum Magicians, Discourse API |
| 7-13 | Advanced | 🔲 Future | See details below |

### What's working now

```bash
# Ingest all 878 EIPs (~90 seconds with Voyage)
python scripts/ingest_eips.py

# Query with citations
python scripts/query_cli.py "What is EIP-1559?" --mode cited

# Query with NLI validation
python scripts/query_cli.py "What is EIP-1559?" --mode validated

# Query with synthesis (Phase 5 - handles complex queries)
python scripts/query_cli.py "Compare EIP-1559 and EIP-4844 gas models" --mode synthesis

# Start API server
uvicorn src.api.main:app --reload
```

### Known limitations (fixed in future phases)

- Semantic search misses exact term matches ("SELFDESTRUCT")
- Query "L2 costs" doesn't find EIP-4844 (talks about "blob transactions")
- No metadata filtering yet (status, category, author)

---

## Phase 0: Hello World RAG ✅ COMPLETE

**Corpus**: EIPs only (markdown files from ethereum/EIPs repo)

### What you build

- Clone EIPs repo, parse markdown files
- Fixed-size chunking (512 tokens, no sophistication)
- Single embedding model (Voyage-3 or OpenAI ada-002)
- PostgreSQL + pgvector for storage
- Top-k retrieval → stuff into prompt → LLM generates answer
- CLI or simple API endpoint

### Sample queries it handles

- "What is EIP-4844?"
- "What does EIP-1559 change?"
- "Who authored EIP-721?"

### What it CAN'T do well

- Complex multi-hop queries
- Verify if answer is correct
- Handle relationships between EIPs
- Deal with ambiguous terminology

### Intelligence level

🟢 Basic (glorified search + summarize)

### Validation

- Ask 10 simple factual questions about EIPs
- Answers should be roughly correct, maybe with some hallucination

### Deliverables

```
src/
  ingestion/
    eip_loader.py          # Clone repo, parse markdown
  chunking/
    fixed_chunker.py       # Simple 512-token chunks
  embeddings/
    voyage_embedder.py     # Embed chunks
  storage/
    pg_vector_store.py     # Store in PostgreSQL
  retrieval/
    simple_retriever.py    # Top-k vector search
  generation/
    simple_generator.py    # Stuff context + generate
  api/
    main.py                # FastAPI endpoint
```

### Working system

> "Ask a question about an EIP, get an answer"

---

## Phase 1: Trustworthy Citations ✅ COMPLETE

**Corpus**: EIPs only

### What you ADD

- Section-aware chunking (respect EIP structure: Abstract, Motivation, Specification, etc.)
- Keep code blocks atomic (never split)
- Store revision hashes (git commit SHA)
- Attach evidence spans to responses (which chunk supported which claim)
- Format citations in output: "According to EIP-4844 (Motivation section)..."

### Sample queries improved

- Same queries, but now with source attribution
- User can verify claims by checking cited sections

### New capability

```
Q: "What gas changes did EIP-1559 introduce?"

A: "EIP-1559 introduced a base fee that adjusts dynamically based on
block utilization [EIP-1559, Specification > Gas Fee Mechanics]. It also
added a priority fee (tip) paid directly to validators [EIP-1559, Abstract]."
```

### Intelligence level

🟢🟢 Basic + Verifiable

### Validation

- All responses include at least one citation
- Citations point to real content (spot-check manually)

### Deliverables

```
src/
  ingestion/
    eip_parser.py          # Extract sections, frontmatter
  chunking/
    section_chunker.py     # Section-aware, atomic code blocks
  storage/
    revision_store.py      # Track git commits
  evidence/
    evidence_span.py       # EvidenceSpan dataclass
    citation_formatter.py  # Format citations in output
  generation/
    cited_generator.py     # Generate with citations
```

### Working system

> "Ask about EIPs, get answers you can verify"

---

## Phase 2: Citation Validation ✅ COMPLETE

**Corpus**: EIPs only

### What you ADD

- NLI model (DeBERTa-v3-large-mnli) for automated citation checking
- Claim decomposition: break complex claims into atomic facts
- Evidence ledger: formal structure mapping claims → evidence spans
- **Reject or flag claims that aren't supported**
- Drift detection: alert if corpus changed since citation created

### How it works

1. LLM generates answer with citations
2. System decomposes answer into atomic claims
3. NLI checks: does cited evidence entail each claim?
4. Claims with low entailment are flagged or removed

### New capability

```
Q: "Does EIP-4844 reduce L2 costs?"

A: "Yes, EIP-4844 introduces blob transactions that reduce L2 data
costs by approximately 10-100x [EIP-4844, Abstract].

⚠️ Note: Specific cost reduction figures vary by L2 implementation
(claim partially supported)."
```

### Intelligence level

🟢🟢🟢 Self-correcting

### Validation

- Deliberately ask questions where LLM might hallucinate
- System should flag low-confidence claims
- Compare: with vs without validation (measure hallucination reduction)

### Deliverables

```
src/
  validation/
    nli_validator.py       # DeBERTa-based entailment check
    claim_decomposer.py    # Break claims into atomic facts
    evidence_ledger.py     # Formal claim → evidence mapping
    drift_detector.py      # Detect corpus changes
  generation/
    validated_generator.py # Generate + validate + flag
```

### Working system

> "Answers are checked against evidence before delivery"

---

## Phase 3: Hybrid Search ⏳ NEXT UP

**Corpus**: EIPs only

### What you ADD

- BM25 full-text search alongside vector search
- Reciprocal Rank Fusion (RRF) to combine results
- YAML frontmatter extraction (status, category, author, requires, etc.)
- Metadata filtering (find all "Final" EIPs, all EIPs by Vitalik, etc.)
- Reranking with Cohere rerank-v3

### Why this matters

- Vector search: good for semantic similarity ("gas optimization")
- BM25: good for exact terms ("SELFDESTRUCT", "EIP-4844")
- Combined: best of both worlds

### New capability

```
Q: "Which EIPs mention SELFDESTRUCT?"
A: [Returns EIP-6780, EIP-6049, etc. - BM25 finds exact term]

Q: "Which EIPs are about making Ethereum cheaper?"
A: [Returns EIP-1559, EIP-4844, EIP-4488, etc. - semantic search]

Q: "Show me all Draft EIPs about account abstraction"
A: [Metadata filter: status=Draft + semantic: account abstraction]
```

### Intelligence level

🟢🟢🟢🟢 Robust retrieval

### Validation

- Test exact term queries (BM25 should dominate)
- Test semantic queries (vectors should dominate)
- RRF should beat either alone

### Deliverables

```
src/
  retrieval/
    bm25_retriever.py      # Full-text search
    hybrid_retriever.py    # RRF fusion
    reranker.py            # Cohere rerank
  storage/
    metadata_store.py      # EIP frontmatter
    pg_schema.py           # Add tsvector column, GIN index
  filters/
    metadata_filter.py     # Filter by status, category, author
```

### Working system

> "Finds relevant EIPs whether you use exact terms or concepts"

---

## Phase 4: EIP Graph (Week 4)

**Corpus**: EIPs only

### What you ADD

- FalkorDB graph database
- Extract relationships from frontmatter: `requires`, `superseded-by`, `replaces`
- Build EIP dependency graph
- Graph-augmented retrieval: when asked about EIP-X, also fetch related EIPs
- Dependency chain queries

### New capability

```
Q: "What depends on EIP-1559?"

A: "23 EIPs directly or transitively depend on EIP-1559:
- Direct: EIP-3198 (BASEFEE opcode), EIP-3675 (Paris), ...
- Transitive: EIP-4844 (via EIP-3675), ..."

Q: "Explain EIP-4844 in context"
A: [System retrieves EIP-4844 + EIP-4488 + EIP-1559 + relevant specs]
```

### Intelligence level

🟢🟢🟢🟢🟢 Relationship-aware

### Validation

- Verify graph matches frontmatter data exactly
- Dependency queries return correct chains
- Graph context improves answer quality (A/B test)

### Deliverables

```
src/
  graph/
    falkordb_store.py      # FalkorDB connection
    eip_graph_builder.py   # Build graph from frontmatter
    dependency_traverser.py # Traverse requires/supersedes
  retrieval/
    graph_augmented.py     # Fetch related EIPs automatically
  api/
    graph_endpoints.py     # /eip/{id}/dependencies
```

### Working system

> "Understands how EIPs relate to each other"

---

## Phase 5: Query Decomposition (Weeks 5-6)

**Corpus**: EIPs only

### What you ADD

- Query classifier: simple vs complex
- Multi-hop query decomposition
- Staged retrieval: decompose → retrieve for each sub-question → synthesize
- Retrieval budget: max chunks per sub-question, total budget

### Why this matters

Complex queries like "Compare the gas models of EIP-1559 and EIP-4844" need:

1. Retrieve EIP-1559 gas model
2. Retrieve EIP-4844 gas model
3. Synthesize comparison

Single retrieval can't do this well.

### New capability

```
Q: "How did the community's position on state expiry evolve
    from EIP-4444 to EIP-4844?"

Decomposed:
1. What does EIP-4444 propose for state?
2. What does EIP-4844 propose for state?
3. What are the key differences?
4. How do these relate chronologically?

A: [Synthesized answer drawing from multiple retrievals]
```

### Intelligence level

🟢🟢🟢🟢🟢🟢 Multi-step reasoning

### Validation

- Complex queries should trigger decomposition
- Each sub-query should retrieve relevant chunks
- Final answer should synthesize correctly

### Deliverables

```
src/
  routing/
    query_classifier.py    # Simple vs complex
    query_decomposer.py    # Break into sub-questions
  retrieval/
    staged_retriever.py    # Retrieve per sub-question
    budget_manager.py      # Track retrieval budget
  generation/
    synthesis_generator.py # Combine sub-answers
```

### Working system

> "Handles complex questions by breaking them down"

---

## Phase 6: Expand Corpus - Forums (Weeks 7-8)

**Corpus**: EIPs + ethresear.ch + Ethereum Magicians

### What you ADD

- Discourse API ingestion (both forums)
- Store raw markdown (not cooked HTML)
- Thread structure: replies, reply-to relationships
- Forum reply graph edges: REPLIES_TO
- Cross-reference: link forum discussions to EIPs they mention
- Deduplication: detect crossposts between forums

### API details (verified Jan 2026)

Both ethresear.ch and ethereum-magicians.org run Discourse. API tested and working:

```python
# List recent topics
GET /latest.json?page=N

# Search for EIP mentions
GET /search.json?q=EIP-4844

# Get topic with posts
GET /t/{topic_id}.json
# Returns: title, posts_count, category_id, post_stream.posts[]

# Get individual post with RAW markdown
GET /posts/{post_id}.json
# Returns: username, raw, cooked, reply_to_post_number, created_at

# Key insight: use 'raw' not 'cooked'
# - raw: original markdown (stable) ✅
# - cooked: rendered HTML (mutates over time) ❌
```

Rate limits: ~1 req/sec works without auth. API key available for higher limits.

Example post structure:
```json
{
  "id": 38143,
  "username": "dcrapis",
  "raw": "*Thanks to Barnabé Monnot...* EIP-4844 introduces...",
  "reply_to_post_number": null,
  "created_at": "2023-03-16T17:14:04.265Z"
}
```

### New capability

```
Q: "What were the main objections to EIP-1559?"

A: "Key objections raised in eth research and Magicians discussions:
1. Miner revenue concerns [er-thread-7894, post 12 by abc.eth]
2. MEV implications [em-thread-4521, post 3]
3. Fee predictability skepticism [er-thread-7894, posts 45-52]
..."

Q: "What did Vitalik say about proto-danksharding?"
A: [Searches forum posts by vitalik.eth mentioning danksharding/EIP-4844]
```

### Intelligence level

🟢🟢🟢🟢🟢🟢🟢 Discussion-aware

### Validation

- Forum replies maintain thread structure
- Citations work for forum posts
- Cross-corpus retrieval works (EIP + forum threads together)

### Deliverables

```
src/
  ingestion/
    discourse_client.py    # Discourse API wrapper
    ethresearch_loader.py  # ethresear.ch ingestion
    magicians_loader.py    # Ethereum Magicians ingestion
  chunking/
    forum_chunker.py       # Respect reply boundaries
  graph/
    forum_graph.py         # REPLIES_TO edges
    cross_reference.py     # Link discussions to EIPs
  dedup/
    dedup_service.py       # Detect crossposts
```

### Working system

> "Knows both what EIPs say AND what people said about them"

---

## Phase 7: Concept Resolution (Week 9)

**Corpus**: EIPs + Forums

### What you ADD

- Alias table: map different names to same concept
  - "EIP-4844" = "proto-danksharding" = "blob transactions" = "danksharding-lite"
  - "The Merge" = "EIP-3675" = "Paris upgrade" = "PoS transition"
- Automatic alias extraction from corpus
- Query expansion using aliases
- Canonical concept IDs

### Why this matters

Users might search "blobs" and expect EIP-4844 results. Without alias resolution, semantic search might not connect them strongly enough.

### New capability

```
Q: "What is proto-danksharding?"
A: [System resolves to EIP-4844, retrieves canonical docs + forum discussions]

Q: "When was the merge?"
A: [System knows Merge = EIP-3675 = Paris = PoS transition,
    retrieves across all terms]
```

### Intelligence level

🟢🟢🟢🟢🟢🟢🟢🟢 Terminology-aware

### Validation

- Alias queries should return same results as canonical queries
- No duplicate results from alias expansion

### Deliverables

```
src/
  concepts/
    alias_table.py         # Store canonical → aliases
    alias_extractor.py     # Extract aliases from corpus
    query_expander.py      # Expand queries with aliases
    concept_resolver.py    # Resolve any name to canonical
```

### Working system

> "Understands that people call things by different names"

---

## Phase 8: Agentic Retrieval (Weeks 10-11)

**Corpus**: EIPs + Forums

### What you ADD

- ReAct-style interleaved thinking + retrieval
- Agent can decide: "I need more information about X" mid-generation
- Hard budget caps: max retrievals, max tokens, max cost
- Retrieval reflection: "Did I find what I needed?"
- Backtrack on dead ends

### Why this matters

Pre-planned retrieval can't handle truly complex questions where you don't know what you need until you start answering.

### New capability

```
Q: "Why didn't account abstraction proposals like EIP-86 succeed,
    and how does EIP-4337 address those issues?"

Agent reasoning:
1. [THINK] I need to understand EIP-86 first
2. [RETRIEVE] EIP-86 abstract and motivation
3. [THINK] It mentions security concerns. Let me find discussion about that.
4. [RETRIEVE] Forum discussions mentioning EIP-86 concerns
5. [THINK] Now I need EIP-4337 to compare
6. [RETRIEVE] EIP-4337 rationale section
7. [ANSWER] Here's the synthesis...
```

### Intelligence level

🟢🟢🟢🟢🟢🟢🟢🟢🟢 Adaptive reasoning

### Validation

- Budget limits enforced (agent stops when limit hit)
- Agent retrieves appropriate content (not random)
- Complex questions answered better than static retrieval

### Deliverables

```
src/
  agents/
    react_agent.py         # ReAct loop implementation
    retrieval_tool.py      # Tool for agent to call
    budget_enforcer.py     # Hard caps on retrieval
    reflection.py          # "Did I find what I needed?"
    backtrack.py           # Abandon dead ends
```

### Working system

> "Thinks about what it needs and goes looking for it"

---

## Phase 9: Structured Outputs (Week 12)

**Corpus**: EIPs + Forums

### What you ADD

- Timeline generation: chronological view of events
- Argument map: structure debates as pro/con with evidence
- Dependency view: visualizable graph of relationships
- Comparison tables: structured feature comparisons
- All outputs cite sources

### New capability

```
Q: "Create a timeline of account abstraction proposals"

A:
| Date       | Event                              | Source      |
|------------|-------------------------------------|-------------|
| 2016-09    | EIP-86 proposed                     | [EIP-86]    |
| 2017-03    | Vitalik discusses limitations       | [er-1234]   |
| 2020-10    | EIP-2938 proposed                   | [EIP-2938]  |
| 2021-09    | EIP-4337 proposed                   | [EIP-4337]  |
...

Q: "Map the arguments for and against state rent"
A: [Structured argument map with citations]
```

### Intelligence level

🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢 Structured synthesis

### Validation

- Timelines are chronologically accurate
- Argument maps include actual opposing views (not strawmen)
- All entries have valid citations

### Deliverables

```
src/
  structured/
    timeline_builder.py    # Build chronological views
    argument_mapper.py     # Pro/con structure
    comparison_table.py    # Feature comparisons
    dependency_view.py     # Graph visualization data
  generation/
    structured_generator.py # Generate structured outputs
```

### Working system

> "Can organize information into useful structures"

---

## Phase 10: Expand Corpus - Transcripts & Papers (Weeks 13-14)

**Corpus**: EIPs + Forums + ACD transcripts + arXiv papers

### What you ADD

- All Core Devs call transcripts (ethereum/pm repo)
- arXiv paper ingestion (cs.CR, cs.DC with Ethereum tags)
- PDF extraction with quality scoring
- Speaker attribution in transcripts
- Paper citation graph

### New capability

```
Q: "What did core devs say about EIP-4844 timeline in the last 3 months?"
A: [Searches ACD transcripts, attributes quotes to speakers]

Q: "What academic research supports EIP-1559's fee mechanism?"
A: [Retrieves Roughgarden paper, other academic analysis]
```

### Intelligence level

🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢 Full corpus

### Deliverables

```
src/
  ingestion/
    acd_transcript_loader.py  # ethereum/pm transcripts
    arxiv_fetcher.py          # arXiv API
    pdf_extractor.py          # PyMuPDF extraction
    quality_scorer.py         # PDF quality scoring
  chunking/
    transcript_chunker.py     # Speaker-aware chunking
    paper_chunker.py          # Section-aware for papers
  graph/
    citation_graph.py         # Paper citations
```

### Working system

> "Knows the formal proposals, discussions, meeting decisions, AND academic backing"

---

## Phase 11: Inferred Relationships (Weeks 15-16)

**Corpus**: Full

### What you ADD

- LLM-inferred graph edges (CONFLICTS_WITH, ADDRESSES_CONCERN, INSPIRED_BY)
- Confidence scores on inferred edges
- Selective traversal: only follow high-confidence edges
- Incremental re-inference on corpus updates

### Why this matters

Explicit frontmatter relationships are incomplete. "EIP-4844 was inspired by EIP-4488" isn't in frontmatter but is true.

### New capability

```
Q: "What EIPs conflict with each other?"

A: "Several EIP pairs have conflicting approaches:
- EIP-4444 vs EIP-4444b [inferred, confidence: 0.89]
- EIP-86 vs EIP-4337 design philosophy [inferred, confidence: 0.72]
..."
```

### Intelligence level

🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢 Relationship inference

### Validation

- High-confidence inferences should be accurate (human review sample)
- Low-confidence edges should not appear in answers without caveat

### Deliverables

```
src/
  graph/
    relationship_inferrer.py   # LLM-based inference
    confidence_scorer.py       # Score inferred edges
    selective_traverser.py     # Only follow high-confidence
    incremental_updater.py     # Re-infer on corpus change
```

### Working system

> "Discovers relationships that weren't explicitly stated"

---

## Phase 12: Multi-Model Ensemble (Weeks 17-18)

**Corpus**: Full

### What you ADD

- Cost-aware model routing (Sonnet for simple, Opus for complex)
- Retrieval confidence scoring
- Ensemble synthesis: run multiple models, combine best parts
- Conditional ensemble: only for high-stakes queries (roadmap generation)

### New capability

```
Q: "What is EIP-721?"
[Simple query → Sonnet → fast, cheap answer]

Q: "Generate an alternative roadmap for Ethereum scaling that
    prioritizes L1 execution over L2-centric approach"
[Complex synthesis → Opus + high retrieval budget → ensemble for quality]
```

### Intelligence level

🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢 Adaptive intelligence

### Deliverables

```
src/
  routing/
    cost_router.py            # Route by complexity/cost
    confidence_scorer.py      # Score retrieval confidence
  ensemble/
    multi_model_runner.py     # Run multiple models
    synthesis_combiner.py     # Combine outputs
    conditional_trigger.py    # When to use ensemble
```

### Working system

> "Uses the right level of intelligence for each question"

---

## Phase 13: Client Codebase Analysis (Weeks 19-22)

**Corpus**: Full + Geth + Prysm codebases

### What you ADD

- Tree-sitter parsing for Go
- Function/struct-level chunking
- Code-aware embeddings
- Cross-reference: link EIP specs to implementations
- Implementation query support

### New capability

```
Q: "How does Geth implement EIP-1559's base fee calculation?"
A: [Retrieves actual Go code from go-ethereum/core/state_processor.go,
    explains implementation, links to spec]

Q: "Generate a speculative implementation plan for EIP-7702 in Prysm"
A: [Analyzes similar patterns in Prysm, suggests implementation approach]
```

### Intelligence level

🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢 Code-aware

### Deliverables

```
src/
  ingestion/
    git_code_loader.py        # Clone and track repos
  parsing/
    treesitter_parser.py      # Tree-sitter Go/Rust parsing
    code_unit_extractor.py    # Functions, structs, interfaces
  chunking/
    code_chunker.py           # Function-level chunks
  embeddings/
    code_embedder.py          # Code-aware embeddings
  graph/
    spec_impl_linker.py       # Link EIPs to implementations
```

### Working system

> "Understands not just what EIPs say, but how they're implemented"

---

## Summary: Capability Progression

| Phase | Corpus | Key Capability | Query Example |
|-------|--------|----------------|---------------|
| 0 | EIPs | Basic Q&A | "What is EIP-4844?" |
| 1 | EIPs | Source citations | "What does it say in the spec?" |
| 2 | EIPs | Verified citations | "Is this actually true?" |
| 3 | EIPs | Hybrid search | "Find EIPs about gas" |
| 4 | EIPs | Relationships | "What depends on this?" |
| 5 | EIPs | Multi-hop reasoning | "Compare these two" |
| 6 | +Forums | Community voice | "What did people think?" |
| 7 | +Forums | Terminology | "What's proto-danksharding?" |
| 8 | +Forums | Adaptive retrieval | Complex research questions |
| 9 | +Forums | Structured output | Timelines, argument maps |
| 10 | +ACD/Papers | Full context | Meeting decisions, academic backing |
| 11 | Full | Inferred relations | Hidden connections |
| 12 | Full | Smart routing | Cost-appropriate intelligence |
| 13 | +Code | Implementation | "How is this built?" |

---

## Quick Start: Phase 0 in 3 Hours

```bash
# 1. Clone EIPs repo (5 min)
git clone https://github.com/ethereum/EIPs.git data/eips

# 2. Set up PostgreSQL + pgvector (10 min)
docker run -d --name pgvector \
  -e POSTGRES_PASSWORD=dev \
  -p 5432:5432 \
  pgvector/pgvector:pg16

# 3. Install dependencies (5 min)
pip install asyncpg pgvector voyageai fastapi uvicorn

# 4. Run ingestion script (30 min)
python scripts/ingest_eips.py

# 5. Start API (5 min)
uvicorn src.api.main:app --reload

# 6. Test it (5 min)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is EIP-4844?"}'
```

---

## Decision Points

At each phase boundary, evaluate:

1. **Is this enough?** - Current system might be sufficient for your use case
2. **What's the biggest gap?** - Prioritize phases that address your main pain points
3. **What's the cost/benefit?** - Later phases add complexity; is the intelligence gain worth it?

The plan is designed so you can skip phases or reorder them based on your priorities. The only hard dependencies are:

- Phase 2 requires Phase 1 (can't validate citations without citations)
- Phase 4 requires Phase 1 (graph needs parsed frontmatter)
- Phase 8 requires Phase 5 (agentic builds on decomposition)
- Phase 11 requires Phase 4 (inference extends graph)
