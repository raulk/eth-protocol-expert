# Data Source Extraction Checklist

Concrete, source-by-source extraction targets and where to wire them in. Use this as the implementation checklist for enriching data + semantics.

## Global schema additions (shared)
- `documents.metadata`:
  - `entities`: normalized entities (EIP numbers, forks, clients, roles)
  - `temporal`: dates, fork names, version tags
  - `trust_tier`: eip/spec/api/forum
- `chunks.metadata` (add column if needed):
  - `section_kind`: spec/params/notes/faq/code/table
  - `normative_level`: MUST/SHOULD/MAY
  - `source_section_path`: normalized section header path

## EIPs (ethereum/EIPs)
- Extract parameters table into structured metadata:
  - gas costs, limits, constants, fork name, activation block
- Parse normative keywords per section:
  - capture MUST/SHOULD/MAY occurrences and tag `normative_level`
- Graph edges:
  - `requires`, `supersedes`, `depends_on` into relationship graph
- Wire:
  - `src/ingestion/eip_parser.py`
  - `src/graph/eip_graph_builder.py`

## ERCs (ethereum/ERCs)
- Extract interface signatures:
  - function names, events, error types, optional extensions
- Tag compliance sections (requirements vs optional)
- Wire:
  - `src/ingestion/erc_loader.py`
  - `src/chunking/section_chunker.py`

## RIPs (ethereum/RIPs)
- Extract rollup parameters:
  - sequencing, proof system, fraud/validity properties
- Map RIPs to referenced EIPs and clients
- Wire:
  - `src/ingestion/rip_loader.py`
  - `src/graph/relationship_inferrer.py`

## Consensus specs (ethereum/consensus-specs)
- Extract constants and state transition functions:
  - store constants in `documents.metadata.constants`
- Version diffs by fork (Altair/Bellatrix/Capella/Deneb)
- Wire:
  - `src/ingestion/consensus_spec_loader.py`
  - `src/structured/timeline_builder.py`

## Execution specs (ethereum/execution-specs)
- Extract opcode tables + gas schedule changes
- Normalize preconditions/transitions/postconditions into metadata
- Wire:
  - `src/ingestion/execution_spec_loader.py`
  - `src/structured/comparison_table.py`

## ACD transcripts
- Extract speaker, decision, and action items
- Build decision timeline and link to forks
- Wire:
  - `src/ingestion/acd_transcript_loader.py`
  - `src/structured/timeline_builder.py`

## ethresear.ch (forum)
- Identify proposal stage and map to EIP numbers
- Extract “open questions” vs “consensus points”
- Wire:
  - `src/ingestion/ethresearch_loader.py`
  - `src/structured/argument_mapper.py`

## Ethereum Magicians (forum)
- Same as ethresear.ch + stricter injection stripping
- Link threads to EIP PRs/issues where referenced
- Wire:
  - `src/ingestion/magicians_loader.py`
  - `src/ingestion/safe_context_builder.py` (if present)

## arXiv papers
- Extract abstract + key results + method tags
- Build citation graph edges to EIPs/specs
- Wire:
  - `src/ingestion/arxiv_fetcher.py`
  - `src/graph/citation_graph.py`

## research repo (ethereum/research)
- Extract code snippets with language metadata
- Link research notes to EIPs/specs
- Wire:
  - `src/ingestion/research_loader.py`
  - `src/parsing/code_unit_extractor.py`

## devp2p specs
- Extract message schemas + handshake sequences
- Build protocol capability entities
- Wire:
  - `src/ingestion/devp2p_loader.py`
  - `src/structured/dependency_view.py`

## portal-network-specs
- Extract content routing + content key formats
- Link to base devp2p protocol specs
- Wire:
  - `src/ingestion/portal_spec_loader.py`
  - `src/graph/relationship_inferrer.py`

## execution-apis (JSON-RPC)
- Parse OpenAPI into method schemas (params, types, examples)
- Track version diffs for endpoints
- Wire:
  - `src/ingestion/execution_apis_loader.py`
  - `src/structured/comparison_table.py`

## beacon-apis
- Same as execution-apis with endpoint tagging by role
- Wire:
  - `src/ingestion/beacon_apis_loader.py`
  - `src/structured/comparison_table.py`

## builder-specs
- Extract roles + PBS flow steps as sequences
- Link to MEV-related EIPs
- Wire:
  - `src/ingestion/builder_specs_loader.py`
  - `src/structured/timeline_builder.py`

## Client codebases (if indexed)
- Extract function signatures + call graph
- Map functions to spec sections for spec-impl alignment
- Wire:
  - `src/parsing/treesitter_parser.py`
  - `src/graph/spec_impl_linker.py`

## Cross-corpus semantics (all sources)
- Normalize entities to a canonical ontology:
  - EIP numbers, fork names, clients, roles, protocol components
- Temporal anchors:
  - fork activation dates, version tags, decision timestamps
- Trust-tier weighting in retrieval:
  - multiplicative/additive/tie-breaker (decide in Phase 6)
- Wire:
  - `src/concepts/alias_table.py`
  - `src/concepts/concept_resolver.py`
  - `src/retrieval/hybrid_retriever.py`
