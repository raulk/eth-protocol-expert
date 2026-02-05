# Chunking and embedding upgrade plan

**Date**: January 2026
**Status**: Proposed
**Author**: Claude

## Executive summary

This plan upgrades the Ethereum Protocol Intelligence System's chunking and embedding pipeline to January 2026 SOTA. The upgrades are expected to improve retrieval accuracy by 40-67% based on published benchmarks.

**Key changes:**
1. Switch to voyage-code-3 for code embeddings
2. Add voyage-rerank-2.5 reranking stage
3. Implement Anthropic's Contextual Retrieval across all document types
4. Add hierarchical chunking with parent-child relationships
5. Generate synthetic queries for code chunks
6. Hybrid BM25 + dense retrieval
7. Query-time graph expansion for code
8. Late chunking for dense documents
9. HyDE (Hypothetical Document Embeddings) for complex queries
10. ColBERT multi-vector embeddings (optional, high storage)

---

## Current state

### Embedding models

| Document Type | Current Model | Dimensions |
|---------------|---------------|------------|
| EIPs | voyage-4-large | 1024 |
| Forum posts | voyage-4-large | 1024 |
| Papers | voyage-4-large | 1024 |
| Transcripts | voyage-4-large | 1024 |
| **Code** | voyage-4-large | 1024 |

**Problem**: Code uses general-purpose model instead of code-optimized model.

### Chunking strategies

| Chunker | Max Tokens | Overlap | Key Features |
|---------|------------|---------|--------------|
| FixedChunker | 512 | 64 | Basic token-based |
| SectionChunker | 512 | 64 | EIP sections, atomic code blocks |
| CodeChunker | 512 | 0 | Function-level, header preservation |
| ForumChunker | 512 | 64 | Post boundaries, reply context |
| PaperChunker | 512 | 64 | Section-aware, equation preservation |
| TranscriptChunker | 512 | 32 | Speaker boundaries, timestamps |

**Problems**:
- No contextual prepending (chunks lack document context)
- No hierarchical structure (flat chunks only)
- No reranking stage
- Code chunker uses brace counting for splits (not AST-aware)

### Retrieval pipeline

```
Query → Embed → Vector Search (top-k) → Return
```

**Missing**: Hybrid BM25, reranking, query-time graph expansion.

---

## Target state (January 2026 SOTA)

### Embedding models

| Document Type | Target Model | Dimensions | Rationale |
|---------------|--------------|------------|-----------|
| EIPs | voyage-4-large | 1024 | General purpose, strong |
| Forum posts | voyage-4-large | 1024 | General purpose |
| Papers | voyage-4-large | 1024 | General purpose |
| Transcripts | voyage-4-large | 1024 | General purpose |
| **Code** | **voyage-code-3** | 1024 | 13.8% better than general on code |

### Retrieval pipeline

```
Query → Embed → Hybrid Search (BM25 + Vector) → Rerank (voyage-rerank-2.5) → Return
                      ↓
              Graph Expansion (for code queries)
```

### Chunking enhancements

1. **Contextual Retrieval**: Prepend LLM-generated context to every chunk
2. **Hierarchical chunks**: Parent (summary) + children (details)
3. **Synthetic queries**: For code, generate natural language questions

---

## Phase 1: Embedding model upgrade

**Effort**: Low
**Impact**: High for code retrieval
**Files**: `src/embeddings/code_embedder.py`, `scripts/ingest_client.py`

### 1.1 Update CodeEmbedder to voyage-code-3

```python
# src/embeddings/code_embedder.py
class CodeEmbedder:
    def __init__(
        self,
        model: str = "voyage-code-3",  # Changed from voyage-code-2
        output_dimension: int = 1024,   # Changed from 1536
    ):
```

**Status**: ✅ Already completed

### 1.2 Update ingest_client.py to use CodeEmbedder

Replace `VoyageEmbedder` with `CodeEmbedder` + `CodeChunker` pipeline:

```python
from src.chunking.code_chunker import CodeChunker
from src.embeddings.code_embedder import CodeEmbedder

chunker = CodeChunker(max_tokens=512)
embedder = CodeEmbedder(model="voyage-code-3")

code_chunks = chunker.chunk(code_units, language=repo.language)
embeddings = embedder.embed_chunks(code_chunks)
```

**Status**: Pending

### 1.3 Re-embed existing code chunks

After model change, re-run ingestion for all clients:

```bash
uv run python scripts/ingest_client.py --repo go-ethereum
uv run python scripts/ingest_client.py --repo lighthouse
uv run python scripts/ingest_client.py --repo prysm
uv run python scripts/ingest_client.py --repo reth
```

---

## Phase 2: Add reranking

**Effort**: Low
**Impact**: High (improves top-k precision significantly)
**Files**: `src/retrieval/reranker.py` (new), `src/retrieval/simple_retriever.py`

### 2.1 Create Reranker class

```python
# src/retrieval/reranker.py
import voyageai

class VoyageReranker:
    """Rerank search results using voyage-rerank-2.5."""

    def __init__(
        self,
        model: str = "rerank-2.5",
        top_k: int = 10,
    ):
        self.client = voyageai.Client()
        self.model = model
        self.top_k = top_k

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        instruction: str | None = None,
    ) -> list[SearchResult]:
        """Rerank results using cross-encoder."""
        documents = [r.chunk.content for r in results]

        response = self.client.rerank(
            query=query,
            documents=documents,
            model=self.model,
            top_k=self.top_k,
            instruction=instruction,  # voyage-rerank-2.5 supports instructions
        )

        reranked = []
        for item in response.results:
            original = results[item.index]
            reranked.append(SearchResult(
                chunk=original.chunk,
                similarity=item.relevance_score,
            ))

        return reranked
```

### 2.2 Integrate into retrieval pipeline

```python
# src/retrieval/simple_retriever.py
class SimpleRetriever:
    def __init__(
        self,
        store: PgVectorStore,
        embedder: VoyageEmbedder,
        reranker: VoyageReranker | None = None,
    ):
        self.reranker = reranker

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        rerank: bool = True,
    ) -> RetrievalResult:
        # Retrieve more candidates for reranking
        candidates_k = top_k * 3 if rerank and self.reranker else top_k

        results = await self.store.search(query_embedding, limit=candidates_k)

        if rerank and self.reranker:
            results = self.reranker.rerank(query, results)[:top_k]

        return RetrievalResult(results=results)
```

### 2.3 Add code-specific reranking instructions

```python
CODE_RERANK_INSTRUCTION = """
Rank these code snippets by relevance to the query.
Prioritize:
1. Direct implementation of the queried functionality
2. Type definitions and structs related to the query
3. Helper functions called by the main implementation
"""
```

---

## Phase 3: Contextual retrieval

**Effort**: Medium
**Impact**: High (67% retrieval failure reduction per Anthropic)
**Files**: All chunkers, new `src/chunking/context_generator.py`

### 3.1 Create context generator

```python
# src/chunking/context_generator.py
from anthropic import Anthropic

class ChunkContextGenerator:
    """Generate contextual prefixes for chunks using Claude."""

    PROMPT_TEMPLATE = """
<document>
{document_content}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk_content}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else.
"""

    def __init__(self, model: str = "claude-3-5-haiku-20241022"):
        self.client = Anthropic()
        self.model = model

    def generate_context(
        self,
        chunk_content: str,
        document_content: str,
    ) -> str:
        """Generate contextual prefix for a chunk."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=150,
            messages=[{
                "role": "user",
                "content": self.PROMPT_TEMPLATE.format(
                    document_content=document_content[:10000],  # Truncate if needed
                    chunk_content=chunk_content,
                ),
            }],
        )
        return response.content[0].text

    def contextualize_chunk(
        self,
        chunk: Chunk,
        document_content: str,
    ) -> Chunk:
        """Add contextual prefix to chunk content."""
        context = self.generate_context(chunk.content, document_content)

        return Chunk(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            content=f"{context}\n\n{chunk.content}",  # Prepend context
            token_count=chunk.token_count + len(context.split()),
            chunk_index=chunk.chunk_index,
            section_path=chunk.section_path,
        )
```

### 3.2 Document-specific context prompts

```python
# EIP context prompt
EIP_CONTEXT_PROMPT = """
This chunk is from EIP-{eip_number}: {title}.
EIP Status: {status}, Type: {type}, Category: {category}
{requires_text}

Situate this chunk within the EIP specification:
"""

# Code context prompt
CODE_CONTEXT_PROMPT = """
This code is from the {repository} Ethereum client ({language}).
File: {file_path}
Function: {function_name}
{dependencies_text}

Describe what this code does in the context of Ethereum protocol implementation:
"""

# Forum context prompt
FORUM_CONTEXT_PROMPT = """
This is from a discussion on {source} titled "{topic_title}".
Post #{post_number} by @{username}, replying to post #{reply_to}.
Topic tags: {tags}

Situate this post within the discussion:
"""

# Transcript context prompt
TRANSCRIPT_CONTEXT_PROMPT = """
This is from AllCoreDevs Call #{call_number} ({date}).
Speaker: {speaker}
Agenda items discussed: {agenda_items}

Describe the context of this discussion:
"""
```

### 3.3 Cost optimization with prompt caching

Use Anthropic's prompt caching to reduce costs:

```python
# Cache the document content across chunks from same document
self.client.messages.create(
    model=self.model,
    max_tokens=150,
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": document_content,
                "cache_control": {"type": "ephemeral"},  # Cache this
            },
            {
                "type": "text",
                "text": f"Chunk to situate:\n{chunk_content}",
            },
        ],
    }],
)
```

**Estimated cost**: ~$1 per million document tokens (with caching)

---

## Phase 4: Hierarchical chunking

**Effort**: Medium
**Impact**: Medium-High
**Files**: All chunkers, `src/storage/pg_vector_store.py`

### 4.1 Schema update for parent-child relationships

```sql
ALTER TABLE chunks ADD COLUMN parent_chunk_id TEXT REFERENCES chunks(chunk_id);
ALTER TABLE chunks ADD COLUMN chunk_level INTEGER DEFAULT 0;  -- 0=leaf, 1=parent, 2=grandparent
CREATE INDEX idx_chunks_parent ON chunks(parent_chunk_id);
```

### 4.2 Hierarchical chunk structure

```
Level 2: Document Summary
    └── Level 1: Section Summary
            └── Level 0: Detail Chunk (current)
```

### 4.3 Example: EIP hierarchical chunks

```python
def chunk_eip_hierarchical(self, parsed_eip: ParsedEIP) -> list[Chunk]:
    chunks = []

    # Level 2: Document summary
    doc_summary = self._generate_eip_summary(parsed_eip)
    doc_chunk = Chunk(
        chunk_id=f"eip-{parsed_eip.eip_number}-summary",
        document_id=f"eip-{parsed_eip.eip_number}",
        content=doc_summary,
        chunk_level=2,
        parent_chunk_id=None,
    )
    chunks.append(doc_chunk)

    # Level 1: Section summaries
    for section in parsed_eip.sections:
        section_summary = self._generate_section_summary(section)
        section_chunk = Chunk(
            chunk_id=f"eip-{parsed_eip.eip_number}-{section.name}-summary",
            content=section_summary,
            chunk_level=1,
            parent_chunk_id=doc_chunk.chunk_id,
        )
        chunks.append(section_chunk)

        # Level 0: Detail chunks (existing logic)
        detail_chunks = self._chunk_section(section)
        for detail in detail_chunks:
            detail.parent_chunk_id = section_chunk.chunk_id
            detail.chunk_level = 0
            chunks.append(detail)

    return chunks
```

### 4.4 Retrieval with parent context

```python
async def retrieve_with_context(
    self,
    query: str,
    top_k: int = 10,
    include_parent: bool = True,
) -> RetrievalResult:
    """Retrieve chunks and optionally include parent context."""
    results = await self.retrieve(query, top_k=top_k)

    if include_parent:
        for result in results:
            if result.chunk.parent_chunk_id:
                parent = await self.store.get_chunk(result.chunk.parent_chunk_id)
                result.parent_context = parent.content

    return results
```

---

## Phase 5: Synthetic query generation (code)

**Effort**: Medium
**Impact**: Medium-High for code retrieval
**Files**: `src/chunking/code_chunker.py`, new `src/chunking/query_generator.py`

### 5.1 Generate questions per code chunk

```python
# src/chunking/query_generator.py
class CodeQueryGenerator:
    """Generate natural language queries for code chunks."""

    PROMPT = """
Given this code snippet from an Ethereum client:

```{language}
{code}
```

Generate 2-3 natural language questions that a developer might ask that this code would answer. Focus on:
- What functionality does this implement?
- What Ethereum concept does this relate to?
- What problem does this solve?

Return only the questions, one per line.
"""

    def generate_queries(
        self,
        code: str,
        language: str,
        function_name: str,
    ) -> list[str]:
        response = self.client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": self.PROMPT.format(
                    language=language,
                    code=code,
                ),
            }],
        )

        queries = response.content[0].text.strip().split("\n")
        return [q.strip("- ").strip() for q in queries if q.strip()]
```

### 5.2 Dual embedding strategy

Store both code embedding AND query embeddings:

```python
# Embed the code itself
code_embedding = embedder.embed_code(chunk.content, language)

# Embed synthetic queries
queries = query_generator.generate_queries(chunk.content, language, chunk.function_name)
query_embeddings = [embedder.embed_query(q) for q in queries]

# Store all embeddings pointing to same chunk
await store.store_code_chunk(
    chunk=chunk,
    code_embedding=code_embedding,
    query_embeddings=query_embeddings,
)
```

### 5.3 Schema for multi-embedding

```sql
CREATE TABLE code_chunk_embeddings (
    id SERIAL PRIMARY KEY,
    chunk_id TEXT REFERENCES chunks(chunk_id),
    embedding_type TEXT,  -- 'code' or 'query'
    query_text TEXT,      -- NULL for code embeddings
    embedding vector(1024)
);

CREATE INDEX idx_code_embeddings ON code_chunk_embeddings
    USING ivfflat (embedding vector_cosine_ops);
```

---

## Phase 6: Code-specific improvements

**Effort**: Medium
**Impact**: Medium
**Files**: `src/chunking/code_chunker.py`

### 6.1 AST-aware block splitting

Replace brace counting with tree-sitter AST traversal:

```python
def _split_large_unit_ast(
    self,
    unit: CodeUnit,
    parsed_file: ParsedFile,
    max_tokens: int,
) -> list[CodeChunk]:
    """Split large function using AST block boundaries."""

    # Find the function node in AST
    func_node = self._find_function_node(parsed_file.tree, unit.name)

    # Get block-level children (statements, control flow)
    blocks = self._get_block_children(func_node)

    chunks = []
    current_blocks = []
    current_tokens = self._count_tokens(self._get_function_header(func_node))

    for block in blocks:
        block_text = self._node_to_text(block)
        block_tokens = self._count_tokens(block_text)

        if current_tokens + block_tokens > max_tokens and current_blocks:
            # Create chunk from accumulated blocks
            chunks.append(self._create_split_chunk(
                unit, current_blocks, chunk_index=len(chunks)
            ))
            current_blocks = []
            current_tokens = self._count_tokens(self._get_function_header(func_node))

        current_blocks.append(block)
        current_tokens += block_tokens

    # Last chunk
    if current_blocks:
        chunks.append(self._create_split_chunk(unit, current_blocks, len(chunks)))

    return chunks
```

### 6.2 Include callers/callees context

```python
def chunk_with_call_context(
    self,
    unit: CodeUnit,
    call_graph: dict[str, list[str]],
) -> CodeChunk:
    """Include caller/callee information in chunk."""

    callers = call_graph.get(f"callers:{unit.name}", [])[:3]
    callees = call_graph.get(f"callees:{unit.name}", [])[:3]

    context_parts = []
    if callers:
        context_parts.append(f"Called by: {', '.join(callers)}")
    if callees:
        context_parts.append(f"Calls: {', '.join(callees)}")

    context = "\n".join(context_parts)

    return CodeChunk(
        content=f"{context}\n\n{unit.content}" if context else unit.content,
        # ... other fields
    )
```

---

## Phase 7: Document-specific enhancements

### 7.1 Forum: Quote resolution

```python
def _resolve_quotes(self, content: str, topic: LoadedForumTopic) -> str:
    """Replace quote blocks with references to original posts."""

    # Pattern: [quote="username, post:N, topic:M"]...[/quote]
    quote_pattern = r'\[quote="([^"]+), post:(\d+)[^\]]*"\](.*?)\[/quote\]'

    def replace_quote(match):
        username = match.group(1)
        post_num = match.group(2)
        quoted_text = match.group(3)[:100]  # Truncate
        return f'> @{username} (post #{post_num}): "{quoted_text}..."'

    return re.sub(quote_pattern, replace_quote, content, flags=re.DOTALL)
```

### 7.2 Papers: Citation resolution

```python
def _resolve_citations(self, content: str, references: list[str]) -> str:
    """Replace [1], [2] with actual reference text."""

    def replace_citation(match):
        ref_num = int(match.group(1))
        if ref_num <= len(references):
            ref_text = references[ref_num - 1][:50]
            return f'[{ref_num}: {ref_text}...]'
        return match.group(0)

    return re.sub(r'\[(\d+)\]', replace_citation, content)
```

### 7.3 Transcripts: Topic segmentation

```python
def _segment_by_topic(
    self,
    transcript: ACDTranscript,
) -> list[tuple[str, list[SpeakerSegment]]]:
    """Segment transcript by agenda topics."""

    # Common topic indicators
    topic_patterns = [
        r"(?:let's|let us|now|next|moving on to)\s+(?:talk about|discuss|move to)\s+(.+)",
        r"agenda item[:\s]+(.+)",
        r"topic[:\s]+(.+)",
    ]

    segments_by_topic = []
    current_topic = "Introduction"
    current_segments = []

    for segment in transcript.segments:
        # Check if this segment starts a new topic
        for pattern in topic_patterns:
            match = re.search(pattern, segment.text, re.IGNORECASE)
            if match:
                if current_segments:
                    segments_by_topic.append((current_topic, current_segments))
                current_topic = match.group(1).strip()
                current_segments = []
                break

        current_segments.append(segment)

    if current_segments:
        segments_by_topic.append((current_topic, current_segments))

    return segments_by_topic
```

---

## Phase 8: Hybrid BM25 + dense retrieval

**Effort**: Medium
**Impact**: High (improves tail recall)
**Files**: `src/retrieval/hybrid_retriever.py` (new), `src/storage/pg_vector_store.py`

### 8.1 Add BM25 index to PostgreSQL

```sql
-- Add full-text search column
ALTER TABLE chunks ADD COLUMN content_tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;

CREATE INDEX idx_chunks_fts ON chunks USING GIN (content_tsv);
```

### 8.2 Create HybridRetriever

```python
# src/retrieval/hybrid_retriever.py
class HybridRetriever:
    """Combine BM25 and dense vector retrieval."""

    def __init__(
        self,
        store: PgVectorStore,
        embedder: VoyageEmbedder,
        bm25_weight: float = 0.3,
        dense_weight: float = 0.7,
    ):
        self.store = store
        self.embedder = embedder
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight

    async def retrieve(
        self,
        query: str,
        top_k: int = 20,
    ) -> list[SearchResult]:
        """Retrieve using both BM25 and dense, then fuse scores."""

        # Dense retrieval
        query_embedding = self.embedder.embed_query(query)
        dense_results = await self.store.search(query_embedding, limit=top_k * 2)

        # BM25 retrieval
        bm25_results = await self.store.search_bm25(query, limit=top_k * 2)

        # Reciprocal Rank Fusion (RRF)
        fused = self._rrf_fusion(dense_results, bm25_results, k=60)

        return fused[:top_k]

    def _rrf_fusion(
        self,
        dense: list[SearchResult],
        bm25: list[SearchResult],
        k: int = 60,
    ) -> list[SearchResult]:
        """Reciprocal Rank Fusion scoring."""
        scores = {}

        for rank, result in enumerate(dense):
            chunk_id = result.chunk.chunk_id
            scores[chunk_id] = scores.get(chunk_id, 0) + self.dense_weight / (k + rank + 1)

        for rank, result in enumerate(bm25):
            chunk_id = result.chunk.chunk_id
            scores[chunk_id] = scores.get(chunk_id, 0) + self.bm25_weight / (k + rank + 1)

        # Sort by fused score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        # Rebuild results
        all_results = {r.chunk.chunk_id: r for r in dense + bm25}
        return [all_results[cid] for cid in sorted_ids if cid in all_results]
```

### 8.3 BM25 search method

```python
# Add to PgVectorStore
async def search_bm25(
    self,
    query: str,
    limit: int = 20,
) -> list[SearchResult]:
    """Full-text search using PostgreSQL ts_rank."""
    async with self.pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT chunk_id, document_id, content, section_path,
                   ts_rank(content_tsv, plainto_tsquery('english', $1)) as score
            FROM chunks
            WHERE content_tsv @@ plainto_tsquery('english', $1)
            ORDER BY score DESC
            LIMIT $2
        """, query, limit)

        return [self._row_to_search_result(row) for row in rows]
```

---

## Phase 9: Query-time graph expansion

**Effort**: Medium
**Impact**: High for multi-hop code questions
**Files**: `src/retrieval/graph_expander.py` (new), `src/graph/falkordb_store.py`

### 9.1 Graph expansion at query time

Based on [RANGER](https://arxiv.org/html/2509.25257) and [DKB](https://arxiv.org/html/2601.08773) papers.

```python
# src/retrieval/graph_expander.py
class GraphExpander:
    """Expand retrieval results using code knowledge graph."""

    def __init__(
        self,
        graph_store: FalkorDBStore,
        max_hops: int = 2,
        max_expanded: int = 5,
    ):
        self.graph = graph_store
        self.max_hops = max_hops
        self.max_expanded = max_expanded

    async def expand(
        self,
        results: list[SearchResult],
        query_type: str = "implementation",
    ) -> list[SearchResult]:
        """Expand results by traversing code relationships."""

        expanded = list(results)
        seen_ids = {r.chunk.chunk_id for r in results}

        for result in results[:5]:  # Expand top-5 only
            # Get related nodes from graph
            related = await self._get_related(result.chunk, query_type)

            for rel_chunk_id, relationship in related:
                if rel_chunk_id not in seen_ids:
                    chunk = await self._fetch_chunk(rel_chunk_id)
                    if chunk:
                        expanded.append(SearchResult(
                            chunk=chunk,
                            similarity=result.similarity * 0.8,  # Decay
                            relationship=relationship,
                        ))
                        seen_ids.add(rel_chunk_id)

                        if len(expanded) >= len(results) + self.max_expanded:
                            break

        return expanded

    async def _get_related(
        self,
        chunk: Chunk,
        query_type: str,
    ) -> list[tuple[str, str]]:
        """Get related chunks based on query type."""

        if query_type == "implementation":
            # Find callers and callees
            return await self.graph.query("""
                MATCH (f:Function {chunk_id: $chunk_id})-[r:CALLS|CALLED_BY]->(related)
                RETURN related.chunk_id, type(r)
                LIMIT $limit
            """, chunk_id=chunk.chunk_id, limit=self.max_expanded)

        elif query_type == "spec":
            # Find EIP implementations
            return await self.graph.query("""
                MATCH (e:EIP)-[:IMPLEMENTED_BY]->(f:Function {chunk_id: $chunk_id})
                MATCH (f)-[:CALLS]->(related)
                RETURN related.chunk_id, 'RELATED_IMPL'
                LIMIT $limit
            """, chunk_id=chunk.chunk_id, limit=self.max_expanded)

        return []
```

### 9.2 Query type detection

```python
def detect_query_type(query: str) -> str:
    """Detect if query is about spec, implementation, or general."""

    impl_patterns = [
        r"how is .* implemented",
        r"implementation of",
        r"code for",
        r"function that",
        r"where is .* defined",
    ]

    spec_patterns = [
        r"what does eip.* specify",
        r"according to eip",
        r"specification",
        r"what is the .* mechanism",
    ]

    query_lower = query.lower()

    for pattern in impl_patterns:
        if re.search(pattern, query_lower):
            return "implementation"

    for pattern in spec_patterns:
        if re.search(pattern, query_lower):
            return "spec"

    return "general"
```

### 9.3 Integrate with retrieval pipeline

```python
class EnhancedRetriever:
    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        graph_expander: GraphExpander,
        reranker: VoyageReranker,
    ):
        self.hybrid = hybrid_retriever
        self.expander = graph_expander
        self.reranker = reranker

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        expand_graph: bool = True,
    ) -> RetrievalResult:
        # Step 1: Hybrid retrieval
        results = await self.hybrid.retrieve(query, top_k=top_k * 2)

        # Step 2: Graph expansion (for code queries)
        if expand_graph:
            query_type = detect_query_type(query)
            if query_type in ("implementation", "spec"):
                results = await self.expander.expand(results, query_type)

        # Step 3: Rerank
        results = self.reranker.rerank(query, results)[:top_k]

        return RetrievalResult(results=results)
```

---

## Phase 10: Late chunking

**Effort**: High
**Impact**: Medium-High for dense documents
**Files**: `src/chunking/late_chunker.py` (new)

### 10.1 Late chunking concept

Traditional: Chunk → Embed each chunk separately
Late: Embed full document → Pool embeddings into chunks

This preserves cross-chunk context in the embeddings.

### 10.2 Implementation with jina-embeddings-v3

```python
# src/chunking/late_chunker.py
from transformers import AutoModel, AutoTokenizer
import torch

class LateChunker:
    """Late chunking using long-context embedding models."""

    def __init__(
        self,
        model_name: str = "jinaai/jina-embeddings-v3",
        chunk_size: int = 512,
        overlap: int = 64,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_and_embed(
        self,
        document: str,
        document_id: str,
    ) -> list[EmbeddedChunk]:
        """Embed full document, then pool into chunk embeddings."""

        # Step 1: Tokenize full document
        inputs = self.tokenizer(
            document,
            return_tensors="pt",
            max_length=8192,  # Long context
            truncation=True,
        )

        # Step 2: Get token-level embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            token_embeddings = outputs.last_hidden_state[0]  # [seq_len, hidden_dim]

        # Step 3: Define chunk boundaries
        tokens = inputs["input_ids"][0]
        chunk_boundaries = self._get_chunk_boundaries(len(tokens))

        # Step 4: Pool embeddings for each chunk
        chunks = []
        for i, (start, end) in enumerate(chunk_boundaries):
            # Mean pooling over chunk tokens
            chunk_embedding = token_embeddings[start:end].mean(dim=0)

            # Decode chunk text
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)

            chunks.append(EmbeddedChunk(
                chunk=Chunk(
                    chunk_id=f"{document_id}-late-{i}",
                    document_id=document_id,
                    content=chunk_text,
                    token_count=end - start,
                    chunk_index=i,
                ),
                embedding=chunk_embedding.tolist(),
                model="jina-embeddings-v3-late",
            ))

        return chunks

    def _get_chunk_boundaries(
        self,
        seq_len: int,
    ) -> list[tuple[int, int]]:
        """Calculate chunk start/end positions with overlap."""
        boundaries = []
        start = 0

        while start < seq_len:
            end = min(start + self.chunk_size, seq_len)
            boundaries.append((start, end))
            start = end - self.overlap

            if start >= seq_len - self.overlap:
                break

        return boundaries
```

### 10.3 When to use late chunking

| Document Type | Use Late Chunking? | Rationale |
|---------------|-------------------|-----------|
| EIPs | ✅ Yes | Dense specs with cross-references |
| Papers | ✅ Yes | Dense, equations reference earlier |
| Forum posts | ❌ No | Already has natural boundaries |
| Transcripts | ❌ No | Speaker boundaries matter more |
| Code | ❌ No | Function boundaries matter more |

---

## Phase 11: HyDE (Hypothetical Document Embeddings)

**Effort**: Low
**Impact**: Medium for complex queries
**Files**: `src/retrieval/hyde.py` (new)

### 11.1 HyDE concept

Instead of embedding the query directly, generate a hypothetical answer and embed that.

### 11.2 Implementation

```python
# src/retrieval/hyde.py
from anthropic import Anthropic

class HyDERetriever:
    """Hypothetical Document Embedding retriever."""

    HYDE_PROMPT = """
Given this question about Ethereum protocol:

{query}

Write a short, technical paragraph that would answer this question.
Include specific technical details, EIP numbers if relevant, and code concepts.
Write as if this is an excerpt from documentation or a code comment.
"""

    def __init__(
        self,
        embedder: VoyageEmbedder,
        store: PgVectorStore,
        llm_model: str = "claude-3-5-haiku-20241022",
    ):
        self.embedder = embedder
        self.store = store
        self.client = Anthropic()
        self.llm_model = llm_model

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        use_hyde: bool = True,
    ) -> list[SearchResult]:
        """Retrieve using HyDE for complex queries."""

        if use_hyde and self._should_use_hyde(query):
            # Generate hypothetical document
            hypothetical = self._generate_hypothetical(query)

            # Embed the hypothetical document
            embedding = self.embedder.embed_text(hypothetical, input_type="document")
        else:
            # Standard query embedding
            embedding = self.embedder.embed_query(query)

        return await self.store.search(embedding, limit=top_k)

    def _generate_hypothetical(self, query: str) -> str:
        """Generate a hypothetical document answering the query."""
        response = self.client.messages.create(
            model=self.llm_model,
            max_tokens=300,
            messages=[{
                "role": "user",
                "content": self.HYDE_PROMPT.format(query=query),
            }],
        )
        return response.content[0].text

    def _should_use_hyde(self, query: str) -> bool:
        """Determine if HyDE would help this query."""
        # Use HyDE for complex, multi-part questions
        indicators = [
            len(query.split()) > 10,  # Long queries
            "how" in query.lower() and "implement" in query.lower(),
            "difference between" in query.lower(),
            "compare" in query.lower(),
            "explain" in query.lower(),
        ]
        return any(indicators)
```

### 11.3 Code-specific HyDE prompt

```python
CODE_HYDE_PROMPT = """
Given this question about Ethereum client code:

{query}

Write a code comment or docstring that would appear above a function that answers this question.
Include:
- Function purpose
- Key parameters
- What EIP it implements (if applicable)
- Related functions it calls

Write only the comment/docstring, no actual code.
"""
```

---

## Phase 12: ColBERT multi-vector embeddings (Optional)

**Effort**: High
**Impact**: High (but high storage cost)
**Files**: `src/embeddings/colbert_embedder.py` (new), schema changes

### 12.1 ColBERT concept

Instead of one vector per chunk, store one vector per token. Score using MaxSim.

### 12.2 Storage requirements

| Approach | Vectors per Chunk | Storage Multiplier |
|----------|-------------------|-------------------|
| Single-vector | 1 | 1x |
| ColBERT (512 tokens) | 512 | ~50x (with compression) |

### 12.3 Implementation sketch

```python
# src/embeddings/colbert_embedder.py
from colbert.infra import Run, RunConfig
from colbert.modeling.checkpoint import Checkpoint

class ColBERTEmbedder:
    """ColBERT token-level embeddings."""

    def __init__(
        self,
        checkpoint: str = "colbert-ir/colbertv2.0",
    ):
        self.checkpoint = Checkpoint(checkpoint)

    def embed_document(self, text: str) -> list[list[float]]:
        """Get token-level embeddings for a document."""
        return self.checkpoint.docFromText([text])[0].tolist()

    def embed_query(self, query: str) -> list[list[float]]:
        """Get token-level embeddings for a query."""
        return self.checkpoint.queryFromText([query])[0].tolist()

    def score(
        self,
        query_vecs: list[list[float]],
        doc_vecs: list[list[float]],
    ) -> float:
        """MaxSim scoring between query and document."""
        # For each query token, find max similarity to any doc token
        max_sims = []
        for q_vec in query_vecs:
            sims = [self._cosine(q_vec, d_vec) for d_vec in doc_vecs]
            max_sims.append(max(sims))

        return sum(max_sims)
```

### 12.4 Schema for multi-vector storage

```sql
CREATE TABLE colbert_embeddings (
    id SERIAL PRIMARY KEY,
    chunk_id TEXT REFERENCES chunks(chunk_id),
    token_index INTEGER,
    embedding vector(128)  -- ColBERT uses 128-dim
);

CREATE INDEX idx_colbert_ivfflat ON colbert_embeddings
    USING ivfflat (embedding vector_cosine_ops);
```

### 12.5 Recommendation

**Defer ColBERT to Phase 2** unless storage costs are acceptable. The other improvements (reranking, contextual retrieval) provide most of the benefit at lower complexity.

---

## Phase 13: Add overlap to CodeChunker

**Effort**: Low
**Impact**: Medium
**Files**: `src/chunking/code_chunker.py`

### 13.1 Update CodeChunker

```python
class CodeChunker:
    def __init__(
        self,
        max_tokens: int = 512,
        overlap_tokens: int = 64,  # Add overlap
        encoding_name: str = "cl100k_base",
    ):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.tokenizer = tiktoken.get_encoding(encoding_name)

    def _split_large_unit(
        self,
        unit: CodeUnit,
        language: str,
    ) -> list[CodeChunk]:
        # ... existing logic ...

        # Add overlap when creating split chunks
        for i, block_group in enumerate(block_groups):
            # Include overlap from previous chunk
            if i > 0 and self.overlap_tokens > 0:
                overlap_content = self._get_overlap_from_previous(
                    block_groups[i-1], self.overlap_tokens
                )
                block_group = overlap_content + block_group

            chunks.append(self._create_chunk(...))
```

---

## Implementation schedule

### Week 1: Foundation (P0)

- [x] Update CodeEmbedder to voyage-code-3
- [ ] Create VoyageReranker class
- [ ] Integrate reranker into retrieval pipeline
- [ ] Update ingest_client.py to use CodeChunker + CodeEmbedder
- [ ] Add overlap to CodeChunker

### Week 2: Hybrid retrieval (P0/P1)

- [ ] Add BM25 full-text search index to PostgreSQL
- [ ] Create HybridRetriever with RRF fusion
- [ ] Integrate hybrid + rerank pipeline
- [ ] Benchmark retrieval improvements

### Week 3: Contextual retrieval (P0)

- [ ] Create ChunkContextGenerator
- [ ] Implement document-specific context prompts
- [ ] Add prompt caching for cost efficiency
- [ ] Update all chunkers to support contextual mode

### Week 4: Graph expansion (P1)

- [ ] Create GraphExpander class
- [ ] Implement query type detection
- [ ] Add bidirectional traversal for code queries
- [ ] Integrate with retrieval pipeline

### Week 5: Hierarchical chunking (P1)

- [ ] Schema migration for parent-child relationships
- [ ] Update SectionChunker for hierarchical EIPs
- [ ] Update CodeChunker for hierarchical code (file → function)
- [ ] Update retrieval to include parent context

### Week 6: Code-specific (P1/P2)

- [ ] Implement CodeQueryGenerator (synthetic queries)
- [ ] Add multi-embedding storage for code
- [ ] AST-aware block splitting
- [ ] Call graph context integration

### Week 7: Document-specific (P2)

- [ ] Forum quote resolution
- [ ] Paper citation resolution
- [ ] Transcript topic segmentation

### Week 8: Advanced techniques (P2)

- [ ] Implement HyDE for complex queries
- [ ] Implement late chunking for EIPs and papers
- [ ] End-to-end testing
- [ ] Re-index all documents

### Future (P3)

- [ ] Evaluate ColBERT multi-vector (if storage permits)
- [ ] Evaluate self-hosted Qodo-Embed-1-7B

---

## Cost estimates

### One-time re-indexing

| Task | Estimated Cost |
|------|----------------|
| Contextual generation (all docs) | ~$50 (with caching) |
| Synthetic query generation (code) | ~$20 |
| Re-embedding code with voyage-code-3 | ~$10 |
| Late chunking embeddings (EIPs + papers) | ~$15 |
| **Total** | ~$95 |

### Ongoing costs (per 1000 queries)

| Component | Cost |
|-----------|------|
| voyage-code-3 embedding | $0.06 |
| voyage-rerank-2.5 | $0.05 |
| HyDE generation (20% of queries) | $0.02 |
| Graph expansion (code queries) | ~$0.00 (local DB) |
| **Total** | ~$0.13 per 1000 queries |

### Infrastructure

| Component | Requirement |
|-----------|-------------|
| PostgreSQL storage | +20% for BM25 index |
| FalkorDB | Already deployed |
| GPU (if late chunking) | Optional, can use CPU |
| ColBERT storage (if enabled) | +50x chunk storage |

---

## Success metrics

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| Code retrieval accuracy | Baseline | +20% | CoIR-style eval |
| Top-10 recall | ~85% | >95% | Test queries |
| MRR (Mean Reciprocal Rank) | TBD | +30% | Test queries |
| Multi-hop code questions | Poor | Good | Manual eval |
| Cross-EIP questions | Moderate | Good | Test suite |
| Retrieval latency (p95) | ~200ms | <300ms | With reranking |
| User satisfaction | N/A | Track | Query feedback |

### Benchmark test set

Create a test set of 100 queries across categories:

| Category | Count | Example |
|----------|-------|---------|
| EIP spec lookup | 20 | "What is the base fee formula in EIP-1559?" |
| Code implementation | 20 | "How is blob gas calculated in go-ethereum?" |
| Multi-hop code | 15 | "What functions call the base fee calculation?" |
| Cross-document | 15 | "How does EIP-4844 relate to EIP-1559?" |
| Forum discussion | 15 | "What concerns were raised about blob pricing?" |
| Transcript | 15 | "What did Tim Beiko say about Dencun timeline?" |

---

## Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Context generation adds latency | Pre-compute at indexing time |
| Reranking adds latency | Cache rerank results for common queries |
| Cost overrun on context gen | Use prompt caching, batch processing |
| Breaking existing queries | A/B test before full rollout |
| voyage-code-3 dimension mismatch | Verify 1024 dims match existing schema |
| HyDE generates wrong hypothetical | Only use for complex queries, fallback to direct |
| Graph expansion returns noise | Limit to 2 hops, score decay |
| Late chunking model size | Use CPU inference, batch processing |
| BM25 index size growth | Monitor, consider partial indexing |
| ColBERT storage explosion | Defer to Phase 2, evaluate ROI first |

## Dependencies

| Phase | Depends On |
|-------|------------|
| Phase 2 (Reranking) | Phase 1 (embedding model) |
| Phase 3 (Contextual) | None (can run in parallel) |
| Phase 4 (Graph expansion) | Existing FalkorDB graph |
| Phase 5 (Hierarchical) | Phase 3 (contextual) |
| Phase 6 (Code-specific) | Phase 1, Phase 4 |
| Phase 8 (Hybrid BM25) | None |
| Phase 10 (Late chunking) | None (separate embedding space) |
| Phase 11 (HyDE) | Phase 2 (reranking) |
| Phase 12 (ColBERT) | None (optional, independent) |

---

## References

- [Anthropic Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- [voyage-code-3 announcement](https://blog.voyageai.com/2024/12/04/voyage-code-3/)
- [voyage-rerank-2.5](https://blog.voyageai.com/2025/08/11/rerank-2-5/)
- [Voyage 4 model family](https://blog.voyageai.com/2026/01/15/voyage-4/)
- [RANGER: Graph-Enhanced Retrieval](https://arxiv.org/html/2509.25257)
- [Late Chunking paper](https://arxiv.org/abs/2409.04701)
- [Qodo-Embed-1](https://www.qodo.ai/blog/qodo-embed-1-code-embedding-code-retrieval/)
