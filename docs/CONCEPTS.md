# Key concepts

This document explains the key concepts you'll encounter when using the Ethereum Protocol Intelligence System.

## Ethereum concepts

### EIP (Ethereum Improvement Proposal)

An EIP is a design document providing information to the Ethereum community about a proposed feature, process, or standard. EIPs are the primary mechanism for proposing new features, collecting community input, and documenting design decisions.

**EIP types:**

| Type | Description | Examples |
|------|-------------|----------|
| **Standards Track** | Changes affecting Ethereum implementations | EIP-1559, EIP-4844 |
| **Core** | Consensus-critical changes requiring a fork | EIP-1559 (London fork) |
| **Networking** | Network protocol changes | EIP-8 (devp2p) |
| **Interface** | API/RPC improvements | EIP-1193 (Provider API) |
| **ERC** | Application-level standards | ERC-20, ERC-721 |
| **Meta** | Process and governance | EIP-1 (EIP Purpose) |
| **Informational** | Guidelines and information | EIP-2228 |

**EIP lifecycle:**

```
Draft → Review → Last Call → Final
                    ↓
              Stagnant/Withdrawn
```

### Key EIPs to know

| EIP | Name | Significance |
|-----|------|--------------|
| **EIP-1559** | Fee Market Change | Base fee burning, dynamic block sizes |
| **EIP-4844** | Shard Blob Transactions | Proto-danksharding, blob data |
| **EIP-1** | EIP Purpose and Guidelines | Defines the EIP process itself |
| **EIP-20** | Token Standard (ERC-20) | Fungible token interface |
| **EIP-721** | NFT Standard (ERC-721) | Non-fungible token interface |
| **EIP-4337** | Account Abstraction | Smart contract wallets |

### Base fee

Introduced in EIP-1559, the base fee is a minimum fee per gas that all transactions must pay. It:
- Adjusts dynamically based on block fullness
- Is burned (removed from circulation)
- Makes fee estimation more predictable

### Blob transactions

Introduced in EIP-4844, blob transactions carry large data "blobs" that:
- Cannot be accessed by smart contracts
- Are stored temporarily (~18 days)
- Use a separate fee market
- Enable cheaper rollup data posting

## RAG concepts

### RAG (Retrieval-Augmented Generation)

RAG is a technique that enhances LLM responses by retrieving relevant documents before generation. Instead of relying solely on the model's training data, RAG:

1. **Retrieves** relevant chunks from a knowledge base
2. **Augments** the prompt with this context
3. **Generates** a response grounded in the retrieved information

```
Query → Retrieve chunks → Build prompt → Generate response
         ↓                    ↓
    Vector search      [Context] + [Query]
```

### Chunks

Documents are split into smaller pieces called chunks for efficient retrieval. This system uses:

| Strategy | Description | Use case |
|----------|-------------|----------|
| **Fixed chunking** | Split by token count | Simple, predictable |
| **Section chunking** | Split by document structure | Preserves EIP sections |
| **Code chunking** | Split by function boundaries | Source code analysis |

### Embeddings

Embeddings are dense vector representations of text that capture semantic meaning. Similar texts have similar vectors, enabling semantic search.

```
"What is EIP-1559?" → [0.12, -0.34, 0.56, ...] (1024 dimensions)
```

This system uses:
- **Voyage AI** (voyage-4-large) for document embeddings
- **voyage-code-2** for source code embeddings

### Vector similarity search

Finding relevant chunks by comparing embedding vectors. The system uses cosine similarity:

```
similarity = dot(query_vector, chunk_vector) / (|query| × |chunk|)
```

Higher similarity (closer to 1.0) means more relevant content.

### pgvector

PostgreSQL extension that adds vector data types and similarity search operators. Enables storing and querying embeddings directly in the database.

## Validation concepts

### NLI (Natural Language Inference)

NLI determines the logical relationship between two texts:

| Relationship | Meaning | Example |
|--------------|---------|---------|
| **Entailment** | Evidence supports claim | "Base fee is burned" ✓ |
| **Contradiction** | Evidence contradicts claim | "Base fee goes to miners" ✗ |
| **Neutral** | No clear relationship | Unrelated information |

### Claim decomposition

Breaking complex claims into atomic facts for individual verification:

```
Original: "EIP-1559 introduced base fee burning and dynamic block sizes"
    ↓
Atomic facts:
  1. "EIP-1559 introduced base fee burning"
  2. "EIP-1559 introduced dynamic block sizes"
```

### Support levels

Classification of how well evidence supports a claim:

| Level | Meaning | Action |
|-------|---------|--------|
| **STRONG** | Evidence clearly supports | ✅ Trustworthy |
| **PARTIAL** | Some support, not complete | 🟡 Acceptable |
| **WEAK** | Minimal support | ⚠️ Flag for review |
| **NONE** | No supporting evidence | ❌ Unsupported |
| **CONTRADICTION** | Evidence contradicts | 🚫 Incorrect |

### Evidence span

An immutable reference to a specific piece of source text:

```python
EvidenceSpan(
    document_id="eip-1559",
    section_path="Specification > Base Fee",
    span_text="The base fee is burned...",
    start_offset=1234,
    end_offset=1456
)
```

### Evidence ledger

A record linking claims to their supporting evidence:

```
Claim: "The base fee is burned"
  └── Evidence: EIP-1559, Specification section
  └── Evidence: EIP-1559, Abstract section
```

## System concepts

### Query modes

| Mode | Description | Use case |
|------|-------------|----------|
| **Simple** | Fast RAG, no citations | Quick exploration |
| **Cited** | Adds source attribution | Research, documentation |
| **Validated** | NLI claim verification | Fact-checking, high-stakes |

### Concept resolution

Mapping aliases to canonical terms for better retrieval:

```
"proto-danksharding" → "EIP-4844"
"1559" → "EIP-1559"
"base fee" → "base_fee"
"blob tx" → "blob_transaction"
```

### Cost routing

Selecting the appropriate model based on query complexity:

| Tier | Model | Use case |
|------|-------|----------|
| **Fast** | Claude Haiku | Simple factual queries |
| **Standard** | Claude Sonnet | Most queries |
| **Thorough** | Claude Opus | Complex analysis |

### Agentic retrieval

Using a ReAct-style agent to iteratively retrieve and reason:

```
Think → Retrieve → Observe → Think → Retrieve → Answer
```

Includes:
- **Budget enforcement**: Limits on retrievals, tokens, LLM calls
- **Backtracking**: Detecting dead-end retrieval paths
- **Reflection**: Quality assessment of retrieved context
