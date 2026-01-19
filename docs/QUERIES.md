# Querying the system

This guide covers how to effectively query the Ethereum Protocol Intelligence System.

## Query methods

### Command-line interface

The primary way to query the system:

```bash
uv run python scripts/query_cli.py "Your question here"
```

### REST API

For programmatic access:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Your question here"}'
```

### Python API

For integration into other applications:

```python
import asyncio
from src.embeddings import VoyageEmbedder
from src.storage import PgVectorStore
from src.retrieval import SimpleRetriever
from src.generation import SimpleGenerator

async def query(question: str) -> str:
    store = PgVectorStore()
    await store.connect()

    embedder = VoyageEmbedder()
    retriever = SimpleRetriever(embedder, store)
    generator = SimpleGenerator(retriever)

    result = await generator.generate(question)
    await store.close()

    return result.response
```

## Query modes

### Simple mode

Fast, basic RAG generation without citations or validation.

```bash
uv run python scripts/query_cli.py "What is EIP-1559?" --mode simple
```

**Best for:**
- Quick exploration
- Simple factual questions
- Development and testing

**Output:**
```
RESPONSE (Simple Mode - Phase 0)
============================================================
EIP-1559 is a significant Ethereum Improvement Proposal that reformed
the transaction fee mechanism...

------------------------------------------------------------
Tokens: 1256 input, 268 output
```

### Cited mode

Adds inline citations and source tracking.

```bash
uv run python scripts/query_cli.py "How do blob transactions work?" --mode cited
```

**Best for:**
- Research and documentation
- Verifying information sources
- Understanding which EIPs inform the answer

**Output:**
```
RESPONSE (Cited Mode - Phase 1)
============================================================
Blob transactions, introduced in EIP-4844 [1], are a new transaction
type that carries large data blobs [2]. These blobs cannot be accessed
by smart contracts [1] but their commitments can be verified...

------------------------------------------------------------
SOURCES:
------------------------------------------------------------
  - EIP-4844: Abstract
  - EIP-4844: Specification
  - EIP-7762: Rationale
```

### Validated mode

Verifies claims using NLI (Natural Language Inference).

```bash
uv run python scripts/query_cli.py "What is the base fee?" --mode validated
```

**Best for:**
- High-stakes queries
- Fact-checking
- Understanding answer reliability

**Output:**
```
RESPONSE (Validated Mode - Phase 2)
============================================================
The base fee is a mechanism introduced in EIP-1559 that determines
the minimum fee per gas. ⚠️ [WEAKLY SUPPORTED]

It adjusts dynamically based on block utilization...

---
⚠️ **Validation Notice**: 2 claim(s) may have insufficient evidence support.

------------------------------------------------------------
VALIDATION SUMMARY:
------------------------------------------------------------
  Total claims: 5
  Supported: 3
  Weak: 1
  Unsupported: 1
  Support ratio: 60.0%
  Trustworthy: No
```

## CLI options

```bash
uv run python scripts/query_cli.py --help
```

| Option | Default | Description |
|--------|---------|-------------|
| `--mode` | `simple` | Query mode: `simple`, `cited`, `validated` |
| `--top-k` | `5` | Number of chunks to retrieve |
| `--no-sources` | `false` | Hide source citations in output |
| `--local` | `false` | Use local embeddings (no API) |

### Examples

```bash
# Retrieve more context
uv run python scripts/query_cli.py "Explain EIP-4844 in detail" --top-k 10

# Hide source list (still shows inline citations)
uv run python scripts/query_cli.py "What is EIP-1559?" --mode cited --no-sources

# Use local embeddings (offline, no API key needed)
uv run python scripts/query_cli.py "What is EIP-1559?" --local
```

## Query techniques

### Be specific

More specific queries get better results:

```bash
# ❌ Too vague
uv run python scripts/query_cli.py "Tell me about fees"

# ✅ Specific
uv run python scripts/query_cli.py "How does the base fee adjustment work in EIP-1559?"
```

### Reference EIP numbers

The system understands EIP references:

```bash
# These all work
uv run python scripts/query_cli.py "What is EIP-1559?"
uv run python scripts/query_cli.py "Explain 1559"
uv run python scripts/query_cli.py "What does EIP-1559 change about transaction fees?"
```

### Ask comparative questions

Compare multiple EIPs or concepts:

```bash
uv run python scripts/query_cli.py "What's the difference between EIP-1559 and the legacy fee mechanism?"

uv run python scripts/query_cli.py "How do EIP-4844 blobs compare to calldata for rollups?"
```

### Ask about relationships

Explore EIP dependencies and relationships:

```bash
uv run python scripts/query_cli.py "Which EIPs does EIP-4844 depend on?"

uv run python scripts/query_cli.py "What EIPs are related to account abstraction?"
```

### Ask technical details

Get specific technical information:

```bash
uv run python scripts/query_cli.py "What are the exact parameters for EIP-1559 base fee adjustment?"

uv run python scripts/query_cli.py "What is the gas cost of the TSTORE opcode in EIP-1153?"
```

## Understanding results

### Similarity scores

Higher scores indicate more relevant chunks:

```
retrieved_chunks    top_similarity=0.58 num_results=5
```

| Score | Interpretation |
|-------|----------------|
| 0.7+ | Highly relevant |
| 0.5-0.7 | Relevant |
| 0.3-0.5 | Somewhat relevant |
| <0.3 | Possibly off-topic |

### Validation flags

In validated mode, claims are flagged with support levels:

| Flag | Meaning | Action |
|------|---------|--------|
| (no flag) | Strongly supported | Trust the claim |
| ⚠️ [WEAKLY SUPPORTED] | Some support | Verify independently |
| ⚠️ [NO SUPPORTING EVIDENCE] | No evidence found | Treat with caution |
| ⚠️ [CONTRADICTED BY EVIDENCE] | Evidence disagrees | Do not trust |

### Token usage

Output shows API token consumption:

```
Tokens: 1256 input, 268 output
```

Useful for:
- Cost estimation
- Debugging context length issues
- Optimizing retrieval parameters

## Troubleshooting queries

### "No relevant chunks found"

The system couldn't find matching content.

**Solutions:**
- Rephrase the query
- Use more standard terminology
- Check if the topic is covered in EIPs

### Low similarity scores

Retrieved chunks may not be relevant.

**Solutions:**
- Increase `--top-k` to retrieve more chunks
- Use EIP numbers explicitly
- Ask more specific questions

### Validation shows many unsupported claims

The answer may include information not in the retrieved context.

**Solutions:**
- Increase `--top-k` for more context
- Ask more focused questions
- Use cited mode to see what sources are available

### Slow response time

First validated query downloads the NLI model (~1.5GB).

**Solutions:**
- Use `--mode simple` for faster responses
- The model is cached after first use
- Use `--local` for offline operation

## Example queries

### Understanding EIPs

```bash
# What is an EIP?
uv run python scripts/query_cli.py "What is EIP-1559 and why was it introduced?"

# Technical details
uv run python scripts/query_cli.py "What are the exact gas costs defined in EIP-2929?"

# Implementation status
uv run python scripts/query_cli.py "What is the status of EIP-4844?"
```

### Comparing features

```bash
# Fee mechanisms
uv run python scripts/query_cli.py "How does EIP-1559 fee mechanism differ from first-price auctions?"

# Storage opcodes
uv run python scripts/query_cli.py "What's the difference between SSTORE and TSTORE?"
```

### Historical context

```bash
# Evolution
uv run python scripts/query_cli.py "How has Ethereum's fee market evolved?"

# Fork history
uv run python scripts/query_cli.py "What EIPs were included in the London hard fork?"
```

### Technical specifications

```bash
# Parameters
uv run python scripts/query_cli.py "What is the BASE_FEE_MAX_CHANGE_DENOMINATOR in EIP-1559?"

# Opcodes
uv run python scripts/query_cli.py "What is the specification for the MCOPY opcode?"
```
