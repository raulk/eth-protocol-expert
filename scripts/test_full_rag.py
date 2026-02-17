"""Full RAG pipeline test with LLM synthesis."""

import asyncio
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

password = os.environ.get("POSTGRES_PASSWORD")
db_url = f"postgresql://postgres:{password}@localhost:5432/eth_protocol"


async def test_full_rag():
    print("=" * 60)
    print("FULL RAG PIPELINE TEST")
    print("=" * 60)

    from src.chunking import FixedChunker
    from src.embeddings import create_embedder
    from src.generation import SimpleGenerator
    from src.ingestion import EIPParser
    from src.ingestion.eip_loader import LoadedEIP
    from src.retrieval import SimpleRetriever
    from src.storage import PgVectorStore

    # Connect to database
    print("\n1. Setting up infrastructure...")
    store = PgVectorStore(db_url)
    await store.connect()
    await store.initialize_schema()
    embedder = create_embedder()
    print("   Infrastructure ready!")

    # Load multiple EIPs for richer context
    print("\n2. Loading test EIPs...")

    eips = [
        {
            "number": 1559,
            "content": """---
eip: 1559
title: Fee market change for ETH 1.0 chain
status: Final
type: Standards Track
category: Core
created: 2019-04-13
---

## Abstract

This EIP introduces a transaction pricing mechanism that includes fixed-per-block
network fee that is burned and dynamically expands/contracts block sizes to deal
with transient congestion.

## Motivation

Ethereum historically priced transaction fees using a simple auction mechanism,
where users send transactions with bids ("gasprices") and miners choose transactions
with the highest bids.

The problems with first-price auctions:
1. Mismatch between volatility of transaction fee levels and social cost of transactions
2. Needless delays for users
3. Inefficiencies of first price auctions
4. Instability of blockchains with no block reward

## Specification

As of FORK_BLOCK_NUMBER, a new EIP-2718 transaction is introduced with TransactionType 2.

Parameters:
- INITIAL_BASE_FEE: 1000000000 (1 gwei)
- BASE_FEE_MAX_CHANGE_DENOMINATOR: 8
- ELASTICITY_MULTIPLIER: 2
""",
        },
        {
            "number": 4844,
            "content": """---
eip: 4844
title: Shard Blob Transactions
status: Final
type: Standards Track
category: Core
created: 2022-02-25
---

## Abstract

Introduce a new transaction format for "blob-carrying transactions" which contain
a large amount of data that cannot be accessed by EVM execution, but whose commitment
can be accessed. Proto-danksharding (EIP-4844) is a step toward full danksharding.

## Motivation

Rollups are the dominant scaling solution for Ethereum. They post transaction data
to Ethereum as calldata, which is expensive. This EIP introduces blob-carrying
transactions, where the blob data is stored separately and priced using a separate
fee market.

Key benefits:
1. Reduced cost for rollup data
2. Separate fee market for blob data
3. Path to full danksharding

## Specification

We introduce a new transaction type with two components:
- A transaction payload similar to EIP-2930 transactions
- A list of "blobs" of data (each blob is 4096 field elements, ~125 KB)

MAX_BLOB_GAS_PER_BLOCK: 786432
TARGET_BLOB_GAS_PER_BLOCK: 393216
BLOB_GASPRICE_UPDATE_FRACTION: 3338477
""",
        },
    ]

    parser = EIPParser()
    chunker = FixedChunker(max_tokens=300)

    for eip in eips:
        loaded = LoadedEIP(
            eip_number=eip["number"],
            file_path=Path(f"eip-{eip['number']}.md"),
            raw_content=eip["content"],
            git_commit="test",
            loaded_at=datetime.now(),
        )
        parsed = parser.parse(loaded)
        chunks = chunker.chunk_eip(parsed)
        embedded = embedder.embed_chunks(chunks)
        await store.store_embedded_chunks(embedded)
        print(f"   Loaded EIP-{eip['number']}: {len(chunks)} chunks")

    # Test retrieval and generation
    print("\n3. Testing RAG pipeline...")
    retriever = SimpleRetriever(embedder, store)
    generator = SimpleGenerator(retriever)

    queries = [
        "What is the base fee in EIP-1559?",
        "How do blob transactions work in EIP-4844?",
    ]

    for query in queries:
        print(f"\n   Q: {query}")

        # Generate answer (retrieves and generates)
        result = await generator.generate(query)
        print(f"\n   Answer:\n   {result.response}")
        print(f"\n   Sources: {[s.chunk.document_id for s in result.retrieval.results[:2]]}")
        print("-" * 60)

    # Clean up
    await store.clear_all()
    await store.close()

    print("\n" + "=" * 60)
    print("FULL RAG PIPELINE TEST PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_full_rag())
