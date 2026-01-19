"""End-to-end test script for Ethereum Protocol Intelligence System."""

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

password = os.environ.get("POSTGRES_PASSWORD")
db_url = f"postgresql://postgres:{password}@localhost:5432/eth_protocol"


async def test_end_to_end():
    print("=" * 60)
    print("END-TO-END TEST: Database + Embeddings + Search")
    print("=" * 60)

    from src.chunking import FixedChunker
    from src.embeddings import VoyageEmbedder
    from src.ingestion import EIPParser
    from src.retrieval import SimpleRetriever
    from src.storage import PgVectorStore

    # 1. Connect to database
    print("\n1. Connecting to PostgreSQL...")
    store = PgVectorStore(db_url)
    await store.connect()
    print("   Connected!")

    # 2. Initialize schema
    print("\n2. Initializing schema...")
    await store.initialize_schema()
    print("   Schema ready!")

    # 3. Create embedder
    print("\n3. Creating embedder...")
    embedder = VoyageEmbedder()
    print("   Embedder ready!")

    # 4. Create test content
    print("\n4. Creating test EIP content...")
    from datetime import datetime
    from pathlib import Path

    from src.ingestion.eip_loader import LoadedEIP

    test_eip_content = """---
eip: 1559
title: Fee Market Change
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
with the highest bids, and transactions included in the block pay the bid that they
specified.

## Specification

As of FORK_BLOCK_NUMBER, a new EIP-2718 transaction is introduced with TransactionType 2.
The intrinsic cost of the new transaction is inherited from EIP-2930.
"""

    # 5. Parse and chunk the content
    print("\n5. Parsing and chunking content...")
    loaded_eip = LoadedEIP(
        eip_number=1559,
        file_path=Path("eip-1559.md"),
        raw_content=test_eip_content,
        git_commit="test",
        loaded_at=datetime.now(),
    )
    parser = EIPParser()
    parsed_eip = parser.parse(loaded_eip)
    print(f"   Parsed EIP-{parsed_eip.eip_number}: {parsed_eip.title}")

    chunker = FixedChunker(max_tokens=200)
    chunks = chunker.chunk_eip(parsed_eip)
    print(f"   Created {len(chunks)} chunks")

    # 6. Embed chunks
    print("\n6. Embedding chunks...")
    embedded_chunks = embedder.embed_chunks(chunks)
    print(f"   Embedded {len(embedded_chunks)} chunks")

    # 7. Store in database
    print("\n7. Storing in database...")
    await store.store_embedded_chunks(embedded_chunks)
    print("   Stored!")

    # 8. Count what's stored
    doc_count = await store.count_documents()
    chunk_count = await store.count_chunks()
    print(f"   Documents: {doc_count}, Chunks: {chunk_count}")

    # 9. Create retriever and search
    print('\n8. Searching for "transaction fees"...')
    retriever = SimpleRetriever(embedder, store)
    results = await retriever.retrieve("transaction fees", top_k=3)
    print(f"   Found {len(results.results)} results")

    for i, r in enumerate(results.results):
        print(f"   [{i+1}] Score: {r.similarity:.3f} - {r.chunk.content[:50]}...")

    # 10. Test Phase 7: Concept Resolution
    print("\n9. Testing Phase 7 (Concept Resolution)...")
    from src.concepts import AliasTable, QueryExpander

    alias_table = AliasTable()
    expander = QueryExpander(alias_table)
    expanded = expander.expand("What is EIP-1559 base fee?")
    print(f"   Canonical terms: {expanded.canonical_terms}")
    print(f"   Expanded terms: {expanded.expanded_terms[:3]}")

    # 11. Test Phase 12: Cost Router
    print("\n10. Testing Phase 12 (Cost Router)...")
    from src.ensemble import CostRouter

    router = CostRouter()
    decision = router.route_sync("What is EIP-1559?")
    print(f"   Query routed to: {decision.model_tier.value} ({decision.model_id})")

    # Cleanup
    await store.close()

    print("\n" + "=" * 60)
    print("ALL END-TO-END TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_end_to_end())
