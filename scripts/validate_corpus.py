#!/usr/bin/env python3
"""Validate full corpus and test retrieval quality.

Usage:
    uv run python scripts/validate_corpus.py
"""

import asyncio
import os

import structlog
from dotenv import load_dotenv
load_dotenv()  # Must run before any src.* imports

from src.embeddings import create_embedder
from src.generation import CitedGenerator
from src.retrieval import SimpleRetriever
from src.storage import PgVectorStore

logger = structlog.get_logger()


TEST_QUERIES = [
    "What are the different approaches that have been suggested over time for encrypted mempools?",
    "What are the upstream dependencies of ZK EVMs?",
    "Why was RISC-V chosen over Wasm?",
]


async def validate_corpus() -> None:
    """Validate corpus ingestion and query capabilities."""
    store = PgVectorStore()
    await store.connect()

    embedder = create_embedder()
    retriever = SimpleRetriever(embedder, store)

    print("\n" + "=" * 60)
    print("CORPUS VALIDATION")
    print("=" * 60)

    # 1. Get document stats by document_type
    print("\n1. Document counts by type:")
    async with store.connection() as conn:
        rows = await conn.fetch("""
            SELECT document_type, COUNT(*) as count
            FROM documents
            GROUP BY document_type
            ORDER BY count DESC
        """)
        if rows:
            for row in rows:
                doc_type = row["document_type"] or "unknown"
                print(f"   {doc_type}: {row['count']}")
        else:
            print("   No documents found")

    # 2. Get chunk counts
    print("\n2. Chunk counts:")
    async with store.connection() as conn:
        row = await conn.fetchrow("SELECT COUNT(*) as count FROM chunks")
        print(f"   Total chunks: {row['count']}")

    # 3. Test retrieval for each query
    print("\n3. Retrieval test results:")
    for query in TEST_QUERIES:
        print(f"\n   Query: {query[:60]}...")
        results = await retriever.retrieve(query, top_k=5)

        # Collect unique document types from results
        sources = set()
        for r in results.results:
            doc_id = r.chunk.document_id
            # Extract source type from document_id prefix
            if "-" in doc_id:
                source_type = doc_id.split("-")[0]
            else:
                source_type = "eip"
            sources.add(source_type)

        print(f"   Sources found: {sources}")
        if results.results:
            print(f"   Top result document_id: {results.results[0].chunk.document_id}")
        else:
            print("   Top result: None")

    # 4. Test generation (if API key available)
    print("\n4. Generation test:")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        try:
            generator = CitedGenerator(retriever=retriever, api_key=api_key)
            test_q = TEST_QUERIES[0]
            result = await generator.generate(test_q)
            print(f"   Query: {test_q[:50]}...")
            print(f"   Response length: {len(result.response)} chars")
            print(f"   Sources cited: {len(result.retrieval.results)}")
            print(f"   First 200 chars: {result.response[:200]}...")
        except Exception as e:
            print(f"   Generation test failed: {e}")
    else:
        print("   Generation test skipped: ANTHROPIC_API_KEY not set")

    await store.close()

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60 + "\n")


def main() -> None:
    asyncio.run(validate_corpus())


if __name__ == "__main__":
    main()
