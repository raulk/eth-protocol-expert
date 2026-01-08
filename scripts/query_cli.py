#!/usr/bin/env python3
"""CLI for querying the Ethereum Protocol Intelligence System.

Usage:
    python scripts/query_cli.py "What is EIP-4844?"
    python scripts/query_cli.py "What is EIP-4844?" --mode simple
    python scripts/query_cli.py "What is EIP-4844?" --mode validated
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog
from dotenv import load_dotenv

from src.embeddings.local_embedder import LocalEmbedder
from src.embeddings.voyage_embedder import VoyageEmbedder
from src.generation.cited_generator import CitedGenerator
from src.generation.simple_generator import SimpleGenerator
from src.retrieval.simple_retriever import SimpleRetriever
from src.storage.pg_vector_store import PgVectorStore

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ]
)
logger = structlog.get_logger()


async def query(
    question: str,
    mode: str = "cited",
    top_k: int = 10,
    show_sources: bool = True,
    use_local_embeddings: bool = False,
):
    """Run a query against the system."""
    load_dotenv()

    # Initialize components
    store = PgVectorStore()
    await store.connect()

    try:
        if use_local_embeddings:
            embedder = LocalEmbedder()
        else:
            embedder = VoyageEmbedder()
        retriever = SimpleRetriever(embedder=embedder, store=store)

        if mode == "simple":
            generator = SimpleGenerator(retriever=retriever)
            result = await generator.generate(query=question, top_k=top_k)

            print("\n" + "=" * 60)
            print("RESPONSE (Simple Mode - Phase 0)")
            print("=" * 60)
            print(result.response)

        elif mode == "cited":
            generator = CitedGenerator(retriever=retriever)
            result = await generator.generate(query=question, top_k=top_k)

            print("\n" + "=" * 60)
            print("RESPONSE (Cited Mode - Phase 1)")
            print("=" * 60)
            print(result.response_with_citations)

            if show_sources:
                print("\n" + "-" * 60)
                print("SOURCES:")
                print("-" * 60)
                for source in result.evidence_ledger.get_all_sources():
                    sections = ", ".join(source["sections"]) if source["sections"] else "Full document"
                    print(f"  - {source['document_id'].upper()}: {sections}")

        elif mode == "validated":
            # Import validated generator only when needed (requires torch)
            from src.generation.validated_generator import ValidatedGenerator

            generator = ValidatedGenerator(retriever=retriever)
            result = await generator.generate(query=question, top_k=top_k)

            print("\n" + "=" * 60)
            print("RESPONSE (Validated Mode - Phase 2)")
            print("=" * 60)
            print(result.validated_response)

            print("\n" + "-" * 60)
            print("VALIDATION SUMMARY:")
            print("-" * 60)
            print(f"  Total claims: {result.total_claims}")
            print(f"  Supported: {result.supported_claims}")
            print(f"  Weak: {result.weak_claims}")
            print(f"  Unsupported: {result.unsupported_claims}")
            print(f"  Support ratio: {result.support_ratio:.1%}")
            print(f"  Trustworthy: {'Yes' if result.is_trustworthy else 'No'}")

        else:
            print(f"Unknown mode: {mode}")
            return

        # Show token usage
        print("\n" + "-" * 60)
        print(f"Tokens: {result.input_tokens} input, {result.output_tokens} output")
        print("-" * 60)

    finally:
        await store.close()


def main():
    parser = argparse.ArgumentParser(description="Query the Ethereum Protocol Intelligence System")
    parser.add_argument("question", help="The question to ask")
    parser.add_argument(
        "--mode",
        choices=["simple", "cited", "validated"],
        default="cited",
        help="Generation mode",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of chunks to retrieve",
    )
    parser.add_argument(
        "--no-sources",
        action="store_true",
        help="Don't show sources",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local BGE embeddings instead of Voyage API",
    )

    args = parser.parse_args()

    asyncio.run(query(
        question=args.question,
        mode=args.mode,
        top_k=args.top_k,
        show_sources=not args.no_sources,
        use_local_embeddings=args.local,
    ))


if __name__ == "__main__":
    main()
