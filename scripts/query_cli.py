#!/usr/bin/env python3
"""CLI for querying the Ethereum Protocol Intelligence System.

Usage:
    python scripts/query_cli.py "What is EIP-4844?"
    python scripts/query_cli.py "What is EIP-4844?" --mode simple
    python scripts/query_cli.py "What is EIP-4844?" --mode validated
    python scripts/query_cli.py "How does EIP-1559 interact with EIP-4844?" --mode agentic
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog
from dotenv import load_dotenv
load_dotenv()  # Must run before any src.* imports

from src.embeddings import create_embedder
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


async def run_agentic_query(
    question: str,
    store: PgVectorStore,
    embedder,
    max_retrievals: int = 5,
    show_reasoning: bool = True,
):
    """Run an agentic query with ReAct loop."""
    from src.agents import AgentBudget, ReactAgent, RetrievalTool

    retriever = SimpleRetriever(embedder=embedder, store=store)
    tool = RetrievalTool(simple_retriever=retriever, default_limit=5)

    budget = AgentBudget(
        max_retrievals=max_retrievals,
        max_tokens=10000,
        max_llm_calls=max_retrievals * 2 + 2,
    )

    agent = ReactAgent(
        retrieval_tool=tool,
        budget=budget,
        enable_reflection=True,
        enable_backtracking=True,
    )

    result = await agent.run(question)

    print("\n" + "=" * 60)
    print("RESPONSE (Agentic Mode - Phase 8)")
    print("=" * 60)
    print(result.answer)

    if show_reasoning:
        print("\n" + "-" * 60)
        print("REASONING CHAIN:")
        print("-" * 60)
        for i, thought in enumerate(result.thoughts):
            action_str = thought.action.value.upper()
            print(f"  {i + 1}. [{action_str}] {thought.content[:100]}...")
            if thought.action_input:
                print(f"     → {thought.action_input[:80]}...")

    print("\n" + "-" * 60)
    print("AGENT STATS:")
    print("-" * 60)
    print(f"  Retrievals: {result.retrieval_count}")
    print(f"  LLM calls: {result.llm_calls}")
    print(f"  Tokens retrieved: {result.total_tokens_retrieved}")
    print(f"  Termination: {result.termination_reason}")
    print("-" * 60)

    return result


async def query(
    question: str,
    mode: str = "cited",
    top_k: int = 10,
    show_sources: bool = True,
    max_retrievals: int = 5,
    show_reasoning: bool = True,
):
    """Run a query against the system."""

    # Initialize components
    store = PgVectorStore()
    await store.connect()

    try:
        embedder = create_embedder()

        if mode == "agentic":
            await run_agentic_query(
                question=question,
                store=store,
                embedder=embedder,
                max_retrievals=max_retrievals,
                show_reasoning=show_reasoning,
            )
            return

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
                    sections = (
                        ", ".join(source["sections"]) if source["sections"] else "Full document"
                    )
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
        choices=["simple", "cited", "validated", "agentic"],
        default="cited",
        help="Generation mode (default: cited)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of chunks to retrieve (for non-agentic modes)",
    )
    parser.add_argument(
        "--no-sources",
        action="store_true",
        help="Don't show sources",
    )
    parser.add_argument(
        "--max-retrievals",
        type=int,
        default=5,
        help="Maximum retrieval attempts for agentic mode (default: 5)",
    )
    parser.add_argument(
        "--no-reasoning",
        action="store_true",
        help="Don't show reasoning chain for agentic mode",
    )

    args = parser.parse_args()

    asyncio.run(
        query(
            question=args.question,
            mode=args.mode,
            top_k=args.top_k,
            show_sources=not args.no_sources,
            max_retrievals=args.max_retrievals,
            show_reasoning=not args.no_reasoning,
        )
    )


if __name__ == "__main__":
    main()
