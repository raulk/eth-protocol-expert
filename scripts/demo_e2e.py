"""End-to-end demo of the Ethereum Protocol Intelligence System.

Demonstrates all phases:
- Phase 0/1: Basic vector retrieval with citations
- Phase 2: NLI validation
- Phase 3: Hybrid search (BM25 + vectors with RRF)
- Phase 4: EIP Graph traversal
- Phase 5: Query decomposition for complex queries
"""

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")


async def demo_vector_search():
    """Phase 0/1: Simple vector search."""
    from src.embeddings import create_embedder
    from src.retrieval.simple_retriever import SimpleRetriever
    from src.storage.pg_vector_store import PgVectorStore

    print("\n" + "=" * 60)
    print("PHASE 0/1: Vector Search Demo")
    print("=" * 60)

    store = PgVectorStore()
    await store.connect()
    try:
        embedder = create_embedder()
        retriever = SimpleRetriever(embedder, store)

        query = "What is EIP-1559 and how does the base fee work?"
        print(f"\nQuery: {query}")

        results = await retriever.retrieve(query, top_k=3)

        print(f"\nTop {len(results.results)} results:")
        for i, result in enumerate(results.results, 1):
            print(f"\n{i}. [{result.chunk.document_id}] (score: {result.similarity:.4f})")
            print(f"   Section: {result.chunk.section_path or 'N/A'}")
            content_preview = result.chunk.content[:200].replace('\n', ' ')
            print(f"   Content: {content_preview}...")
    finally:
        await store.close()


async def demo_hybrid_search():
    """Phase 3: Hybrid search with BM25 + vectors."""
    from src.embeddings import create_embedder
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.storage.pg_vector_store import PgVectorStore

    print("\n" + "=" * 60)
    print("PHASE 3: Hybrid Search Demo (BM25 + Vectors + RRF)")
    print("=" * 60)

    store = PgVectorStore()
    await store.connect()
    try:
        embedder = create_embedder()
        retriever = HybridRetriever(embedder, store)

        query = "SELFDESTRUCT opcode deprecation"
        print(f"\nQuery: {query}")

        results = await retriever.retrieve(query, limit=3)

        print(f"\nTop {len(results.results)} results (RRF fusion):")
        for i, result in enumerate(results.results, 1):
            print(f"\n{i}. [{result.chunk.document_id}] (RRF score: {result.rrf_score:.4f})")
            print(f"   Vector rank: {result.vector_rank}, BM25 rank: {result.bm25_rank}")
            content_preview = result.chunk.content[:200].replace('\n', ' ')
            print(f"   Content: {content_preview}...")
    finally:
        await store.close()


async def demo_metadata_filter():
    """Phase 3: Metadata filtering."""
    from src.embeddings import create_embedder
    from src.filters.metadata_filter import MetadataQuery
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.storage.pg_vector_store import PgVectorStore

    print("\n" + "=" * 60)
    print("PHASE 3: Metadata Filtering Demo")
    print("=" * 60)

    store = PgVectorStore()
    await store.connect()
    try:
        embedder = create_embedder()
        retriever = HybridRetriever(embedder, store)

        filter_query = MetadataQuery(
            statuses=["Final"],
            categories=["Core"]
        )

        query = "gas pricing mechanism"
        print(f"\nQuery: {query}")
        print("Filters: status=Final, category=Core")

        results = await retriever.retrieve(
            query,
            limit=3,
            metadata_query=filter_query
        )

        print(f"\nTop {len(results.results)} filtered results:")
        for i, result in enumerate(results.results, 1):
            print(f"\n{i}. [{result.chunk.document_id}] (RRF: {result.rrf_score:.4f})")
            content_preview = result.chunk.content[:150].replace('\n', ' ')
            print(f"   Content: {content_preview}...")
    finally:
        await store.close()


async def build_eip_graph():
    """Build the EIP dependency graph."""
    import asyncpg

    from src.graph.falkordb_store import FalkorDBStore

    print("\n" + "=" * 60)
    print("Building EIP Dependency Graph...")
    print("=" * 60)

    conn = await asyncpg.connect(os.environ.get("DATABASE_URL"))
    try:
        rows = await conn.fetch("""
            SELECT document_id, eip_number, title, status, type, category, requires
            FROM documents
            WHERE eip_number IS NOT NULL
        """)
    finally:
        await conn.close()

    print(f"Found {len(rows)} EIPs to process")

    graph_store = FalkorDBStore()
    graph_store.initialize_schema()

    print("Creating EIP nodes...")
    nodes_created = 0
    for row in rows:
        try:
            result = graph_store.create_eip_node(
                number=row['eip_number'],
                title=row['title'] or "",
                status=row['status'] or "Draft",
                type_=row['type'] or "Standards Track",
                category=row['category']
            )
            nodes_created += result.nodes_created
        except Exception:
            pass

    print(f"Created {nodes_created} nodes")

    print("Creating REQUIRES relationships...")
    rels_created = 0
    for row in rows:
        requires = row['requires']
        if requires:
            for req_eip in requires:
                try:
                    result = graph_store.create_requires_relationship(row['eip_number'], req_eip)
                    rels_created += result.relationships_created
                except Exception:
                    pass

    print(f"Created {rels_created} REQUIRES relationships")
    print("Graph built successfully!")

    return graph_store


def demo_graph_traversal(graph_store=None):
    """Phase 4: EIP graph traversal."""
    from src.graph.dependency_traverser import DependencyTraverser, TraversalDirection
    from src.graph.falkordb_store import FalkorDBStore

    print("\n" + "=" * 60)
    print("PHASE 4: Graph Traversal Demo")
    print("=" * 60)

    if graph_store is None:
        graph_store = FalkorDBStore()

    traverser = DependencyTraverser(graph_store)

    eip_number = 4844
    print(f"\nDependencies of EIP-{eip_number}:")

    deps = traverser.get_dependencies(eip_number, TraversalDirection.UPSTREAM, max_depth=2)
    if deps.all_dependencies:
        for dep in deps.all_dependencies[:5]:
            print(f"  - EIP-{dep.eip_number}: {dep.title or 'N/A'} (depth: {dep.depth})")
    else:
        print("  (No upstream dependencies)")

    eip_number = 1559
    print(f"\nEIPs that depend on EIP-{eip_number}:")

    deps = traverser.get_dependencies(eip_number, TraversalDirection.DOWNSTREAM, max_depth=1)
    if deps.all_dependents:
        for dep in deps.all_dependents[:5]:
            print(f"  - EIP-{dep.eip_number}: {dep.title or 'N/A'}")
    else:
        print("  (No downstream dependents found)")

    print("\nMost depended-upon EIPs:")
    top_deps = traverser.get_most_depended_upon(limit=5)
    for eip, count in top_deps:
        print(f"  - EIP-{eip}: {count} dependents")


async def demo_query_decomposition():
    """Phase 5: Query decomposition for complex queries."""
    from src.routing.query_classifier import QueryClassifier

    print("\n" + "=" * 60)
    print("PHASE 5: Query Decomposition Demo")
    print("=" * 60)

    classifier = QueryClassifier(use_llm=False)

    queries = [
        "What is EIP-4844?",
        "Compare the gas models of EIP-1559 and EIP-4844",
        "How did account abstraction evolve from EIP-86 to EIP-4337?",
    ]

    for query in queries:
        print(f"\nQuery: {query}")

        result = await classifier.classify(query)
        print(f"  Type: {result.query_type.value}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Needs decomposition: {result.needs_decomposition}")
        print(f"  Estimated sub-questions: {result.estimated_sub_questions}")
        print(f"  Reasoning: {result.reasoning}")


async def demo_full_rag_pipeline():
    """Full RAG pipeline with generation."""
    from src.embeddings import create_embedder
    from src.generation.cited_generator import CitedGenerator
    from src.retrieval.simple_retriever import SimpleRetriever
    from src.storage.pg_vector_store import PgVectorStore

    print("\n" + "=" * 60)
    print("FULL RAG PIPELINE: Query -> Retrieve -> Generate")
    print("=" * 60)

    store = PgVectorStore()
    await store.connect()
    try:
        embedder = create_embedder()
        retriever = SimpleRetriever(embedder, store)
        generator = CitedGenerator(retriever)

        query = "What is EIP-1559 and how does it change Ethereum's fee market?"
        print(f"\nQuery: {query}")

        print("\nRetrieving relevant context and generating answer...")
        result = await generator.generate(query, top_k=5)

        print(f"\nRetrieved {len(result.retrieval.results)} chunks")
        print(f"Generated {len(result.evidence_ledger.claims)} claims")

        print("\n" + "-" * 40)
        print("ANSWER WITH CITATIONS:")
        print("-" * 40)
        print(result.response_with_citations)

        print("\n" + "-" * 40)
        print("EVIDENCE COVERAGE:")
        print("-" * 40)
        coverage_ratio = result.evidence_ledger.compute_coverage()
        total_claims = len(result.evidence_ledger.claims)
        supported_claims = sum(
            1 for c in result.evidence_ledger.claims
            if result.evidence_ledger.evidence_map.get(c.claim_id)
        )
        print(f"  Total claims: {total_claims}")
        print(f"  Supported claims: {supported_claims}")
        print(f"  Unsupported claims: {total_claims - supported_claims}")
        print(f"  Coverage ratio: {coverage_ratio:.1%}")
    finally:
        await store.close()


async def main():
    """Run all demos."""
    print("\n" + "#" * 60)
    print("# ETHEREUM PROTOCOL INTELLIGENCE SYSTEM - E2E DEMO")
    print("#" * 60)

    # Phase 0/1: Basic vector search
    await demo_vector_search()

    # Phase 3: Hybrid search
    await demo_hybrid_search()

    # Phase 3: Metadata filtering
    await demo_metadata_filter()

    # Phase 4: Build graph
    graph_store = await build_eip_graph()

    # Phase 4: Graph traversal
    demo_graph_traversal(graph_store)

    # Phase 5: Query decomposition
    await demo_query_decomposition()

    # Full RAG pipeline
    await demo_full_rag_pipeline()

    print("\n" + "#" * 60)
    print("# DEMO COMPLETE")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
