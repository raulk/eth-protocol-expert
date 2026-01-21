#!/usr/bin/env python3
"""Evaluation runner for the Ethereum Protocol Intelligence System.

USAGE:
    python tests/eval/run_eval.py                           # Run all tests
    python tests/eval/run_eval.py --phase 0                 # Run phase 0 tests only
    python tests/eval/run_eval.py --mode cited              # Use cited generator
    python tests/eval/run_eval.py --mode validated          # Use validated generator
    python tests/eval/run_eval.py --limit 10                # Run first 10 tests only
    python tests/eval/run_eval.py --verbose                 # Show detailed output

Metrics are saved to data/eval/metrics.json with a timestamp.
"""

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import structlog
from dotenv import load_dotenv

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


@dataclass
class QAPair:
    """A question-answer pair from the evaluation dataset."""
    question: str
    expected_answer: str
    phase: int | None = None
    category: str | None = None
    expected_sources: list[str] | None = None
    metadata: dict | None = None


@dataclass
class EvalResult:
    """Result from evaluating a single QA pair."""
    question: str
    expected_answer: str
    generated_answer: str
    retrieval_count: int
    retrieved_sources: list[str]
    latency_ms: float
    mode: str

    # Metrics
    answer_similarity: float = 0.0
    retrieval_recall: float = 0.0
    citation_accuracy: float = 0.0

    # Metadata
    phase: int | None = None
    category: str | None = None
    expected_sources: list[str] | None = None
    error: str | None = None


@dataclass
class EvalSummary:
    """Summary metrics from evaluation run."""
    timestamp: str
    mode: str
    total_questions: int
    successful: int
    failed: int

    # Aggregate metrics
    avg_answer_similarity: float
    avg_retrieval_recall: float
    avg_citation_accuracy: float
    avg_latency_ms: float

    # Per-phase metrics
    phase_metrics: dict[str, dict]

    # Individual results
    results: list[dict]


def load_qa_pairs(filepath: Path) -> list[QAPair]:
    """Load QA pairs from a JSONL file."""
    pairs = []
    if not filepath.exists():
        logger.warning("qa_file_not_found", path=str(filepath))
        return pairs

    with filepath.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            pairs.append(QAPair(
                question=data["question"],
                expected_answer=data["expected_answer"],
                phase=data.get("phase"),
                category=data.get("category"),
                expected_sources=data.get("expected_sources"),
                metadata=data.get("metadata"),
            ))

    return pairs


def compute_answer_similarity(expected: str, generated: str) -> float:
    """Compute semantic similarity between expected and generated answers.

    Uses word overlap as a simple baseline. For production, consider using
    embeddings or NLI-based similarity.
    """
    expected_words = set(expected.lower().split())
    generated_words = set(generated.lower().split())

    if not expected_words:
        return 1.0 if not generated_words else 0.0

    intersection = expected_words & generated_words

    precision = len(intersection) / len(generated_words) if generated_words else 0
    recall = len(intersection) / len(expected_words) if expected_words else 0

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return f1


def compute_retrieval_recall(
    expected_sources: list[str] | None,
    retrieved_sources: list[str],
) -> float:
    """Compute recall@k for retrieval.

    Measures what fraction of expected sources were retrieved.
    """
    if not expected_sources:
        return 1.0

    expected_set = {s.lower() for s in expected_sources}
    retrieved_set = {s.lower() for s in retrieved_sources}

    found = expected_set & retrieved_set
    return len(found) / len(expected_set)


def compute_citation_accuracy(
    generated_answer: str,
    retrieved_sources: list[str],
) -> float:
    """Compute citation accuracy for cited mode.

    Measures whether citations in the response reference actual retrieved sources.
    """
    import re

    citation_pattern = re.compile(r'\[(\d+)\]')
    citations = citation_pattern.findall(generated_answer)

    if not citations:
        return 1.0

    valid_citations = 0
    for citation in citations:
        idx = int(citation) - 1
        if 0 <= idx < len(retrieved_sources):
            valid_citations += 1

    return valid_citations / len(citations)


async def evaluate_single(
    qa_pair: QAPair,
    retriever: SimpleRetriever,
    generator: SimpleGenerator | CitedGenerator,
    mode: str,
    top_k: int = 10,
) -> EvalResult:
    """Evaluate a single QA pair."""
    start_time = time.perf_counter()

    try:
        result = await generator.generate(query=qa_pair.question, top_k=top_k)
        latency_ms = (time.perf_counter() - start_time) * 1000

        if mode == "cited":
            generated_answer = result.response_with_citations
            retrieval = result.retrieval
        else:
            generated_answer = result.response
            retrieval = result.retrieval

        retrieved_sources = [
            r.chunk.document_id for r in retrieval.results
        ]

        answer_similarity = compute_answer_similarity(
            qa_pair.expected_answer,
            generated_answer,
        )

        retrieval_recall = compute_retrieval_recall(
            qa_pair.expected_sources,
            retrieved_sources,
        )

        citation_accuracy = compute_citation_accuracy(
            generated_answer,
            retrieved_sources,
        ) if mode == "cited" else 0.0

        return EvalResult(
            question=qa_pair.question,
            expected_answer=qa_pair.expected_answer,
            generated_answer=generated_answer,
            retrieval_count=len(retrieval.results),
            retrieved_sources=retrieved_sources,
            latency_ms=latency_ms,
            mode=mode,
            answer_similarity=answer_similarity,
            retrieval_recall=retrieval_recall,
            citation_accuracy=citation_accuracy,
            phase=qa_pair.phase,
            category=qa_pair.category,
            expected_sources=qa_pair.expected_sources,
        )

    except Exception as e:
        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.error("evaluation_error", question=qa_pair.question[:50], error=str(e))
        return EvalResult(
            question=qa_pair.question,
            expected_answer=qa_pair.expected_answer,
            generated_answer="",
            retrieval_count=0,
            retrieved_sources=[],
            latency_ms=latency_ms,
            mode=mode,
            phase=qa_pair.phase,
            category=qa_pair.category,
            expected_sources=qa_pair.expected_sources,
            error=str(e),
        )


async def run_evaluation(
    qa_pairs: list[QAPair],
    mode: str = "simple",
    top_k: int = 10,
    verbose: bool = False,
) -> EvalSummary:
    """Run evaluation on all QA pairs."""
    load_dotenv()

    store = PgVectorStore()
    await store.connect()

    try:
        embedder = VoyageEmbedder()
        retriever = SimpleRetriever(embedder=embedder, store=store)

        if mode == "cited":
            generator = CitedGenerator(retriever=retriever)
        elif mode == "validated":
            from src.generation.validated_generator import ValidatedGenerator
            generator = ValidatedGenerator(retriever=retriever)
        else:
            generator = SimpleGenerator(retriever=retriever)

        results: list[EvalResult] = []

        for i, qa_pair in enumerate(qa_pairs):
            if verbose:
                print(f"\n[{i + 1}/{len(qa_pairs)}] {qa_pair.question[:60]}...")

            result = await evaluate_single(
                qa_pair=qa_pair,
                retriever=retriever,
                generator=generator,
                mode=mode,
                top_k=top_k,
            )
            results.append(result)

            if verbose:
                status = "ERROR" if result.error else "OK"
                print(f"  -> {status} | similarity={result.answer_similarity:.2f} "
                      f"recall={result.retrieval_recall:.2f} latency={result.latency_ms:.0f}ms")

        successful = [r for r in results if not r.error]
        failed = [r for r in results if r.error]

        avg_similarity = (
            sum(r.answer_similarity for r in successful) / len(successful)
            if successful else 0.0
        )
        avg_recall = (
            sum(r.retrieval_recall for r in successful) / len(successful)
            if successful else 0.0
        )
        avg_citation = (
            sum(r.citation_accuracy for r in successful) / len(successful)
            if successful and mode == "cited" else 0.0
        )
        avg_latency = (
            sum(r.latency_ms for r in successful) / len(successful)
            if successful else 0.0
        )

        phase_metrics: dict[str, dict] = {}
        phases = {r.phase for r in results if r.phase is not None}
        for phase in phases:
            phase_results = [r for r in successful if r.phase == phase]
            if phase_results:
                phase_metrics[f"phase_{phase}"] = {
                    "count": len(phase_results),
                    "avg_answer_similarity": sum(r.answer_similarity for r in phase_results) / len(phase_results),
                    "avg_retrieval_recall": sum(r.retrieval_recall for r in phase_results) / len(phase_results),
                    "avg_latency_ms": sum(r.latency_ms for r in phase_results) / len(phase_results),
                }

        return EvalSummary(
            timestamp=datetime.now().isoformat(),
            mode=mode,
            total_questions=len(qa_pairs),
            successful=len(successful),
            failed=len(failed),
            avg_answer_similarity=avg_similarity,
            avg_retrieval_recall=avg_recall,
            avg_citation_accuracy=avg_citation,
            avg_latency_ms=avg_latency,
            phase_metrics=phase_metrics,
            results=[asdict(r) for r in results],
        )

    finally:
        await store.close()


def print_summary(summary: EvalSummary) -> None:
    """Print evaluation summary to stdout."""
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Timestamp: {summary.timestamp}")
    print(f"Mode: {summary.mode}")
    print(f"Total questions: {summary.total_questions}")
    print(f"Successful: {summary.successful}")
    print(f"Failed: {summary.failed}")
    print()
    print("AGGREGATE METRICS:")
    print(f"  Answer similarity (F1): {summary.avg_answer_similarity:.3f}")
    print(f"  Retrieval recall@10:    {summary.avg_retrieval_recall:.3f}")
    if summary.mode == "cited":
        print(f"  Citation accuracy:      {summary.avg_citation_accuracy:.3f}")
    print(f"  Average latency:        {summary.avg_latency_ms:.0f}ms")

    if summary.phase_metrics:
        print()
        print("PER-PHASE METRICS:")
        for phase_name, metrics in sorted(summary.phase_metrics.items()):
            print(f"  {phase_name}:")
            print(f"    Count: {metrics['count']}")
            print(f"    Similarity: {metrics['avg_answer_similarity']:.3f}")
            print(f"    Recall: {metrics['avg_retrieval_recall']:.3f}")
            print(f"    Latency: {metrics['avg_latency_ms']:.0f}ms")

    print("=" * 60)


def save_metrics(summary: EvalSummary, output_dir: Path) -> Path:
    """Save metrics to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"metrics_{timestamp_str}.json"

    with output_file.open("w") as f:
        json.dump(asdict(summary), f, indent=2)

    latest_file = output_dir / "metrics.json"
    with latest_file.open("w") as f:
        json.dump(asdict(summary), f, indent=2)

    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run evaluation on the Ethereum Protocol Intelligence System"
    )
    parser.add_argument(
        "--phase",
        type=int,
        default=None,
        help="Filter by phase (e.g., 0, 1, 2)",
    )
    parser.add_argument(
        "--mode",
        choices=["simple", "cited", "validated"],
        default="simple",
        help="Generation mode (default: simple)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of chunks to retrieve (default: 10)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of questions to evaluate",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed output for each question",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/eval",
        help="Output directory for metrics (default: data/eval)",
    )

    args = parser.parse_args()

    eval_dir = Path(__file__).parent
    qa_files = [
        eval_dir / "qa_v1.jsonl",
        eval_dir / "multihop_v1.jsonl",
    ]

    all_pairs: list[QAPair] = []
    for qa_file in qa_files:
        pairs = load_qa_pairs(qa_file)
        all_pairs.extend(pairs)
        if pairs:
            logger.info("loaded_qa_pairs", file=qa_file.name, count=len(pairs))

    if not all_pairs:
        print("No QA pairs found. Create tests/eval/qa_v1.jsonl or tests/eval/multihop_v1.jsonl")
        sys.exit(1)

    if args.phase is not None:
        all_pairs = [p for p in all_pairs if p.phase == args.phase]
        logger.info("filtered_by_phase", phase=args.phase, count=len(all_pairs))

    if args.limit:
        all_pairs = all_pairs[:args.limit]
        logger.info("limited_pairs", limit=args.limit)

    if not all_pairs:
        print("No QA pairs match the filter criteria")
        sys.exit(1)

    print(f"Running evaluation on {len(all_pairs)} questions (mode={args.mode})...")

    summary = asyncio.run(run_evaluation(
        qa_pairs=all_pairs,
        mode=args.mode,
        top_k=args.top_k,
        verbose=args.verbose,
    ))

    print_summary(summary)

    output_dir = Path(args.output_dir)
    output_file = save_metrics(summary, output_dir)
    print(f"\nMetrics saved to: {output_file}")
    print(f"Latest metrics: {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
