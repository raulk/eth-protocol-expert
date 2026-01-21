#!/usr/bin/env python3
"""Ingest all data sources into the database.

Usage:
    uv run python scripts/ingest_all.py [--skip-eips] [--skip-forums] [--skip-transcripts] [--skip-papers] [--skip-specs]
"""

import argparse
import asyncio
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import structlog
from dotenv import load_dotenv

load_dotenv()

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ]
)
logger = structlog.get_logger()


@dataclass
class IngestionResult:
    """Result of running an ingestion script."""

    name: str
    success: bool
    return_code: int
    skipped: bool = False


def run_script(script_path: str, args: list[str] | None = None) -> IngestionResult:
    """Run an ingestion script and return result.

    Args:
        script_path: Path to the script to run
        args: Additional arguments to pass to the script

    Returns:
        IngestionResult with success status
    """
    script_name = Path(script_path).stem

    if not Path(script_path).exists():
        logger.warning("script_not_found", script=script_path)
        return IngestionResult(
            name=script_name,
            success=False,
            return_code=-1,
            skipped=True,
        )

    cmd = ["uv", "run", "python", script_path]
    if args:
        cmd.extend(args)

    logger.info("running_script", script=script_name, command=" ".join(cmd))

    result = subprocess.run(cmd, capture_output=False)

    success = result.returncode == 0
    if success:
        logger.info("script_completed", script=script_name)
    else:
        logger.error("script_failed", script=script_name, return_code=result.returncode)

    return IngestionResult(
        name=script_name,
        success=success,
        return_code=result.returncode,
    )


async def ingest_all(
    skip_eips: bool = False,
    skip_forums: bool = False,
    skip_transcripts: bool = False,
    skip_papers: bool = False,
    skip_specs: bool = False,
) -> list[IngestionResult]:
    """Run all ingestion scripts in sequence.

    Args:
        skip_eips: Skip EIP ingestion
        skip_forums: Skip forum ingestion (ethresear.ch and Ethereum Magicians)
        skip_transcripts: Skip ACD transcript ingestion
        skip_papers: Skip arXiv paper ingestion
        skip_specs: Skip consensus and execution spec ingestion

    Returns:
        List of IngestionResult for each script run
    """
    results: list[IngestionResult] = []

    if not skip_eips:
        logger.info("phase", name="EIPs")
        results.append(run_script("scripts/ingest_eips.py"))
    else:
        logger.info("skipping_phase", name="EIPs")

    if not skip_forums:
        logger.info("phase", name="ethresear.ch")
        results.append(
            run_script("scripts/ingest_ethresearch.py", ["--max-topics", "1000"])
        )

        logger.info("phase", name="Ethereum Magicians")
        results.append(
            run_script("scripts/ingest_magicians.py", ["--max-topics", "1000"])
        )
    else:
        logger.info("skipping_phase", name="Forums")

    if not skip_transcripts:
        logger.info("phase", name="ACD Transcripts")
        results.append(run_script("scripts/ingest_acd_transcripts.py"))
    else:
        logger.info("skipping_phase", name="ACD Transcripts")

    if not skip_papers:
        logger.info("phase", name="arXiv Papers")
        results.append(run_script("scripts/ingest_arxiv.py", ["--max-papers", "300"]))
    else:
        logger.info("skipping_phase", name="arXiv Papers")

    if not skip_specs:
        logger.info("phase", name="Consensus Specs")
        results.append(run_script("scripts/ingest_consensus_specs.py"))

        logger.info("phase", name="Execution Specs")
        results.append(run_script("scripts/ingest_execution_specs.py"))
    else:
        logger.info("skipping_phase", name="Specs")

    return results


async def reindex_database() -> bool:
    """Rebuild vector indexes after bulk ingestion.

    IVFFlat indexes need rebuilding after bulk inserts for new data
    to be properly indexed in similarity searches.

    Returns:
        True if reindex succeeded, False otherwise
    """
    from src.storage import PgVectorStore

    logger.info("reindexing_database")
    try:
        store = PgVectorStore()
        await store.connect()
        await store.reindex_embeddings()
        await store.close()
        logger.info("reindex_completed")
        return True
    except Exception as e:
        logger.error("reindex_failed", error=str(e))
        return False


def print_summary(results: list[IngestionResult]) -> None:
    """Print a summary of all ingestion results."""
    print("\n" + "=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)

    for result in results:
        if result.skipped:
            status = "SKIPPED (not found)"
        elif result.success:
            status = "SUCCESS"
        else:
            status = f"FAILED (exit code {result.return_code})"
        print(f"  {result.name}: {status}")

    print("=" * 60)

    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success and not r.skipped]
    skipped = [r for r in results if r.skipped]

    print(f"Total: {len(results)} scripts")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed: {len(failed)}")
    print(f"  Skipped: {len(skipped)}")
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest all data sources into the database"
    )
    parser.add_argument(
        "--skip-eips",
        action="store_true",
        help="Skip EIP ingestion",
    )
    parser.add_argument(
        "--skip-forums",
        action="store_true",
        help="Skip forum ingestion (ethresear.ch and Ethereum Magicians)",
    )
    parser.add_argument(
        "--skip-transcripts",
        action="store_true",
        help="Skip ACD transcript ingestion",
    )
    parser.add_argument(
        "--skip-papers",
        action="store_true",
        help="Skip arXiv paper ingestion",
    )
    parser.add_argument(
        "--skip-specs",
        action="store_true",
        help="Skip consensus and execution spec ingestion",
    )

    args = parser.parse_args()

    logger.info(
        "starting_full_ingestion",
        skip_eips=args.skip_eips,
        skip_forums=args.skip_forums,
        skip_transcripts=args.skip_transcripts,
        skip_papers=args.skip_papers,
        skip_specs=args.skip_specs,
    )

    results = asyncio.run(
        ingest_all(
            skip_eips=args.skip_eips,
            skip_forums=args.skip_forums,
            skip_transcripts=args.skip_transcripts,
            skip_papers=args.skip_papers,
            skip_specs=args.skip_specs,
        )
    )

    print_summary(results)

    failed = [r for r in results if not r.success and not r.skipped]
    if failed:
        logger.error("some_ingestions_failed", failed=[r.name for r in failed])
        sys.exit(1)

    # Rebuild vector indexes after bulk ingestion
    reindex_success = asyncio.run(reindex_database())
    if not reindex_success:
        logger.warning("reindex_failed_but_ingestion_succeeded")

    logger.info("all_ingestions_completed_successfully")


if __name__ == "__main__":
    main()
