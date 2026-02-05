#!/usr/bin/env python3
"""Continuous Ingestion - Orchestrate incremental updates across all sources.

This script coordinates incremental syncs for all configured corpus sources,
tracking state to avoid re-processing unchanged content.

Usage:
    uv run python scripts/continuous_ingestion.py --once       # Single sync run
    uv run python scripts/continuous_ingestion.py --sources eips,ercs  # Specific sources
    uv run python scripts/continuous_ingestion.py --dry-run    # Show what would sync
"""

import argparse
import asyncio
import sys
import time
from datetime import datetime, UTC
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog
from dotenv import load_dotenv

from src.ingestion.orchestrator import IngestionOrchestrator, GitHubSyncer
from src.storage.pg_vector_store import PgVectorStore

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ]
)
logger = structlog.get_logger()

ALL_SOURCES = [
    "eips",
    "ercs",
    "rips",
    "consensus_specs",
    "execution_specs",
    "devp2p_specs",
    "portal_specs",
    "builder_specs",
    "execution_apis",
    "beacon_apis",
    "research",
    "ethresearch",
    "magicians",
    "github_issues",
]


async def sync_eips(orchestrator: IngestionOrchestrator, dry_run: bool = False) -> dict:
    """Sync EIPs from ethereum/EIPs repository."""
    from src.ingestion import EIPLoader

    source_name = "eips"
    loader = EIPLoader()
    repo_path = Path("data/eips")

    if dry_run:
        return {"source": source_name, "status": "dry_run"}

    current_commit = loader.clone_or_update()
    state = orchestrator.get_source_state(source_name)

    if state.last_commit == current_commit:
        logger.info("already_up_to_date", source=source_name, commit=current_commit[:8])
        return {"source": source_name, "synced": 0, "status": "already_up_to_date"}

    from scripts.ingest_eips import ingest_eips

    await ingest_eips()

    orchestrator.update_source_state(source_name, commit=current_commit, docs_synced=1)
    return {"source": source_name, "synced": 1, "status": "success"}


async def sync_ercs(orchestrator: IngestionOrchestrator, dry_run: bool = False) -> dict:
    """Sync ERCs from ethereum/ERCs repository."""
    source_name = "ercs"

    if dry_run:
        return {"source": source_name, "status": "dry_run"}

    from src.ingestion import ERCLoader

    loader = ERCLoader()
    current_commit = loader.clone_or_update()
    state = orchestrator.get_source_state(source_name)

    if state.last_commit == current_commit:
        logger.info("already_up_to_date", source=source_name, commit=current_commit[:8])
        return {"source": source_name, "synced": 0, "status": "already_up_to_date"}

    from scripts.ingest_ercs import ingest_ercs

    await ingest_ercs()

    orchestrator.update_source_state(source_name, commit=current_commit, docs_synced=1)
    return {"source": source_name, "synced": 1, "status": "success"}


async def sync_rips(orchestrator: IngestionOrchestrator, dry_run: bool = False) -> dict:
    """Sync RIPs from ethereum/RIPs repository."""
    source_name = "rips"

    if dry_run:
        return {"source": source_name, "status": "dry_run"}

    from src.ingestion import RIPLoader

    loader = RIPLoader()
    current_commit = loader.clone_or_update()
    state = orchestrator.get_source_state(source_name)

    if state.last_commit == current_commit:
        logger.info("already_up_to_date", source=source_name, commit=current_commit[:8])
        return {"source": source_name, "synced": 0, "status": "already_up_to_date"}

    from scripts.ingest_rips import ingest_rips

    await ingest_rips()

    orchestrator.update_source_state(source_name, commit=current_commit, docs_synced=1)
    return {"source": source_name, "synced": 1, "status": "success"}


async def sync_consensus_specs(
    orchestrator: IngestionOrchestrator, dry_run: bool = False
) -> dict:
    """Sync consensus specs from ethereum/consensus-specs."""
    source_name = "consensus_specs"

    if dry_run:
        return {"source": source_name, "status": "dry_run"}

    from src.ingestion import ConsensusSpecLoader

    loader = ConsensusSpecLoader()
    current_commit = loader.clone_or_update()
    state = orchestrator.get_source_state(source_name)

    if state.last_commit == current_commit:
        logger.info("already_up_to_date", source=source_name, commit=current_commit[:8])
        return {"source": source_name, "synced": 0, "status": "already_up_to_date"}

    from scripts.ingest_consensus_specs import ingest_consensus_specs

    await ingest_consensus_specs()

    orchestrator.update_source_state(source_name, commit=current_commit, docs_synced=1)
    return {"source": source_name, "synced": 1, "status": "success"}


async def sync_execution_specs(
    orchestrator: IngestionOrchestrator, dry_run: bool = False
) -> dict:
    """Sync execution specs from ethereum/execution-specs."""
    source_name = "execution_specs"

    if dry_run:
        return {"source": source_name, "status": "dry_run"}

    from src.ingestion import ExecutionSpecLoader

    loader = ExecutionSpecLoader()
    current_commit = loader.clone_or_update()
    state = orchestrator.get_source_state(source_name)

    if state.last_commit == current_commit:
        logger.info("already_up_to_date", source=source_name, commit=current_commit[:8])
        return {"source": source_name, "synced": 0, "status": "already_up_to_date"}

    from scripts.ingest_execution_specs import ingest_execution_specs

    await ingest_execution_specs()

    orchestrator.update_source_state(source_name, commit=current_commit, docs_synced=1)
    return {"source": source_name, "synced": 1, "status": "success"}


async def sync_research(
    orchestrator: IngestionOrchestrator, dry_run: bool = False
) -> dict:
    """Sync research from ethereum/research repository."""
    source_name = "research"

    if dry_run:
        return {"source": source_name, "status": "dry_run"}

    from src.ingestion import ResearchLoader

    loader = ResearchLoader()
    current_commit = loader.clone_or_update()
    state = orchestrator.get_source_state(source_name)

    if state.last_commit == current_commit:
        logger.info("already_up_to_date", source=source_name, commit=current_commit[:8])
        return {"source": source_name, "synced": 0, "status": "already_up_to_date"}

    from scripts.ingest_research import ingest_research

    await ingest_research(batch_size=32)

    orchestrator.update_source_state(source_name, commit=current_commit, docs_synced=1)
    return {"source": source_name, "synced": 1, "status": "success"}


async def sync_ethresearch(
    orchestrator: IngestionOrchestrator, dry_run: bool = False
) -> dict:
    """Sync ethresear.ch forum topics."""
    source_name = "ethresearch"

    if dry_run:
        return {"source": source_name, "status": "dry_run"}

    from scripts.sync_ethresearch import sync_topics

    state = orchestrator.get_source_state(source_name)

    result = await sync_topics(max_topics=500, incremental=True)

    if result.get("synced", 0) == 0:
        logger.info("already_up_to_date", source=source_name)
        return {"source": source_name, "synced": 0, "status": "already_up_to_date"}

    from scripts.ingest_ethresearch import ingest_ethresearch

    await ingest_ethresearch(skip_existing=True)

    orchestrator.update_source_state(
        source_name,
        docs_synced=result.get("synced", 0),
        metadata={"sync_result": result},
    )
    return {"source": source_name, "synced": result.get("synced", 0), "status": "success"}


async def sync_github_issues(
    orchestrator: IngestionOrchestrator, dry_run: bool = False
) -> dict:
    """Sync GitHub issues and PRs for configured repos."""
    source_name = "github_issues"

    if dry_run:
        return {"source": source_name, "status": "dry_run"}

    from scripts.ingest_github_issues import ingest_github_issues

    total = await ingest_github_issues(state_path=str(orchestrator.state_path))
    return {"source": source_name, "synced": total, "status": "success"}


async def sync_all(
    orchestrator: IngestionOrchestrator,
    sources: list[str] | None = None,
    dry_run: bool = False,
) -> dict:
    """Run incremental sync across all or specified sources."""
    start_time = time.time()
    results = []

    source_funcs = {
        "eips": sync_eips,
        "ercs": sync_ercs,
        "rips": sync_rips,
        "consensus_specs": sync_consensus_specs,
        "execution_specs": sync_execution_specs,
        "research": sync_research,
        "ethresearch": sync_ethresearch,
        "github_issues": sync_github_issues,
    }

    if sources is None:
        sources = list(source_funcs.keys())
    else:
        sources = [s for s in sources if s in source_funcs]

    for source in sources:
        try:
            logger.info("starting_sync", source=source)
            result = await source_funcs[source](orchestrator, dry_run)
            results.append(result)
        except Exception as e:
            logger.error("sync_error", source=source, error=str(e))
            results.append({"source": source, "status": "error", "error": str(e)})

    duration = time.time() - start_time
    total_synced = sum(r.get("synced", 0) for r in results)
    errors = sum(1 for r in results if r.get("status") == "error")

    logger.info(
        "sync_complete",
        total_synced=total_synced,
        errors=errors,
        duration_sec=round(duration, 2),
    )

    return {
        "total_synced": total_synced,
        "errors": errors,
        "duration_sec": round(duration, 2),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run continuous ingestion")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single sync iteration",
    )
    parser.add_argument(
        "--sources",
        type=str,
        default=None,
        help=f"Comma-separated list of sources to sync ({','.join(ALL_SOURCES[:5])}...)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be synced without executing",
    )
    parser.add_argument(
        "--state-path",
        type=str,
        default="data/sync_state.json",
        help="Path to sync state file",
    )
    args = parser.parse_args()

    load_dotenv()

    orchestrator = IngestionOrchestrator(state_path=args.state_path)
    orchestrator.load_state()

    sources = args.sources.split(",") if args.sources else None

    if args.dry_run:
        logger.info("dry_run_mode", sources=sources or "all")
        for source in sources or list(orchestrator.state.sources.keys()) or ["(none)"]:
            state = orchestrator.get_source_state(source)
            logger.info(
                "source_state",
                source=source,
                last_sync=state.last_sync,
                last_commit=state.last_commit[:8] if state.last_commit else None,
                docs_synced=state.docs_synced,
            )
        return

    result = asyncio.run(sync_all(orchestrator, sources, args.dry_run))

    if result["errors"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
