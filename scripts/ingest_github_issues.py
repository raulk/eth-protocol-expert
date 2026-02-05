#!/usr/bin/env python3
"""Ingest GitHub issues and PRs into the database.

Usage:
    uv run python scripts/ingest_github_issues.py
    uv run python scripts/ingest_github_issues.py --repos ethereum/EIPs,ethereum/ERCs
    uv run python scripts/ingest_github_issues.py --since 2026-01-01T00:00:00Z
    uv run python scripts/ingest_github_issues.py --prs-only --no-comments
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import structlog
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chunking import TextChunker
from src.embeddings import VoyageEmbedder
from src.ingestion import DEFAULT_GITHUB_REPOS, GitHubIssuesLoader
from src.ingestion.orchestrator import IngestionOrchestrator
from src.storage import PgVectorStore

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ]
)
logger = structlog.get_logger()


def _parse_repo(repo_str: str) -> tuple[str, str]:
    if "/" not in repo_str:
        raise ValueError(f"Invalid repo '{repo_str}'. Expected owner/repo.")
    owner, repo = repo_str.split("/", 1)
    return owner.strip(), repo.strip()


async def ingest_github_issues(
    *,
    repos: list[str] | None = None,
    since: str | None = None,
    include_issues: bool = True,
    include_prs: bool = True,
    include_comments: bool = True,
    max_comments: int = 50,
    max_items: int | None = None,
    max_pages: int | None = None,
    state_path: str = "data/sync_state.json",
    use_state: bool = True,
    reindex: bool = True,
) -> int:
    """Ingest GitHub issues and PRs for configured repos.

    Returns:
        Total documents ingested.
    """
    load_dotenv()

    repo_list = repos or DEFAULT_GITHUB_REPOS
    if not repo_list:
        logger.warning("no_repos_configured")
        return 0

    orchestrator = IngestionOrchestrator(state_path=state_path) if use_state else None
    if orchestrator:
        orchestrator.load_state()

    loader = GitHubIssuesLoader()
    chunker = TextChunker(max_tokens=512, overlap_tokens=64)
    embedder = VoyageEmbedder()
    store = PgVectorStore()

    await store.connect()
    await store.initialize_schema()

    total_docs = 0
    total_chunks = 0

    try:
        for repo_str in repo_list:
            owner, repo = _parse_repo(repo_str)
            source_name = f"github_issues/{owner}/{repo}"

            since_cursor = since
            if orchestrator and since_cursor is None:
                state = orchestrator.get_source_state(source_name)
                since_cursor = state.last_cursor

            logger.info(
                "fetching_github_issues",
                repo=repo_str,
                since=since_cursor,
                issues=include_issues,
                prs=include_prs,
            )

            documents, latest = await loader.fetch_issues(
                owner=owner,
                repo=repo,
                since=since_cursor,
                include_issues=include_issues,
                include_prs=include_prs,
                include_comments=include_comments,
                max_comments=max_comments,
                max_items=max_items,
                max_pages=max_pages,
            )

            if not documents:
                logger.info("no_updates", repo=repo_str)
                if orchestrator:
                    orchestrator.update_source_state(source_name, cursor=latest, docs_synced=0)
                continue

            for doc in documents:
                try:
                    await store.delete_document(doc.document_id)

                    chunks = chunker.chunk_text(
                        doc.content,
                        doc.document_id,
                        section_path="GitHub Issue",
                    )
                    if not chunks:
                        continue

                    embedded = embedder.embed_chunks(chunks)

                    await store.store_generic_document(
                        document_id=doc.document_id,
                        document_type=doc.document_type,
                        title=doc.title,
                        source=doc.source,
                        author=doc.metadata.get("author"),
                        raw_content=doc.content,
                        metadata={
                            **doc.metadata,
                            "fetched_at": doc.fetched_at,
                        },
                    )

                    await store.store_embedded_chunks(embedded)

                    total_docs += 1
                    total_chunks += len(embedded)

                except Exception as e:
                    logger.error(
                        "failed_to_ingest_issue",
                        repo=repo_str,
                        document_id=doc.document_id,
                        error=str(e),
                    )

            if orchestrator:
                orchestrator.update_source_state(
                    source_name,
                    cursor=latest,
                    docs_synced=len(documents),
                    metadata={"latest_updated_at": latest},
                )

            logger.info(
                "repo_ingestion_complete",
                repo=repo_str,
                documents=len(documents),
                latest=latest,
            )

        if reindex and total_docs > 0:
            await store.reindex_embeddings()

    finally:
        await store.close()

    logger.info(
        "github_issues_ingestion_complete",
        documents=total_docs,
        chunks=total_chunks,
    )
    return total_docs


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest GitHub issues and PRs")
    parser.add_argument(
        "--repos",
        type=str,
        help="Comma-separated list of repos (owner/repo)",
    )
    parser.add_argument(
        "--since",
        type=str,
        help="ISO timestamp for incremental sync (overrides state cursor)",
    )
    parser.add_argument(
        "--issues-only",
        action="store_true",
        help="Only ingest issues (skip PRs)",
    )
    parser.add_argument(
        "--prs-only",
        action="store_true",
        help="Only ingest PRs (skip issues)",
    )
    parser.add_argument(
        "--no-comments",
        action="store_true",
        help="Skip fetching comments",
    )
    parser.add_argument(
        "--max-comments",
        type=int,
        default=50,
        help="Max comments per issue (default: 50)",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Max issues/PRs per repo (default: no limit)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Max API pages to fetch per repo (default: no limit)",
    )
    parser.add_argument(
        "--state-path",
        type=str,
        default="data/sync_state.json",
        help="Path to sync state file",
    )
    parser.add_argument(
        "--no-state",
        action="store_true",
        help="Disable state cursor usage",
    )
    parser.add_argument(
        "--no-reindex",
        action="store_true",
        help="Skip vector index rebuild",
    )

    args = parser.parse_args()

    repos = args.repos.split(",") if args.repos else None
    include_issues = not args.prs_only
    include_prs = not args.issues_only

    if args.issues_only and args.prs_only:
        raise SystemExit("Cannot use --issues-only and --prs-only together")

    asyncio.run(
        ingest_github_issues(
            repos=repos,
            since=args.since,
            include_issues=include_issues,
            include_prs=include_prs,
            include_comments=not args.no_comments,
            max_comments=args.max_comments,
            max_items=args.max_items,
            max_pages=args.max_pages,
            state_path=args.state_path,
            use_state=not args.no_state,
            reindex=not args.no_reindex,
        )
    )


if __name__ == "__main__":
    main()
