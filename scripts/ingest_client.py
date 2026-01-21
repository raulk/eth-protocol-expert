#!/usr/bin/env python3
"""Ingest Ethereum client codebases into the database (Phase 10).

Usage:
    python scripts/ingest_client.py --repo go-ethereum
    python scripts/ingest_client.py --repo prysm --language go
    python scripts/ingest_client.py --repo reth --language rust

This script:
1. Clones/updates the client repository
2. Parses Go/Rust source files using tree-sitter
3. Extracts code units (functions, structs, methods)
4. Generates embeddings for code content
5. Links implementations to EIP specifications
"""

import argparse
import asyncio
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog
from dotenv import load_dotenv

from src.chunking import Chunk
from src.embeddings.voyage_embedder import VoyageEmbedder
from src.graph import FalkorDBStore, SpecImplLinker
from src.parsing import CodeUnit, CodeUnitExtractor, TreeSitterParser
from src.storage.pg_vector_store import PgVectorStore

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ]
)
logger = structlog.get_logger()


@dataclass
class ClientRepo:
    """Configuration for an Ethereum client repository."""

    name: str
    url: str
    language: str
    source_dirs: list[str]


KNOWN_REPOS: dict[str, ClientRepo] = {
    "go-ethereum": ClientRepo(
        name="go-ethereum",
        url="https://github.com/ethereum/go-ethereum.git",
        language="go",
        source_dirs=["core", "consensus", "eth", "params", "miner", "trie"],
    ),
    "prysm": ClientRepo(
        name="prysm",
        url="https://github.com/prysmaticlabs/prysm.git",
        language="go",
        source_dirs=["beacon-chain", "validator", "consensus-types"],
    ),
    "reth": ClientRepo(
        name="reth",
        url="https://github.com/paradigmxyz/reth.git",
        language="rust",
        source_dirs=["crates"],
    ),
    "lighthouse": ClientRepo(
        name="lighthouse",
        url="https://github.com/sigp/lighthouse.git",
        language="rust",
        source_dirs=["beacon_node", "consensus", "validator_client"],
    ),
}


def clone_or_update_repo(repo: ClientRepo, data_dir: Path) -> str:
    """Clone or update the repository. Returns the git commit hash."""
    repo_path = data_dir / repo.name

    if repo_path.exists():
        logger.info("updating_repo", repo=repo.name)
        subprocess.run(
            ["git", "fetch", "--all"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "reset", "--hard", "origin/HEAD"],
            cwd=repo_path,
            capture_output=True,
            check=False,
        )
    else:
        logger.info("cloning_repo", repo=repo.name, url=repo.url)
        subprocess.run(
            ["git", "clone", "--depth", "100", repo.url, str(repo_path)],
            capture_output=True,
            check=True,
        )

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def find_source_files(repo_path: Path, repo: ClientRepo) -> list[Path]:
    """Find all source files in the repository."""
    extension = ".go" if repo.language == "go" else ".rs"
    files: list[Path] = []

    for source_dir in repo.source_dirs:
        source_path = repo_path / source_dir
        if source_path.exists():
            for file_path in source_path.rglob(f"*{extension}"):
                if "_test.go" in file_path.name:
                    continue
                if "_test.rs" in file_path.name:
                    continue
                if "/test/" in str(file_path) or "/tests/" in str(file_path):
                    continue
                if "/testdata/" in str(file_path) or "/fixtures/" in str(file_path):
                    continue
                if "/mock" in str(file_path).lower():
                    continue
                files.append(file_path)

    return files


def code_unit_to_chunk(
    unit: CodeUnit,
    repo_name: str,
    language: str,
    git_commit: str,
) -> Chunk:
    """Convert a CodeUnit to a Chunk for storage."""
    relative_path = unit.file_path
    if repo_name in relative_path:
        relative_path = relative_path.split(repo_name + "/", 1)[-1]

    chunk_id = f"code:{repo_name}:{relative_path}:{unit.name}:{unit.line_range[0]}"

    return Chunk(
        chunk_id=chunk_id,
        document_id=f"code:{repo_name}",
        content=unit.content,
        token_count=len(unit.content.split()),
        position=unit.line_range[0],
        section_path=relative_path,
        metadata={
            "content_type": "code",
            "repository": repo_name,
            "language": language,
            "file_path": relative_path,
            "name": unit.name,
            "unit_type": unit.unit_type.value,
            "start_line": unit.line_range[0],
            "end_line": unit.line_range[1],
            "dependencies": unit.dependencies,
            "git_commit": git_commit,
            "indexed_at": datetime.now(UTC).isoformat(),
        },
    )


async def ingest_client(
    repo_name: str,
    data_dir: str = "data",
    batch_size: int = 50,
    limit: int | None = None,
    link_eips: bool = True,
    eip_numbers: list[int] | None = None,
):
    """Main ingestion pipeline for a client codebase."""
    load_dotenv()

    if repo_name not in KNOWN_REPOS:
        available = ", ".join(KNOWN_REPOS.keys())
        raise ValueError(f"Unknown repository: {repo_name}. Available: {available}")

    repo = KNOWN_REPOS[repo_name]
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    logger.info("starting_client_ingestion", repo=repo_name, language=repo.language)

    git_commit = clone_or_update_repo(repo, data_path)
    logger.info("repo_ready", commit=git_commit[:8])

    repo_path = data_path / repo_name
    source_files = find_source_files(repo_path, repo)
    logger.info("found_source_files", count=len(source_files))

    if limit:
        source_files = source_files[:limit]
        logger.info("limiting_to", count=limit)

    parser = TreeSitterParser(languages=[repo.language])
    extractor = CodeUnitExtractor()

    logger.info("parsing_source_files")
    all_units: list[CodeUnit] = []
    parsed_count = 0
    error_count = 0

    for file_path in source_files:
        try:
            parsed_file = parser.parse_file(str(file_path))
            units = extractor.extract_all(parsed_file)
            all_units.extend(units)
            parsed_count += 1

            if parsed_count % 100 == 0:
                logger.info(
                    "parsing_progress",
                    parsed=parsed_count,
                    total=len(source_files),
                    units=len(all_units),
                )
        except Exception as e:
            error_count += 1
            logger.warning("parse_error", file=str(file_path), error=str(e))

    logger.info(
        "parsing_complete",
        files_parsed=parsed_count,
        errors=error_count,
        total_units=len(all_units),
    )

    logger.info("converting_to_chunks")
    all_chunks = [
        code_unit_to_chunk(unit, repo_name, repo.language, git_commit) for unit in all_units
    ]

    embedder = VoyageEmbedder()
    store = PgVectorStore()

    await store.connect()
    await store.initialize_schema()

    try:
        await store.store_document(
            document_id=f"code:{repo_name}",
            eip_number=None,
            title=f"{repo_name} Client Codebase",
            status="active",
            type_="codebase",
            category=repo.language,
            author=repo_name,
            created_date=datetime.now(UTC).strftime("%Y-%m-%d"),
            requires=[],
            raw_content=f"Ethereum client: {repo_name} ({repo.language})",
            git_commit=git_commit,
        )

        logger.info("embedding_code_units", total=len(all_chunks))
        embed_batch_size = 64
        all_embedded = []

        for i in range(0, len(all_chunks), embed_batch_size):
            batch = all_chunks[i : i + embed_batch_size]
            embedded = embedder.embed_chunks(batch)
            all_embedded.extend(embedded)

            if len(all_embedded) >= batch_size * 4:
                await store.store_embedded_chunks(all_embedded, git_commit)
                all_embedded = []
                logger.info(
                    "embedding_progress",
                    embedded=i + len(batch),
                    total=len(all_chunks),
                )

        if all_embedded:
            await store.store_embedded_chunks(all_embedded, git_commit)

        final_chunks = await store.count_chunks()
        logger.info(
            "ingestion_complete",
            repository=repo_name,
            code_units=len(all_units),
            chunks_stored=len(all_chunks),
            total_chunks_in_db=final_chunks,
        )

        if link_eips:
            logger.info("linking_eips_to_implementations")

            codebase_files: dict[str, str] = {}
            for file_path in source_files[:500]:
                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                    relative_path = str(file_path).split(repo_name + "/", 1)[-1]
                    codebase_files[relative_path] = content
                except Exception:
                    pass

            graph_store = FalkorDBStore()
            try:
                graph_store.connect()
                linker = SpecImplLinker(graph_store=graph_store)

                eips_to_link = eip_numbers or [1559, 4844, 4895, 2718, 2930, 3675, 7516]

                for eip in eips_to_link:
                    links = await linker.find_implementations(eip, codebase_files)
                    if links:
                        logger.info(
                            "eip_linked",
                            eip=eip,
                            implementations=len(links),
                            top_function=links[0].function_name if links else None,
                        )

            except Exception as e:
                logger.warning("eip_linking_failed", error=str(e))
            finally:
                graph_store.close()

        await store.reindex_embeddings()

    finally:
        await store.close()

    logger.info(
        "client_ingestion_complete",
        repository=repo_name,
        language=repo.language,
        files_parsed=parsed_count,
        code_units=len(all_units),
        git_commit=git_commit[:8],
    )


def main():
    parser = argparse.ArgumentParser(description="Ingest Ethereum client codebase")
    parser.add_argument(
        "--repo",
        required=True,
        choices=list(KNOWN_REPOS.keys()),
        help="Client repository to ingest",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory for cloned repositories",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for storing chunks",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of files to process (for testing)",
    )
    parser.add_argument(
        "--no-link-eips",
        action="store_true",
        help="Skip EIP-to-implementation linking",
    )
    parser.add_argument(
        "--eips",
        type=str,
        default=None,
        help="Comma-separated list of EIP numbers to link (default: common EIPs)",
    )

    args = parser.parse_args()

    eip_numbers = None
    if args.eips:
        eip_numbers = [int(e.strip()) for e in args.eips.split(",")]

    asyncio.run(
        ingest_client(
            repo_name=args.repo,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            limit=args.limit,
            link_eips=not args.no_link_eips,
            eip_numbers=eip_numbers,
        )
    )


if __name__ == "__main__":
    main()
