#!/usr/bin/env python3
"""Ingest Ethereum client codebases into the database (Phase 10).

Usage:
    python scripts/ingest_client.py --repo go-ethereum
    python scripts/ingest_client.py --repo prysm --language go
    python scripts/ingest_client.py --repo reth --language rust
    python scripts/ingest_client.py --repo all  # Ingest all clients in parallel

This script:
1. Clones/updates the client repository
2. Parses Go/Rust source files using tree-sitter (concurrent)
3. Extracts code units (functions, structs, methods)
4. Generates embeddings for code content (concurrent batches)
5. Links implementations to EIP specifications (concurrent)
"""

import argparse
import asyncio
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog
from dotenv import load_dotenv
load_dotenv()  # Must run before any src.* imports

from src.chunking import Chunk
from src.chunking.code_chunker import CodeChunk, CodeChunker
from src.embeddings import EmbeddedChunk, create_code_embedder
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

# Concurrency limits
MAX_PARSE_WORKERS = 8  # CPU-bound parsing
MAX_EMBED_CONCURRENT = 4  # API rate limiting
MAX_EIP_CONCURRENT = 3  # Graph DB connections


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


def parse_single_file(file_path: str, language: str) -> list[CodeUnit]:
    """Parse a single file and extract code units. Runs in subprocess."""
    try:
        parser = TreeSitterParser(languages=[language])
        extractor = CodeUnitExtractor()
        parsed_file = parser.parse_file(file_path)
        return extractor.extract_all(parsed_file)
    except Exception:
        return []


def parse_files_concurrent(
    source_files: list[Path],
    language: str,
    max_workers: int = MAX_PARSE_WORKERS,
) -> tuple[list[CodeUnit], int, int]:
    """Parse files concurrently using ProcessPoolExecutor."""
    all_units: list[CodeUnit] = []
    parsed_count = 0
    error_count = 0

    file_paths = [str(f) for f in source_files]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(parse_single_file, fp, language) for fp in file_paths]

        for future in as_completed(futures):
            try:
                units = future.result()
                if units:
                    all_units.extend(units)
                    parsed_count += 1
                else:
                    error_count += 1
            except Exception:
                error_count += 1

            total_done = parsed_count + error_count
            if total_done % 100 == 0:
                logger.info(
                    "parsing_progress",
                    parsed=parsed_count,
                    errors=error_count,
                    total=len(source_files),
                    units=len(all_units),
                )

    return all_units, parsed_count, error_count


async def embed_batch_async(
    embedder,
    chunks: list[CodeChunk],
    semaphore: asyncio.Semaphore,
) -> list:
    """Embed a batch of code chunks with semaphore-limited concurrency."""
    async with semaphore:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, embedder.embed_chunks, chunks)


async def embed_chunks_concurrent(
    embedder,
    all_chunks: list[CodeChunk],
    batch_size: int = 64,
    max_concurrent: int = MAX_EMBED_CONCURRENT,
) -> list:
    """Embed code chunks with concurrent API calls."""
    semaphore = asyncio.Semaphore(max_concurrent)
    all_embedded = []

    batches = [all_chunks[i : i + batch_size] for i in range(0, len(all_chunks), batch_size)]

    logger.info("embedding_concurrent", total_chunks=len(all_chunks), batches=len(batches))

    tasks = [embed_batch_async(embedder, batch, semaphore) for batch in batches]

    completed = 0
    for coro in asyncio.as_completed(tasks):
        result = await coro
        all_embedded.extend(result)
        completed += 1
        logger.info(
            "embedding_progress",
            batches_done=completed,
            total_batches=len(batches),
            chunks_embedded=len(all_embedded),
        )

    return all_embedded


async def link_eip_async(
    eip: int,
    linker: SpecImplLinker,
    codebase_files: dict[str, str],
    semaphore: asyncio.Semaphore,
) -> tuple[int, int, str | None]:
    """Link a single EIP with semaphore-limited concurrency."""
    async with semaphore:
        links = await linker.find_implementations(eip, codebase_files)
        top_func = links[0].function_name if links else None
        return eip, len(links), top_func


async def link_eips_concurrent(
    linker: SpecImplLinker,
    codebase_files: dict[str, str],
    eip_numbers: list[int],
    max_concurrent: int = MAX_EIP_CONCURRENT,
) -> None:
    """Link multiple EIPs concurrently."""
    semaphore = asyncio.Semaphore(max_concurrent)

    tasks = [link_eip_async(eip, linker, codebase_files, semaphore) for eip in eip_numbers]

    for coro in asyncio.as_completed(tasks):
        eip, count, top_func = await coro
        if count > 0:
            logger.info(
                "eip_linked",
                eip=eip,
                implementations=count,
                top_function=top_func,
            )


def to_embedded_chunk(code_chunk: CodeChunk, embedding: list[float], repo_name: str, model: str) -> EmbeddedChunk:
    """Bridge CodeEmbedder output back to EmbeddedChunk for storage."""
    return EmbeddedChunk(
        chunk=Chunk(
            chunk_id=code_chunk.chunk_id,
            document_id=f"code:{repo_name}",
            content=code_chunk.content,
            token_count=code_chunk.token_count,
            chunk_index=0,
            section_path=code_chunk.file_path,
        ),
        embedding=embedding,
        model=model,
    )


async def ingest_client(
    repo_name: str,
    data_dir: str = "data",
    batch_size: int = 50,
    limit: int | None = None,
    link_eips: bool = True,
    eip_numbers: list[int] | None = None,
    parse_workers: int = MAX_PARSE_WORKERS,
    embed_concurrent: int = MAX_EMBED_CONCURRENT,
):
    """Main ingestion pipeline for a client codebase with concurrent processing."""

    if repo_name not in KNOWN_REPOS:
        available = ", ".join(KNOWN_REPOS.keys())
        raise ValueError(f"Unknown repository: {repo_name}. Available: {available}")

    repo = KNOWN_REPOS[repo_name]
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    logger.info(
        "starting_client_ingestion",
        repo=repo_name,
        language=repo.language,
        parse_workers=parse_workers,
        embed_concurrent=embed_concurrent,
    )

    git_commit = clone_or_update_repo(repo, data_path)
    logger.info("repo_ready", commit=git_commit[:8])

    repo_path = data_path / repo_name
    source_files = find_source_files(repo_path, repo)
    logger.info("found_source_files", count=len(source_files))

    if limit:
        source_files = source_files[:limit]
        logger.info("limiting_to", count=limit)

    # Concurrent parsing
    logger.info("parsing_source_files_concurrent", workers=parse_workers)
    all_units, parsed_count, error_count = parse_files_concurrent(
        source_files, repo.language, max_workers=parse_workers
    )

    logger.info(
        "parsing_complete",
        files_parsed=parsed_count,
        errors=error_count,
        total_units=len(all_units),
    )

    chunker = CodeChunker()
    logger.info("chunking_code_units")
    code_chunks = chunker.chunk(all_units, language=repo.language)
    logger.info("chunking_complete", code_chunks=len(code_chunks))

    embedder = create_code_embedder()
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

        logger.info(
            "embedding_code_units_concurrent",
            total=len(code_chunks),
            concurrent_batches=embed_concurrent,
        )
        code_embeddings = await embed_chunks_concurrent(
            embedder, code_chunks, batch_size=64, max_concurrent=embed_concurrent
        )

        # Bridge CodeEmbedding → EmbeddedChunk for storage
        all_embedded = [
            to_embedded_chunk(cc, ce.embedding, repo_name, embedder.model)
            for cc, ce in zip(code_chunks, code_embeddings, strict=True)
        ]

        # Store in batches
        store_batch_size = batch_size * 4
        for i in range(0, len(all_embedded), store_batch_size):
            batch = all_embedded[i : i + store_batch_size]
            await store.store_embedded_chunks(batch, git_commit)
            logger.info("stored_chunks", count=len(batch), progress=i + len(batch))

        final_chunks = await store.count_chunks()
        logger.info(
            "ingestion_complete",
            repository=repo_name,
            code_units=len(all_units),
            chunks_stored=len(code_chunks),
            total_chunks_in_db=final_chunks,
        )

        if link_eips:
            logger.info("linking_eips_to_implementations_concurrent")

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

                # Concurrent EIP linking
                await link_eips_concurrent(
                    linker, codebase_files, eips_to_link, max_concurrent=MAX_EIP_CONCURRENT
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


async def ingest_all_clients(
    data_dir: str = "data",
    batch_size: int = 50,
    link_eips: bool = True,
    eip_numbers: list[int] | None = None,
):
    """Ingest all known client repositories in parallel."""

    repos = list(KNOWN_REPOS.keys())
    logger.info("ingesting_all_clients", repos=repos)

    # Run all ingestions concurrently
    tasks = [
        ingest_client(
            repo_name=repo,
            data_dir=data_dir,
            batch_size=batch_size,
            link_eips=link_eips,
            eip_numbers=eip_numbers,
            # Reduce concurrency per repo when running multiple
            parse_workers=4,
            embed_concurrent=2,
        )
        for repo in repos
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    for repo, result in zip(repos, results, strict=True):
        if isinstance(result, Exception):
            logger.error("client_ingestion_failed", repo=repo, error=str(result))
        else:
            logger.info("client_ingestion_succeeded", repo=repo)


def main():
    parser = argparse.ArgumentParser(description="Ingest Ethereum client codebase")
    parser.add_argument(
        "--repo",
        default="all",
        choices=[*KNOWN_REPOS.keys(), "all"],
        help="Client repository to ingest (default: all)",
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
    parser.add_argument(
        "--parse-workers",
        type=int,
        default=MAX_PARSE_WORKERS,
        help=f"Number of parallel workers for parsing (default: {MAX_PARSE_WORKERS})",
    )
    parser.add_argument(
        "--embed-concurrent",
        type=int,
        default=MAX_EMBED_CONCURRENT,
        help=f"Number of concurrent embedding batches (default: {MAX_EMBED_CONCURRENT})",
    )

    args = parser.parse_args()

    eip_numbers = None
    if args.eips:
        eip_numbers = [int(e.strip()) for e in args.eips.split(",")]

    if args.repo == "all":
        asyncio.run(
            ingest_all_clients(
                data_dir=args.data_dir,
                batch_size=args.batch_size,
                link_eips=not args.no_link_eips,
                eip_numbers=eip_numbers,
            )
        )
    else:
        asyncio.run(
            ingest_client(
                repo_name=args.repo,
                data_dir=args.data_dir,
                batch_size=args.batch_size,
                limit=args.limit,
                link_eips=not args.no_link_eips,
                eip_numbers=eip_numbers,
                parse_workers=args.parse_workers,
                embed_concurrent=args.embed_concurrent,
            )
        )


if __name__ == "__main__":
    main()
