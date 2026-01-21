# LlamaIndex GitHub ingestion plan

## Overview

Replace custom git-based loaders with LlamaIndex GitHub readers for unified, cacheable GitHub ingestion across all ethereum org repositories.

## Current state

| Repo | Current Loader | Data Loaded |
|------|----------------|-------------|
| ethereum/EIPs | `EIPLoader` (git clone) | Files only |
| ethereum/consensus-specs | `ConsensusSpecLoader` (git clone) | Files only |
| ethereum/execution-specs | `ExecutionSpecLoader` (git clone) | Files only |
| ethereum/pm | `ACDTranscriptLoader` (git clone) | Files only |
| ethereum/ERCs | None | - |
| ethereum/RIPs | None | - |
| ethereum/devp2p | None | - |

**Gap:** No issues, PRs, or discussions ingested. These contain valuable context (EIP rationale, design debates, meeting agendas).

## Proposed architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    GitHubIngestionManager                        │
│  - Manages all GitHub-based ingestion                           │
│  - Coordinates caching                                          │
│  - Handles rate limiting                                        │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LlamaIndex Readers                           │
│  ┌─────────────────────┐  ┌─────────────────────────────────┐  │
│  │ GithubRepositoryReader │  │ GitHubRepositoryIssuesReader │  │
│  │ - Files              │  │ - Issues                       │  │
│  │ - Branches           │  │ - Pull Requests                │  │
│  │ - Filtering          │  │ - Labels                       │  │
│  └─────────────────────┘  └─────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RawContentCache                               │
│  - Caches LlamaIndex Document objects                           │
│  - Staleness detection via updated_at                           │
│  - Enables re-processing without re-fetching                    │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Domain-Specific Processing                       │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────────┐│
│  │EIPProcessor │  │SpecProcessor│  │ TranscriptProcessor      ││
│  │- Frontmatter│  │- Fork detect│  │ - Speaker extraction     ││
│  │- Cross-refs │  │- Code blocks│  │ - Topic segmentation     ││
│  └─────────────┘  └─────────────┘  └──────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│              Existing Pipeline (unchanged)                       │
│  Chunking → Embedding → Storage → Retrieval → Generation        │
└─────────────────────────────────────────────────────────────────┘
```

---

## LlamaIndex reader configuration

### Repository configurations

```python
GITHUB_REPOS = {
    "ethereum/EIPs": {
        "files": {
            "filter_directories": (["EIPS"], FilterType.INCLUDE),
            "filter_extensions": ([".md"], FilterType.INCLUDE),
        },
        "issues": {
            "state": IssueState.ALL,
            "labels": [],  # All labels
        },
        "document_type": "eip",
    },
    "ethereum/ERCs": {
        "files": {
            "filter_directories": (["ERCS"], FilterType.INCLUDE),
            "filter_extensions": ([".md"], FilterType.INCLUDE),
        },
        "issues": {
            "state": IssueState.ALL,
        },
        "document_type": "erc",
    },
    "ethereum/RIPs": {
        "files": {
            "filter_directories": (["RIPS"], FilterType.INCLUDE),
            "filter_extensions": ([".md"], FilterType.INCLUDE),
        },
        "issues": {
            "state": IssueState.ALL,
        },
        "document_type": "rip",
    },
    "ethereum/consensus-specs": {
        "files": {
            "filter_directories": (["specs"], FilterType.INCLUDE),
            "filter_extensions": ([".md", ".py"], FilterType.INCLUDE),
        },
        "issues": None,  # Skip - not valuable
        "document_type": "consensus_spec",
    },
    "ethereum/execution-specs": {
        "files": {
            "filter_directories": (["src"], FilterType.INCLUDE),
            "filter_extensions": ([".py", ".md"], FilterType.INCLUDE),
        },
        "issues": None,
        "document_type": "execution_spec",
    },
    "ethereum/pm": {
        "files": {
            "filter_directories": (
                ["AllCoreDevs-EL-Meetings", "AllCoreDevs-CL-Meetings"],
                FilterType.INCLUDE,
            ),
            "filter_extensions": ([".md"], FilterType.INCLUDE),
        },
        "issues": {
            "state": IssueState.ALL,
            "labels": [("agenda", FilterType.INCLUDE)],  # Meeting agendas
        },
        "document_type": "acd_transcript",
    },
    "ethereum/devp2p": {
        "files": {
            "filter_directories": (["caps", "discv4", "discv5", "enr", "rlpx"], FilterType.INCLUDE),
            "filter_extensions": ([".md"], FilterType.INCLUDE),
        },
        "issues": None,
        "document_type": "devp2p_spec",
    },
}
```

---

## Implementation phases

### Phase 1: Add LlamaIndex dependency

**File:** `pyproject.toml`

```toml
[project]
dependencies = [
    # ... existing deps
    "llama-index-readers-github>=0.3.0",
]
```

```bash
uv add llama-index-readers-github
```

---

### Phase 2: Create unified GitHub loader

**File:** `src/ingestion/github_loader.py`

```python
"""Unified GitHub repository loader using LlamaIndex."""

import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

import structlog
from llama_index.readers.github import (
    GithubClient,
    GithubRepositoryReader,
    GitHubIssuesClient,
    GitHubRepositoryIssuesReader,
)

from .cache import RawContentCache

logger = structlog.get_logger()


class DocumentType(str, Enum):
    EIP = "eip"
    ERC = "erc"
    RIP = "rip"
    CONSENSUS_SPEC = "consensus_spec"
    EXECUTION_SPEC = "execution_spec"
    ACD_TRANSCRIPT = "acd_transcript"
    DEVP2P_SPEC = "devp2p_spec"
    GITHUB_ISSUE = "github_issue"
    GITHUB_PR = "github_pr"


@dataclass
class GitHubDocument:
    """Unified document from GitHub."""

    document_id: str
    document_type: DocumentType
    source: str  # "ethereum/EIPs"
    title: str
    content: str
    metadata: dict
    fetched_at: datetime


@dataclass
class RepoConfig:
    """Configuration for a GitHub repository."""

    owner: str
    repo: str
    document_type: DocumentType
    file_directories: list[str] | None = None
    file_extensions: list[str] | None = None
    load_issues: bool = False
    issue_labels: list[str] | None = None
    branch: str = "main"


class GitHubLoader:
    """Load documents from GitHub repositories using LlamaIndex readers.

    Supports:
    - Repository files with filtering
    - Issues and pull requests
    - Caching for re-ingestion efficiency
    """

    def __init__(
        self,
        github_token: str | None = None,
        cache: RawContentCache | None = None,
    ):
        self.github_token = github_token or os.environ.get("GITHUB_TOKEN")
        if not self.github_token:
            raise ValueError("GITHUB_TOKEN required")

        self.cache = cache or RawContentCache()
        self._github_client: GithubClient | None = None
        self._issues_client: GitHubIssuesClient | None = None

    def _get_github_client(self) -> GithubClient:
        if not self._github_client:
            self._github_client = GithubClient(
                github_token=self.github_token,
                verbose=False,
            )
        return self._github_client

    def _get_issues_client(self) -> GitHubIssuesClient:
        if not self._issues_client:
            self._issues_client = GitHubIssuesClient(
                github_token=self.github_token,
                verbose=False,
            )
        return self._issues_client

    def load_repository_files(
        self,
        config: RepoConfig,
        use_cache: bool = True,
        max_cache_age_hours: float = 24 * 7,
    ) -> list[GitHubDocument]:
        """Load files from a GitHub repository.

        Args:
            config: Repository configuration
            use_cache: Whether to use cached content
            max_cache_age_hours: Max cache age before refresh

        Returns:
            List of GitHubDocument objects
        """
        source = f"{config.owner}/{config.repo}"
        cache_key = f"files-{config.branch}"

        # Check cache
        if use_cache and not self.cache.is_stale(source, cache_key, max_cache_age_hours):
            cached = self._load_from_cache(source, cache_key)
            if cached:
                logger.info("using_cached_files", source=source, count=len(cached))
                return cached

        # Build filter tuples
        dir_filter = None
        if config.file_directories:
            dir_filter = (config.file_directories, GithubRepositoryReader.FilterType.INCLUDE)

        ext_filter = None
        if config.file_extensions:
            ext_filter = (config.file_extensions, GithubRepositoryReader.FilterType.INCLUDE)

        # Load from GitHub
        reader = GithubRepositoryReader(
            github_client=self._get_github_client(),
            owner=config.owner,
            repo=config.repo,
            use_parser=False,  # Get raw content
            verbose=False,
            filter_directories=dir_filter,
            filter_file_extensions=ext_filter,
        )

        logger.info("loading_github_files", source=source, branch=config.branch)
        llama_docs = reader.load_data(branch=config.branch)

        # Convert to our format
        documents = []
        for doc in llama_docs:
            file_path = doc.metadata.get("file_path", "unknown")
            doc_id = f"{source}/{file_path}".replace("/", "-")

            documents.append(GitHubDocument(
                document_id=doc_id,
                document_type=config.document_type,
                source=source,
                title=Path(file_path).stem,
                content=doc.text,
                metadata={
                    "file_path": file_path,
                    "branch": config.branch,
                    "sha": doc.metadata.get("sha"),
                    **doc.metadata,
                },
                fetched_at=datetime.utcnow(),
            ))

        # Cache results
        if use_cache:
            self._save_to_cache(source, cache_key, documents)

        logger.info("loaded_github_files", source=source, count=len(documents))
        return documents

    def load_issues(
        self,
        config: RepoConfig,
        use_cache: bool = True,
        max_cache_age_hours: float = 24,  # Shorter for issues
        include_prs: bool = True,
    ) -> list[GitHubDocument]:
        """Load issues and optionally PRs from a repository.

        Args:
            config: Repository configuration
            use_cache: Whether to use cached content
            max_cache_age_hours: Max cache age before refresh
            include_prs: Whether to include pull requests

        Returns:
            List of GitHubDocument objects
        """
        source = f"{config.owner}/{config.repo}"
        cache_key = "issues-all" if include_prs else "issues-only"

        # Check cache
        if use_cache and not self.cache.is_stale(source, cache_key, max_cache_age_hours):
            cached = self._load_from_cache(source, cache_key)
            if cached:
                logger.info("using_cached_issues", source=source, count=len(cached))
                return cached

        # Build label filters
        label_filters = None
        if config.issue_labels:
            label_filters = [
                (label, GitHubRepositoryIssuesReader.FilterType.INCLUDE)
                for label in config.issue_labels
            ]

        # Load from GitHub
        reader = GitHubRepositoryIssuesReader(
            github_client=self._get_issues_client(),
            owner=config.owner,
            repo=config.repo,
            verbose=False,
        )

        logger.info("loading_github_issues", source=source)
        llama_docs = reader.load_data(
            state=GitHubRepositoryIssuesReader.IssueState.ALL,
            labelFilters=label_filters,
        )

        # Convert to our format
        documents = []
        for doc in llama_docs:
            issue_number = doc.metadata.get("number", doc.id_)
            is_pr = doc.metadata.get("pull_request") is not None

            if not include_prs and is_pr:
                continue

            doc_type = DocumentType.GITHUB_PR if is_pr else DocumentType.GITHUB_ISSUE
            doc_id = f"{source}-{'pr' if is_pr else 'issue'}-{issue_number}"

            documents.append(GitHubDocument(
                document_id=doc_id,
                document_type=doc_type,
                source=source,
                title=doc.metadata.get("title", f"Issue #{issue_number}"),
                content=doc.text,
                metadata={
                    "number": issue_number,
                    "state": doc.metadata.get("state"),
                    "labels": doc.metadata.get("labels", []),
                    "author": doc.metadata.get("user", {}).get("login"),
                    "created_at": doc.metadata.get("created_at"),
                    "updated_at": doc.metadata.get("updated_at"),
                    "is_pr": is_pr,
                    **doc.metadata,
                },
                fetched_at=datetime.utcnow(),
            ))

        # Cache results
        if use_cache:
            self._save_to_cache(source, cache_key, documents)

        logger.info("loaded_github_issues", source=source, count=len(documents))
        return documents

    def load_repository(
        self,
        config: RepoConfig,
        use_cache: bool = True,
    ) -> list[GitHubDocument]:
        """Load all configured content from a repository.

        Combines files and issues based on config.
        """
        documents = []

        # Load files
        documents.extend(self.load_repository_files(config, use_cache=use_cache))

        # Load issues if configured
        if config.load_issues:
            documents.extend(self.load_issues(config, use_cache=use_cache))

        return documents

    def _load_from_cache(self, source: str, cache_key: str) -> list[GitHubDocument] | None:
        """Load documents from cache."""
        import json

        content = self.cache.get_content_text(source, cache_key)
        if not content:
            return None

        data = json.loads(content)
        return [
            GitHubDocument(
                document_id=d["document_id"],
                document_type=DocumentType(d["document_type"]),
                source=d["source"],
                title=d["title"],
                content=d["content"],
                metadata=d["metadata"],
                fetched_at=datetime.fromisoformat(d["fetched_at"]),
            )
            for d in data
        ]

    def _save_to_cache(self, source: str, cache_key: str, documents: list[GitHubDocument]) -> None:
        """Save documents to cache."""
        import json

        data = [
            {
                "document_id": d.document_id,
                "document_type": d.document_type.value,
                "source": d.source,
                "title": d.title,
                "content": d.content,
                "metadata": d.metadata,
                "fetched_at": d.fetched_at.isoformat(),
            }
            for d in documents
        ]

        self.cache.put(
            source=source,
            item_id=cache_key,
            content=json.dumps(data, indent=2).encode(),
            meta={"count": len(documents)},
            content_subpath=f"{cache_key}.json",
        )


# Pre-configured repositories
ETHEREUM_REPOS = {
    "EIPs": RepoConfig(
        owner="ethereum",
        repo="EIPs",
        document_type=DocumentType.EIP,
        file_directories=["EIPS"],
        file_extensions=[".md"],
        load_issues=True,
    ),
    "ERCs": RepoConfig(
        owner="ethereum",
        repo="ERCs",
        document_type=DocumentType.ERC,
        file_directories=["ERCS"],
        file_extensions=[".md"],
        load_issues=True,
    ),
    "RIPs": RepoConfig(
        owner="ethereum",
        repo="RIPs",
        document_type=DocumentType.RIP,
        file_directories=["RIPS"],
        file_extensions=[".md"],
        load_issues=True,
    ),
    "consensus-specs": RepoConfig(
        owner="ethereum",
        repo="consensus-specs",
        document_type=DocumentType.CONSENSUS_SPEC,
        file_directories=["specs"],
        file_extensions=[".md", ".py"],
        load_issues=False,
    ),
    "execution-specs": RepoConfig(
        owner="ethereum",
        repo="execution-specs",
        document_type=DocumentType.EXECUTION_SPEC,
        file_directories=["src"],
        file_extensions=[".py", ".md"],
        load_issues=False,
    ),
    "pm": RepoConfig(
        owner="ethereum",
        repo="pm",
        document_type=DocumentType.ACD_TRANSCRIPT,
        file_directories=["AllCoreDevs-EL-Meetings", "AllCoreDevs-CL-Meetings"],
        file_extensions=[".md"],
        load_issues=True,
        issue_labels=["agenda"],
    ),
    "devp2p": RepoConfig(
        owner="ethereum",
        repo="devp2p",
        document_type=DocumentType.DEVP2P_SPEC,
        file_directories=None,  # All directories
        file_extensions=[".md"],
        load_issues=False,
    ),
}
```

---

### Phase 3: Create ingestion script

**File:** `scripts/ingest_github.py`

```python
#!/usr/bin/env python3
"""Ingest all configured GitHub repositories.

Usage:
    uv run python scripts/ingest_github.py
    uv run python scripts/ingest_github.py --repos EIPs,ERCs
    uv run python scripts/ingest_github.py --repos EIPs --no-cache
    uv run python scripts/ingest_github.py --repos pm --issues-only
"""

import argparse
import asyncio

import structlog
from dotenv import load_dotenv

from src.chunking import SectionChunker, convert_chunks
from src.embeddings import VoyageEmbedder
from src.ingestion.cache import RawContentCache
from src.ingestion.github_loader import (
    ETHEREUM_REPOS,
    DocumentType,
    GitHubDocument,
    GitHubLoader,
)
from src.storage import PgVectorStore

load_dotenv()
logger = structlog.get_logger()


def get_chunker_for_type(doc_type: DocumentType):
    """Get appropriate chunker for document type."""
    # All GitHub content is markdown-based
    return SectionChunker(max_tokens=512)


async def ingest_documents(
    documents: list[GitHubDocument],
    store: PgVectorStore,
    embedder: VoyageEmbedder,
) -> int:
    """Ingest documents into the database."""
    total_chunks = 0

    for doc in documents:
        chunker = get_chunker_for_type(doc.document_type)

        # Chunk the content
        chunks = chunker.chunk_text(doc.content, document_id=doc.document_id)
        if not chunks:
            continue

        # Embed
        embedded = embedder.embed_chunks(chunks)

        # Store document metadata
        await store.store_generic_document(
            document_id=doc.document_id,
            document_type=doc.document_type.value,
            title=doc.title,
            source=doc.source,
            raw_content=doc.content,
            metadata=doc.metadata,
        )

        # Store chunks
        await store.store_embedded_chunks(embedded)

        total_chunks += len(embedded)
        logger.debug(
            "ingested_document",
            doc_id=doc.document_id,
            type=doc.document_type.value,
            chunks=len(embedded),
        )

    return total_chunks


async def main(
    repos: list[str] | None = None,
    use_cache: bool = True,
    files_only: bool = False,
    issues_only: bool = False,
) -> None:
    """Main ingestion function."""
    cache = RawContentCache()
    loader = GitHubLoader(cache=cache)
    embedder = VoyageEmbedder()
    store = PgVectorStore()

    await store.connect()
    await store.initialize_schema()

    # Select repos to process
    repo_names = repos or list(ETHEREUM_REPOS.keys())

    try:
        for repo_name in repo_names:
            if repo_name not in ETHEREUM_REPOS:
                logger.warning("unknown_repo", repo=repo_name)
                continue

            config = ETHEREUM_REPOS[repo_name]
            logger.info("processing_repo", repo=repo_name)

            documents = []

            if not issues_only:
                documents.extend(
                    loader.load_repository_files(config, use_cache=use_cache)
                )

            if not files_only and config.load_issues:
                documents.extend(
                    loader.load_issues(config, use_cache=use_cache)
                )

            if documents:
                chunks = await ingest_documents(documents, store, embedder)
                logger.info(
                    "repo_complete",
                    repo=repo_name,
                    documents=len(documents),
                    chunks=chunks,
                )

    finally:
        await store.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest GitHub repositories")
    parser.add_argument(
        "--repos",
        type=str,
        help="Comma-separated list of repos (default: all)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching (force re-fetch)",
    )
    parser.add_argument(
        "--files-only",
        action="store_true",
        help="Only load files, skip issues",
    )
    parser.add_argument(
        "--issues-only",
        action="store_true",
        help="Only load issues, skip files",
    )

    args = parser.parse_args()

    repos = args.repos.split(",") if args.repos else None

    asyncio.run(main(
        repos=repos,
        use_cache=not args.no_cache,
        files_only=args.files_only,
        issues_only=args.issues_only,
    ))
```

---

### Phase 4: Update exports and integrate

**File:** `src/ingestion/__init__.py` (additions)

```python
from .github_loader import (
    DocumentType,
    ETHEREUM_REPOS,
    GitHubDocument,
    GitHubLoader,
    RepoConfig,
)

__all__ = [
    # ... existing exports
    "DocumentType",
    "ETHEREUM_REPOS",
    "GitHubDocument",
    "GitHubLoader",
    "RepoConfig",
]
```

---

### Phase 5: Migration from old loaders

The old loaders (`EIPLoader`, `ConsensusSpecLoader`, etc.) can be deprecated gradually:

1. Run new GitHub loader in parallel
2. Compare outputs
3. Remove old loaders once validated

**Migration script:**

```python
# scripts/migrate_to_github_loader.py

async def compare_loaders():
    """Compare old vs new loader outputs."""
    # Old loader
    old_loader = EIPLoader()
    old_loader.clone_or_update()
    old_eips = old_loader.load_all_eips()

    # New loader
    new_loader = GitHubLoader()
    config = ETHEREUM_REPOS["EIPs"]
    new_docs = new_loader.load_repository_files(config)

    # Compare
    old_ids = {f"eip-{e.eip_number}" for e in old_eips}
    new_ids = {d.document_id for d in new_docs}

    print(f"Old loader: {len(old_ids)} EIPs")
    print(f"New loader: {len(new_ids)} documents")
    print(f"Missing in new: {old_ids - new_ids}")
    print(f"Extra in new: {new_ids - old_ids}")
```

---

## Estimated content

| Repository | Files | Issues | Total Docs |
|------------|-------|--------|------------|
| ethereum/EIPs | ~900 | ~5,000 | ~5,900 |
| ethereum/ERCs | ~500 | ~2,000 | ~2,500 |
| ethereum/RIPs | ~20 | ~100 | ~120 |
| ethereum/consensus-specs | ~50 | 0 | ~50 |
| ethereum/execution-specs | ~600 | 0 | ~600 |
| ethereum/pm | ~300 | ~500 | ~800 |
| ethereum/devp2p | ~20 | 0 | ~20 |
| **Total** | **~2,390** | **~7,600** | **~9,990** |

This roughly **doubles** the corpus size by adding issue/PR discussions.

---

## Dependencies

```toml
# pyproject.toml additions
[project]
dependencies = [
    "llama-index-readers-github>=0.3.0",
]
```

---

## Environment variables

```bash
# .env additions
GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx
```

A GitHub personal access token with `repo` scope (for private repos) or just `public_repo` scope for public repos.

---

## Implementation order

1. [ ] Add `llama-index-readers-github` dependency
2. [ ] Create `src/ingestion/cache.py` (from raw-content-cache plan)
3. [ ] Create `src/ingestion/github_loader.py`
4. [ ] Update `src/ingestion/__init__.py` exports
5. [ ] Create `scripts/ingest_github.py`
6. [ ] Add `GITHUB_TOKEN` to `.env`
7. [ ] Test with single repo: `--repos EIPs`
8. [ ] Test caching: run twice, verify second is fast
9. [ ] Run full ingestion: all repos
10. [ ] Validate with test queries
11. [ ] Deprecate old loaders after validation

---

## Continuous ingestion

For continuous/incremental ingestion strategies covering all data sources (GitHub, arXiv, Discourse forums, ACD transcripts), see **[PLAN-continuous-ingestion.md](./PLAN-continuous-ingestion.md)**.

---

## Validation queries

After implementation, these queries should return results from GitHub issues:

```python
# Query about EIP discussions
"What were the main objections to EIP-4844 during the review process?"

# Query about meeting decisions
"When was the Shanghai upgrade date finalized in ACD calls?"

# Query about ERC rationale
"Why does ERC-721 require the receiver to implement onERC721Received?"
```
