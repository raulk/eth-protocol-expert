"""Git Code Loader - Clone and track Ethereum client repositories."""

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import structlog

logger = structlog.get_logger()


@dataclass
class CodeRepository:
    """A cloned code repository."""

    name: str
    url: str
    local_path: Path
    current_commit: str
    language: str


class GitCodeLoader:
    """Clone and manage Ethereum client source code repositories."""

    DEFAULT_REPOS: ClassVar[dict[str, dict[str, str]]] = {
        "go-ethereum": {
            "url": "https://github.com/ethereum/go-ethereum",
            "language": "go",
        },
        "prysm": {
            "url": "https://github.com/prysmaticlabs/prysm",
            "language": "go",
        },
        "lighthouse": {
            "url": "https://github.com/sigp/lighthouse",
            "language": "rust",
        },
        "reth": {
            "url": "https://github.com/paradigmxyz/reth",
            "language": "rust",
        },
    }

    def __init__(self, cache_dir: str = "data/code"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def clone(self, url: str, name: str, language: str | None = None) -> CodeRepository:
        """Clone a repository or return existing clone."""
        repo_path = self.cache_dir / name

        if repo_path.exists():
            logger.info("repository_exists", name=name, path=str(repo_path))
            commit = self._get_current_commit(repo_path)
        else:
            logger.info("cloning_repository", name=name, url=url)
            subprocess.run(
                ["git", "clone", "--depth", "100", url, str(repo_path)],
                check=True,
                capture_output=True,
            )
            commit = self._get_current_commit(repo_path)
            logger.info("cloned_repository", name=name, commit=commit)

        detected_language = language or self._detect_language(repo_path)

        return CodeRepository(
            name=name,
            url=url,
            local_path=repo_path,
            current_commit=commit,
            language=detected_language,
        )

    def clone_default(self, name: str) -> CodeRepository:
        """Clone a pre-configured Ethereum client repository."""
        if name not in self.DEFAULT_REPOS:
            available = ", ".join(self.DEFAULT_REPOS.keys())
            raise ValueError(f"Unknown repository: {name}. Available: {available}")

        config = self.DEFAULT_REPOS[name]
        return self.clone(config["url"], name, config["language"])

    def update(self, repo: CodeRepository) -> str:
        """Pull latest changes and return new commit hash."""
        if not repo.local_path.exists():
            raise ValueError(f"Repository not found: {repo.local_path}")

        old_commit = repo.current_commit

        status = subprocess.run(
            ["git", "-C", str(repo.local_path), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        )
        if status.stdout.strip():
            raise ValueError(f"Repository has uncommitted changes: {repo.local_path}")

        branch = subprocess.run(
            ["git", "-C", str(repo.local_path), "rev-parse", "--abbrev-ref", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if branch == "HEAD":
            raise ValueError(f"Repository is in a detached HEAD state: {repo.local_path}")

        subprocess.run(
            ["git", "-C", str(repo.local_path), "fetch", "origin"],
            check=True,
            capture_output=True,
        )

        subprocess.run(
            ["git", "-C", str(repo.local_path), "merge", "--ff-only", f"origin/{branch}"],
            check=True,
            capture_output=True,
        )

        new_commit = self._get_current_commit(repo.local_path)

        if new_commit != old_commit:
            logger.info(
                "updated_repository",
                name=repo.name,
                old_commit=old_commit[:8],
                new_commit=new_commit[:8],
            )
        else:
            logger.debug("repository_up_to_date", name=repo.name, commit=new_commit[:8])

        return new_commit

    def list_files(self, repo: CodeRepository, pattern: str = "*.go") -> list[str]:
        """List files matching a glob pattern in the repository."""
        if not repo.local_path.exists():
            raise ValueError(f"Repository not found: {repo.local_path}")

        matching_files: list[str] = []

        for file_path in repo.local_path.rglob(pattern):
            if ".git" in file_path.parts:
                continue
            if "vendor" in file_path.parts:
                continue
            if "testdata" in file_path.parts:
                continue

            relative_path = str(file_path.relative_to(repo.local_path))
            matching_files.append(relative_path)

        logger.debug(
            "listed_files",
            repo=repo.name,
            pattern=pattern,
            count=len(matching_files),
        )

        return sorted(matching_files)

    def get_file_content(self, repo: CodeRepository, path: str) -> str:
        """Read the content of a file in the repository."""
        file_path = repo.local_path / path
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        return file_path.read_text(encoding="utf-8")

    def get_file_history(
        self,
        repo: CodeRepository,
        path: str,
        max_commits: int = 10,
    ) -> list[dict[str, str]]:
        """Get commit history for a specific file."""
        result = subprocess.run(
            [
                "git",
                "-C",
                str(repo.local_path),
                "log",
                f"-{max_commits}",
                "--pretty=format:%H|%an|%ae|%s|%ci",
                "--",
                path,
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        history: list[dict[str, str]] = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split("|")
            if len(parts) >= 5:
                history.append(
                    {
                        "commit": parts[0],
                        "author_name": parts[1],
                        "author_email": parts[2],
                        "message": parts[3],
                        "date": parts[4],
                    }
                )

        return history

    def _get_current_commit(self, repo_path: Path) -> str:
        result = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    def _detect_language(self, repo_path: Path) -> str:
        go_files = list(repo_path.rglob("*.go"))
        rust_files = list(repo_path.rglob("*.rs"))

        if len(go_files) > len(rust_files):
            return "go"
        elif len(rust_files) > len(go_files):
            return "rust"
        else:
            return "unknown"
