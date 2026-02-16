"""Loader for ethereum/execution-specs repository."""

import subprocess
from dataclasses import dataclass
from pathlib import Path

import structlog

logger = structlog.get_logger()


@dataclass
class ExecutionSpec:
    """A parsed execution layer specification file."""

    module_path: str
    fork: str
    content: str
    file_path: Path


class ExecutionSpecLoader:
    """Load execution layer specs from ethereum/execution-specs repository."""

    REPO_URL = "https://github.com/ethereum/execution-specs.git"
    FORKS = [
        "frontier",
        "homestead",
        "tangerine_whistle",
        "spurious_dragon",
        "byzantium",
        "constantinople",
        "istanbul",
        "berlin",
        "london",
        "arrow_glacier",
        "gray_glacier",
        "paris",
        "shanghai",
        "cancun",
        "prague",
    ]

    def __init__(self, repo_path: str | Path = "data/execution-specs") -> None:
        self.repo_path = Path(repo_path)

    def clone_or_update(self) -> str:
        """Clone or update the execution-specs repository.

        Returns:
            Current git commit hash.
        """
        if not self.repo_path.exists():
            logger.info("cloning_execution_specs", url=self.REPO_URL)
            self.repo_path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["git", "clone", "--depth", "100", self.REPO_URL, str(self.repo_path)],
                check=True,
                capture_output=True,
            )
        else:
            logger.info("updating_execution_specs")
            subprocess.run(
                ["git", "-C", str(self.repo_path), "pull", "--ff-only"],
                check=True,
                capture_output=True,
            )

        result = subprocess.run(
            ["git", "-C", str(self.repo_path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    def load_all_specs(self) -> list[ExecutionSpec]:
        """Load all Python specification files from the repository."""
        specs: list[ExecutionSpec] = []

        forks_dir = self.repo_path / "src" / "ethereum" / "forks"
        if not forks_dir.exists():
            logger.warning("forks_dir_not_found", path=str(forks_dir))
            return specs

        for fork in self.FORKS:
            fork_dir = forks_dir / fork
            if not fork_dir.exists():
                continue

            for py_file in fork_dir.rglob("*.py"):
                # Skip __pycache__ and test files
                if "__pycache__" in str(py_file) or "test" in py_file.name.lower():
                    continue

                spec = self._parse_spec_file(py_file, fork)
                if spec:
                    specs.append(spec)

        logger.info("loaded_execution_specs", count=len(specs))
        return specs

    def _parse_spec_file(self, file_path: Path, fork: str) -> ExecutionSpec | None:
        """Parse a single Python specification file."""
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning("failed_to_read_spec", path=str(file_path), error=str(e))
            return None

        # Skip empty or very small files
        if len(content) < 100:
            return None

        # Build module path from file path
        rel_path = file_path.relative_to(self.repo_path / "src")
        module_path = str(rel_path).replace("/", ".").replace(".py", "")

        return ExecutionSpec(
            module_path=module_path,
            fork=fork,
            content=content,
            file_path=file_path,
        )
