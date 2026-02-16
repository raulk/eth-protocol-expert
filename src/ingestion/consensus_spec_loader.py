"""Loader for ethereum/consensus-specs repository."""

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import structlog

logger = structlog.get_logger()


@dataclass
class ConsensusSpec:
    """A parsed consensus layer specification document."""

    name: str
    fork: str
    title: str
    content: str
    file_path: Path


class ConsensusSpecLoader:
    """Load consensus layer specs from ethereum/consensus-specs repository."""

    REPO_URL = "https://github.com/ethereum/consensus-specs.git"

    def __init__(self, repo_path: str | Path = "data/consensus-specs") -> None:
        self.repo_path = Path(repo_path)

    def clone_or_update(self) -> str:
        """Clone or update the consensus-specs repository.

        Returns:
            Current git commit hash.
        """
        if not self.repo_path.exists():
            logger.info("cloning_consensus_specs", url=self.REPO_URL)
            self.repo_path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["git", "clone", "--depth", "100", self.REPO_URL, str(self.repo_path)],
                check=True,
                capture_output=True,
            )
        else:
            logger.info("updating_consensus_specs")
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

    def _discover_spec_dirs(self) -> list[tuple[Path, str]]:
        """Discover all spec directories dynamically.

        Returns list of (directory_path, fork_name) tuples.
        Includes main forks and _features subdirectories.
        """
        specs_dir = self.repo_path / "specs"
        if not specs_dir.exists():
            return []

        result: list[tuple[Path, str]] = []

        for item in specs_dir.iterdir():
            if not item.is_dir():
                continue

            if item.name.startswith("."):
                continue

            if item.name == "_features":
                # Index each feature subdirectory with "feature-" prefix
                for feature_dir in item.iterdir():
                    if feature_dir.is_dir() and not feature_dir.name.startswith("."):
                        result.append((feature_dir, f"feature-{feature_dir.name}"))
            else:
                result.append((item, item.name))

        return result

    def load_all_specs(self) -> list[ConsensusSpec]:
        """Load all specification files from the repository."""
        specs: list[ConsensusSpec] = []

        specs_dir = self.repo_path / "specs"
        if not specs_dir.exists():
            logger.warning("specs_dir_not_found", path=str(specs_dir))
            return specs

        spec_dirs = self._discover_spec_dirs()
        logger.info("discovered_spec_dirs", count=len(spec_dirs))

        for fork_dir, fork_name in spec_dirs:
            for md_file in fork_dir.glob("*.md"):
                spec = self._parse_spec_file(md_file, fork_name)
                if spec:
                    specs.append(spec)
                    logger.debug("loaded_spec", name=spec.name, fork=fork_name)

        logger.info("loaded_consensus_specs", count=len(specs))
        return specs

    def _parse_spec_file(self, file_path: Path, fork: str) -> ConsensusSpec | None:
        """Parse a single specification markdown file."""
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning("failed_to_read_spec", path=str(file_path), error=str(e))
            return None

        # Extract title from first heading
        title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        title = title_match.group(1) if title_match else file_path.stem

        name = file_path.stem  # e.g., "beacon-chain", "fork-choice"

        return ConsensusSpec(
            name=name,
            fork=fork,
            title=title,
            content=content,
            file_path=file_path,
        )
