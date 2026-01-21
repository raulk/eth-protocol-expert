"""Generic loader for markdown specification repositories."""

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import structlog

logger = structlog.get_logger()


@dataclass
class MarkdownSpec:
    """A parsed markdown specification document."""

    name: str
    title: str
    content: str
    file_path: Path
    category: str | None = None  # Optional subdirectory/category


class MarkdownSpecLoader:
    """Generic loader for markdown specification repositories.

    Clones a git repository and loads all .md files, extracting title
    from the first heading.
    """

    def __init__(
        self,
        repo_url: str,
        repo_path: str | Path,
        source_name: str,
        exclude_patterns: list[str] | None = None,
    ) -> None:
        """Initialize the markdown spec loader.

        Args:
            repo_url: Git URL to clone from
            repo_path: Local path to clone/store the repo
            source_name: Name for logging and document source field
            exclude_patterns: List of glob patterns to exclude (e.g., ["README.md"])
        """
        self.repo_url = repo_url
        self.repo_path = Path(repo_path)
        self.source_name = source_name
        self.exclude_patterns = exclude_patterns or []

    def clone_or_update(self) -> str:
        """Clone or update the repository.

        Returns:
            Current git commit hash.
        """
        if not self.repo_path.exists():
            logger.info(
                "cloning_repo", source=self.source_name, url=self.repo_url
            )
            self.repo_path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["git", "clone", "--depth", "1", self.repo_url, str(self.repo_path)],
                check=True,
                capture_output=True,
            )
        else:
            logger.info("updating_repo", source=self.source_name)
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

    def get_current_commit(self) -> str:
        """Get current git commit SHA."""
        if not self.repo_path.exists():
            raise ValueError(f"Repo not found at {self.repo_path}")

        result = subprocess.run(
            ["git", "-C", str(self.repo_path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    def _should_exclude(self, file_path: Path) -> bool:
        """Check if file should be excluded based on patterns."""
        for pattern in self.exclude_patterns:
            if file_path.match(pattern):
                return True
        return False

    def _extract_title(self, content: str, file_path: Path) -> str:
        """Extract title from markdown content."""
        # Try to find first heading
        title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        if title_match:
            return title_match.group(1).strip()

        # Fall back to filename
        return file_path.stem.replace("-", " ").replace("_", " ").title()

    def _get_category(self, file_path: Path) -> str | None:
        """Get category from subdirectory path."""
        try:
            rel_path = file_path.relative_to(self.repo_path)
            parts = rel_path.parts[:-1]  # Exclude filename
            if parts:
                return "/".join(parts)
        except ValueError:
            pass
        return None

    def load_all_specs(self) -> list[MarkdownSpec]:
        """Load all markdown files from the repository."""
        specs: list[MarkdownSpec] = []

        if not self.repo_path.exists():
            logger.warning("repo_not_found", path=str(self.repo_path))
            return specs

        for md_file in self.repo_path.rglob("*.md"):
            if self._should_exclude(md_file):
                logger.debug("excluding_file", path=str(md_file))
                continue

            spec = self._parse_spec_file(md_file)
            if spec:
                specs.append(spec)

        logger.info(f"loaded_{self.source_name}_specs", count=len(specs))
        return specs

    def _parse_spec_file(self, file_path: Path) -> MarkdownSpec | None:
        """Parse a single markdown file."""
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(
                "failed_to_read_spec",
                source=self.source_name,
                path=str(file_path),
                error=str(e),
            )
            return None

        # Skip very short files (likely placeholders)
        if len(content.strip()) < 50:
            return None

        title = self._extract_title(content, file_path)
        category = self._get_category(file_path)
        name = file_path.stem

        return MarkdownSpec(
            name=name,
            title=title,
            content=content,
            file_path=file_path,
            category=category,
        )
