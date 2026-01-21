"""Loader for ethereum/research repository.

The research repository contains Python implementations and documentation
of cryptographic research concepts by Vitalik Buterin and others, including:
- STARK implementations
- Binary field operations
- Erasure coding
- Verkle tries
- Consensus simulations
"""

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import structlog

logger = structlog.get_logger()

REPO_URL = "https://github.com/ethereum/research.git"
DEFAULT_PATH = Path(__file__).parent.parent.parent / "data" / "research"


@dataclass
class ResearchDoc:
    """A document from the research repository."""

    name: str
    title: str
    content: str
    file_path: Path
    category: str | None = None
    doc_type: str = "markdown"  # "markdown" or "python"


class ResearchLoader:
    """Loader for ethereum/research repository.

    Loads both markdown documentation and Python code with docstrings,
    extracting research concepts and implementations.
    """

    def __init__(
        self,
        repo_path: str | Path = DEFAULT_PATH,
        include_python: bool = True,
    ) -> None:
        """Initialize the research loader.

        Args:
            repo_path: Local path to clone/store the repo
            include_python: Whether to include Python files (default True)
        """
        self.repo_url = REPO_URL
        self.repo_path = Path(repo_path)
        self.include_python = include_python
        self.exclude_patterns = [
            "**/test*.py",
            "**/__pycache__/**",
            "**/.git/**",
            "**/setup.py",
            "**/__init__.py",
        ]

    def clone_or_update(self) -> str:
        """Clone or update the repository.

        Returns:
            Current git commit hash.
        """
        if not self.repo_path.exists():
            logger.info("cloning_research_repo", url=self.repo_url)
            self.repo_path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["git", "clone", "--depth", "1", self.repo_url, str(self.repo_path)],
                check=True,
                capture_output=True,
            )
        else:
            logger.info("updating_research_repo")
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

    def _get_category(self, file_path: Path) -> str | None:
        """Get category from subdirectory path."""
        try:
            rel_path = file_path.relative_to(self.repo_path)
            parts = rel_path.parts[:-1]
            if parts:
                return "/".join(parts)
        except ValueError:
            pass
        return None

    def _extract_title_from_markdown(self, content: str, file_path: Path) -> str:
        """Extract title from markdown content."""
        title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        if title_match:
            return title_match.group(1).strip()
        return file_path.stem.replace("-", " ").replace("_", " ").title()

    def _extract_title_from_python(self, content: str, file_path: Path) -> str:
        """Extract title from Python file docstring or first comment."""
        # Try module docstring
        docstring_match = re.match(r'^["\'][\'"]{2}(.+?)["\'][\'"]{2}', content, re.DOTALL)
        if docstring_match:
            first_line = docstring_match.group(1).strip().split("\n")[0]
            if first_line:
                return first_line[:100]

        # Try first comment block
        comment_match = re.match(r"^#\s*(.+)$", content, re.MULTILINE)
        if comment_match:
            return comment_match.group(1).strip()[:100]

        # Fall back to filename
        return file_path.stem.replace("_", " ").title()

    def _extract_python_documentation(self, content: str) -> str:
        """Extract documentation from Python file.

        Combines module docstring, function docstrings, and inline comments
        to create searchable documentation.
        """
        parts = []

        # Extract module docstring
        docstring_match = re.match(r'^["\'][\'"]{2}(.+?)["\'][\'"]{2}', content, re.DOTALL)
        if docstring_match:
            parts.append(docstring_match.group(1).strip())

        # Extract function/class definitions with docstrings
        func_pattern = re.compile(
            r'(?:^|\n)((?:def|class)\s+\w+[^:]*:)\s*\n\s*["\'][\'"]{2}(.+?)["\'][\'"]{2}',
            re.DOTALL,
        )
        for match in func_pattern.finditer(content):
            signature = match.group(1).strip()
            docstring = match.group(2).strip()
            parts.append(f"{signature}\n{docstring}")

        # Extract significant comment blocks (3+ lines of comments)
        comment_blocks = []
        current_block = []
        for line in content.split("\n"):
            stripped = line.strip()
            if stripped.startswith("#") and not stripped.startswith("#!"):
                current_block.append(stripped[1:].strip())
            else:
                if len(current_block) >= 3:
                    comment_blocks.append("\n".join(current_block))
                current_block = []
        if len(current_block) >= 3:
            comment_blocks.append("\n".join(current_block))

        parts.extend(comment_blocks)

        if not parts:
            return content

        return "\n\n".join(parts)

    def _parse_markdown_file(self, file_path: Path) -> ResearchDoc | None:
        """Parse a markdown file."""
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning("failed_to_read_file", path=str(file_path), error=str(e))
            return None

        if len(content.strip()) < 50:
            return None

        title = self._extract_title_from_markdown(content, file_path)
        category = self._get_category(file_path)

        return ResearchDoc(
            name=file_path.stem,
            title=title,
            content=content,
            file_path=file_path,
            category=category,
            doc_type="markdown",
        )

    def _parse_python_file(self, file_path: Path) -> ResearchDoc | None:
        """Parse a Python file and extract documentation."""
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning("failed_to_read_file", path=str(file_path), error=str(e))
            return None

        if len(content.strip()) < 100:
            return None

        title = self._extract_title_from_python(content, file_path)
        category = self._get_category(file_path)

        # For Python files, include both the extracted documentation
        # and the full source code for reference
        documentation = self._extract_python_documentation(content)

        # Create a combined content with documentation first, then code
        combined_content = f"""# {title}

## Documentation

{documentation}

## Source Code

```python
{content}
```
"""

        return ResearchDoc(
            name=file_path.stem,
            title=title,
            content=combined_content,
            file_path=file_path,
            category=category,
            doc_type="python",
        )

    def load_all_docs(self) -> list[ResearchDoc]:
        """Load all documents from the repository."""
        docs: list[ResearchDoc] = []

        if not self.repo_path.exists():
            logger.warning("repo_not_found", path=str(self.repo_path))
            return docs

        # Load markdown files
        for md_file in self.repo_path.rglob("*.md"):
            if self._should_exclude(md_file):
                continue

            doc = self._parse_markdown_file(md_file)
            if doc:
                docs.append(doc)

        # Load Python files if enabled
        if self.include_python:
            for py_file in self.repo_path.rglob("*.py"):
                if self._should_exclude(py_file):
                    continue

                doc = self._parse_python_file(py_file)
                if doc:
                    docs.append(doc)

        logger.info("loaded_research_docs", count=len(docs))
        return docs

    def load_markdown_only(self) -> list[ResearchDoc]:
        """Load only markdown documentation files."""
        docs: list[ResearchDoc] = []

        if not self.repo_path.exists():
            logger.warning("repo_not_found", path=str(self.repo_path))
            return docs

        for md_file in self.repo_path.rglob("*.md"):
            if self._should_exclude(md_file):
                continue

            doc = self._parse_markdown_file(md_file)
            if doc:
                docs.append(doc)

        logger.info("loaded_research_markdown_docs", count=len(docs))
        return docs

    def load_python_only(self) -> list[ResearchDoc]:
        """Load only Python implementation files."""
        docs: list[ResearchDoc] = []

        if not self.repo_path.exists():
            logger.warning("repo_not_found", path=str(self.repo_path))
            return docs

        for py_file in self.repo_path.rglob("*.py"):
            if self._should_exclude(py_file):
                continue

            doc = self._parse_python_file(py_file)
            if doc:
                docs.append(doc)

        logger.info("loaded_research_python_docs", count=len(docs))
        return docs
