"""Loader for ethereum/builder-specs specifications."""

from pathlib import Path

from .markdown_spec_loader import MarkdownSpec, MarkdownSpecLoader


class BuilderSpecsLoader(MarkdownSpecLoader):
    """Load builder API specs from ethereum/builder-specs repository.

    The Builder API is an interface for consensus layer clients to source blocks
    built by external entities, implementing proposer-builder separation (PBS).
    """

    REPO_URL = "https://github.com/ethereum/builder-specs.git"

    def __init__(self, repo_path: str | Path = "data/builder-specs") -> None:
        super().__init__(
            repo_url=self.REPO_URL,
            repo_path=repo_path,
            source_name="builder-specs",
            exclude_patterns=[
                "README.md",
                "CONTRIBUTING.md",
                "LICENSE*.md",
                ".github/**/*.md",
            ],
        )
