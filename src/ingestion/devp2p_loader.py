"""Loader for ethereum/devp2p specifications."""

from pathlib import Path

from .markdown_spec_loader import MarkdownSpec, MarkdownSpecLoader


class DevP2PLoader(MarkdownSpecLoader):
    """Load devp2p specs from ethereum/devp2p repository.

    The devp2p repository contains Ethereum peer-to-peer networking specifications
    including RLPx transport, node discovery, and protocol capabilities.
    """

    REPO_URL = "https://github.com/ethereum/devp2p.git"

    def __init__(self, repo_path: str | Path = "data/devp2p") -> None:
        super().__init__(
            repo_url=self.REPO_URL,
            repo_path=repo_path,
            source_name="devp2p",
            exclude_patterns=[
                "README.md",
                "CONTRIBUTING.md",
                "LICENSE*.md",
                ".github/**/*.md",
            ],
        )
