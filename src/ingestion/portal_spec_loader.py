"""Loader for ethereum/portal-network-specs specifications."""

from pathlib import Path

from .markdown_spec_loader import MarkdownSpecLoader


class PortalSpecLoader(MarkdownSpecLoader):
    """Load Portal Network specs from ethereum/portal-network-specs repository.

    The Portal Network enables lightweight protocol access by resource-constrained
    devices through multiple peer-to-peer networks providing data access via JSON-RPC.
    """

    REPO_URL = "https://github.com/ethereum/portal-network-specs.git"

    def __init__(self, repo_path: str | Path = "data/portal-network-specs") -> None:
        super().__init__(
            repo_url=self.REPO_URL,
            repo_path=repo_path,
            source_name="portal-network-specs",
            exclude_patterns=[
                "README.md",
                "CONTRIBUTING.md",
                "LICENSE*.md",
                ".github/**/*.md",
            ],
        )
