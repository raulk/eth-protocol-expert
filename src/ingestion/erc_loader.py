"""ERC Loader - Load ERCs from both ethereum/EIPs and ethereum/ERCs repos."""

import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import structlog

logger = structlog.get_logger()


@dataclass
class LoadedERC:
    """Raw ERC data loaded from disk."""

    erc_number: int
    file_path: Path
    raw_content: str
    git_commit: str
    loaded_at: datetime
    source_repo: str  # 'EIPs' or 'ERCs'


class ERCLoader:
    """Load ERC markdown files from both ethereum/EIPs and ethereum/ERCs repositories.

    ERCs in the EIPs repo have category: ERC in their frontmatter.
    ERCs in the ERCs repo are dedicated ERC files.

    When duplicates exist, the ERCs repo version takes precedence (newer).
    """

    def __init__(
        self,
        eips_data_dir: Path | str = "data/eips",
        ercs_data_dir: Path | str = "data/ercs",
    ):
        self.eips_data_dir = Path(eips_data_dir)
        self.ercs_data_dir = Path(ercs_data_dir)
        self.eips_repo_url = "https://github.com/ethereum/EIPs.git"
        self.ercs_repo_url = "https://github.com/ethereum/ERCs.git"

    def clone_or_update(self) -> dict[str, str]:
        """Clone/update both repos. Returns dict of repo -> commit SHA."""
        commits = {}

        # Clone/update EIPs repo
        if not self.eips_data_dir.exists():
            logger.info("cloning_eips_repo", path=str(self.eips_data_dir))
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    self.eips_repo_url,
                    str(self.eips_data_dir),
                ],
                check=True,
                capture_output=True,
            )
        else:
            logger.info("updating_eips_repo", path=str(self.eips_data_dir))
            subprocess.run(
                ["git", "-C", str(self.eips_data_dir), "pull", "--ff-only"],
                check=True,
                capture_output=True,
            )

        result = subprocess.run(
            ["git", "-C", str(self.eips_data_dir), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        commits["EIPs"] = result.stdout.strip()

        # Clone/update ERCs repo
        if not self.ercs_data_dir.exists():
            logger.info("cloning_ercs_repo", path=str(self.ercs_data_dir))
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    self.ercs_repo_url,
                    str(self.ercs_data_dir),
                ],
                check=True,
                capture_output=True,
            )
        else:
            logger.info("updating_ercs_repo", path=str(self.ercs_data_dir))
            subprocess.run(
                ["git", "-C", str(self.ercs_data_dir), "pull", "--ff-only"],
                check=True,
                capture_output=True,
            )

        result = subprocess.run(
            ["git", "-C", str(self.ercs_data_dir), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        commits["ERCs"] = result.stdout.strip()

        return commits

    def get_current_commits(self) -> dict[str, str]:
        """Get current git commit SHAs for both repos."""
        commits = {}

        if self.eips_data_dir.exists():
            result = subprocess.run(
                ["git", "-C", str(self.eips_data_dir), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            )
            commits["EIPs"] = result.stdout.strip()

        if self.ercs_data_dir.exists():
            result = subprocess.run(
                ["git", "-C", str(self.ercs_data_dir), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            )
            commits["ERCs"] = result.stdout.strip()

        return commits

    def _is_erc(self, content: str) -> bool:
        """Check if content has category: ERC in frontmatter."""
        lines = content.split("\n")
        in_frontmatter = False

        for line in lines:
            stripped = line.strip()
            if stripped == "---":
                if in_frontmatter:
                    break
                in_frontmatter = True
                continue

            if in_frontmatter and stripped.lower().startswith("category:"):
                category = stripped.split(":", 1)[1].strip().lower()
                return category == "erc"

        return False

    def _load_ercs_from_eips_repo(self, commits: dict[str, str]) -> dict[int, LoadedERC]:
        """Load ERCs from the EIPs repo (files with category: ERC)."""
        ercs: dict[int, LoadedERC] = {}
        eips_path = self.eips_data_dir / "EIPS"

        if not eips_path.exists():
            logger.warning("eips_directory_not_found", path=str(eips_path))
            return ercs

        git_commit = commits.get("EIPs", "unknown")
        loaded_at = datetime.utcnow()

        for eip_file in sorted(eips_path.glob("eip-*.md")):
            try:
                raw_content = eip_file.read_text(encoding="utf-8")

                if not self._is_erc(raw_content):
                    continue

                erc_number = int(eip_file.stem.replace("eip-", ""))

                ercs[erc_number] = LoadedERC(
                    erc_number=erc_number,
                    file_path=eip_file,
                    raw_content=raw_content,
                    git_commit=git_commit,
                    loaded_at=loaded_at,
                    source_repo="EIPs",
                )
            except (OSError, ValueError) as e:
                logger.warning("failed_to_load_erc", file=str(eip_file), error=str(e))

        logger.info("loaded_ercs_from_eips", count=len(ercs))
        return ercs

    def _load_ercs_from_ercs_repo(self, commits: dict[str, str]) -> dict[int, LoadedERC]:
        """Load ERCs from the dedicated ERCs repo."""
        ercs: dict[int, LoadedERC] = {}
        ercs_path = self.ercs_data_dir / "ERCS"

        if not ercs_path.exists():
            logger.warning("ercs_directory_not_found", path=str(ercs_path))
            return ercs

        git_commit = commits.get("ERCs", "unknown")
        loaded_at = datetime.utcnow()

        for erc_file in sorted(ercs_path.glob("erc-*.md")):
            try:
                erc_number = int(erc_file.stem.replace("erc-", ""))
                raw_content = erc_file.read_text(encoding="utf-8")

                ercs[erc_number] = LoadedERC(
                    erc_number=erc_number,
                    file_path=erc_file,
                    raw_content=raw_content,
                    git_commit=git_commit,
                    loaded_at=loaded_at,
                    source_repo="ERCs",
                )
            except (OSError, ValueError) as e:
                logger.warning("failed_to_load_erc", file=str(erc_file), error=str(e))

        logger.info("loaded_ercs_from_ercs_repo", count=len(ercs))
        return ercs

    def load_all_ercs(self) -> list[LoadedERC]:
        """Load all ERCs from both repos, deduplicating (ERCs repo takes precedence)."""
        commits = self.get_current_commits()

        # Load from EIPs repo first
        ercs_from_eips = self._load_ercs_from_eips_repo(commits)

        # Load from ERCs repo (will override duplicates)
        ercs_from_ercs = self._load_ercs_from_ercs_repo(commits)

        # Merge: ERCs repo takes precedence for duplicates
        merged = ercs_from_eips.copy()
        duplicates = 0
        for erc_number, erc in ercs_from_ercs.items():
            if erc_number in merged:
                duplicates += 1
            merged[erc_number] = erc

        ercs = list(merged.values())
        ercs.sort(key=lambda e: e.erc_number)

        logger.info(
            "loaded_all_ercs",
            total=len(ercs),
            from_eips=len(ercs_from_eips),
            from_ercs=len(ercs_from_ercs),
            duplicates_resolved=duplicates,
        )
        return ercs

    def load_erc(self, erc_number: int) -> LoadedERC | None:
        """Load a specific ERC by number (ERCs repo takes precedence)."""
        commits = self.get_current_commits()

        # Check ERCs repo first
        erc_file = self.ercs_data_dir / "ERCS" / f"erc-{erc_number}.md"
        if erc_file.exists():
            raw_content = erc_file.read_text(encoding="utf-8")
            return LoadedERC(
                erc_number=erc_number,
                file_path=erc_file,
                raw_content=raw_content,
                git_commit=commits.get("ERCs", "unknown"),
                loaded_at=datetime.utcnow(),
                source_repo="ERCs",
            )

        # Fall back to EIPs repo
        eip_file = self.eips_data_dir / "EIPS" / f"eip-{erc_number}.md"
        if eip_file.exists():
            raw_content = eip_file.read_text(encoding="utf-8")
            if self._is_erc(raw_content):
                return LoadedERC(
                    erc_number=erc_number,
                    file_path=eip_file,
                    raw_content=raw_content,
                    git_commit=commits.get("EIPs", "unknown"),
                    loaded_at=datetime.utcnow(),
                    source_repo="EIPs",
                )

        return None
