"""RIP Loader - Clone and load RIP markdown files from ethereum/RIPs repo."""

import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import structlog

logger = structlog.get_logger()


@dataclass
class LoadedRIP:
    """Raw RIP data loaded from disk."""

    rip_number: int
    file_path: Path
    raw_content: str
    git_commit: str
    loaded_at: datetime


class RIPLoader:
    """Load RIP markdown files from the ethereum/RIPs repository.

    RIPs (Rollup Improvement Proposals) are stored at https://github.com/ethereum/RIPs
    """

    def __init__(self, data_dir: Path | str = "data/rips"):
        self.data_dir = Path(data_dir)
        self.repo_url = "https://github.com/ethereum/RIPs.git"

    def clone_or_update(self) -> str:
        """Clone the RIPs repo or pull latest changes. Returns current commit SHA."""
        if not self.data_dir.exists():
            logger.info("cloning_rips_repo", path=str(self.data_dir))
            subprocess.run(
                ["git", "clone", "--depth", "1", self.repo_url, str(self.data_dir)],
                check=True,
                capture_output=True,
            )
        else:
            logger.info("updating_rips_repo", path=str(self.data_dir))
            subprocess.run(
                ["git", "-C", str(self.data_dir), "pull", "--ff-only"],
                check=True,
                capture_output=True,
            )

        result = subprocess.run(
            ["git", "-C", str(self.data_dir), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    def get_current_commit(self) -> str:
        """Get current git commit SHA."""
        if not self.data_dir.exists():
            raise ValueError(f"RIPs repo not found at {self.data_dir}")

        result = subprocess.run(
            ["git", "-C", str(self.data_dir), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    def load_all_rips(self) -> list[LoadedRIP]:
        """Load all RIP markdown files."""
        rips_path = self.data_dir / "RIPS"
        if not rips_path.exists():
            raise ValueError(f"RIPS directory not found at {rips_path}")

        git_commit = self.get_current_commit()
        loaded_at = datetime.utcnow()
        rips = []

        for rip_file in sorted(rips_path.glob("rip-*.md")):
            try:
                rip_number = int(rip_file.stem.replace("rip-", ""))
                raw_content = rip_file.read_text(encoding="utf-8")

                rips.append(
                    LoadedRIP(
                        rip_number=rip_number,
                        file_path=rip_file,
                        raw_content=raw_content,
                        git_commit=git_commit,
                        loaded_at=loaded_at,
                    )
                )
            except (OSError, ValueError) as e:
                logger.warning("failed_to_load_rip", file=str(rip_file), error=str(e))

        logger.info("loaded_rips", count=len(rips))
        return rips

    def load_rip(self, rip_number: int) -> LoadedRIP | None:
        """Load a specific RIP by number."""
        rip_file = self.data_dir / "RIPS" / f"rip-{rip_number}.md"
        if not rip_file.exists():
            return None

        git_commit = self.get_current_commit()
        raw_content = rip_file.read_text(encoding="utf-8")

        return LoadedRIP(
            rip_number=rip_number,
            file_path=rip_file,
            raw_content=raw_content,
            git_commit=git_commit,
            loaded_at=datetime.utcnow(),
        )
