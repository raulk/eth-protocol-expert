"""EIP Loader - Clone and load EIP markdown files from ethereum/EIPs repo."""

import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import structlog

logger = structlog.get_logger()


@dataclass
class LoadedEIP:
    """Raw EIP data loaded from disk."""
    eip_number: int
    file_path: Path
    raw_content: str
    git_commit: str
    loaded_at: datetime


class EIPLoader:
    """Load EIP markdown files from the ethereum/EIPs repository."""

    def __init__(self, data_dir: Path | str = "data/eips"):
        self.data_dir = Path(data_dir)
        self.repo_url = "https://github.com/ethereum/EIPs.git"

    def clone_or_update(self) -> str:
        """Clone the EIPs repo or pull latest changes. Returns current commit SHA."""
        if not self.data_dir.exists():
            logger.info("cloning_eips_repo", path=str(self.data_dir))
            subprocess.run(
                ["git", "clone", "--depth", "1", self.repo_url, str(self.data_dir)],
                check=True,
                capture_output=True,
            )
        else:
            logger.info("updating_eips_repo", path=str(self.data_dir))
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
            raise ValueError(f"EIPs repo not found at {self.data_dir}")

        result = subprocess.run(
            ["git", "-C", str(self.data_dir), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    def load_all_eips(self) -> list[LoadedEIP]:
        """Load all EIP markdown files."""
        eips_path = self.data_dir / "EIPS"
        if not eips_path.exists():
            raise ValueError(f"EIPS directory not found at {eips_path}")

        git_commit = self.get_current_commit()
        loaded_at = datetime.utcnow()
        eips = []

        for eip_file in sorted(eips_path.glob("eip-*.md")):
            try:
                eip_number = int(eip_file.stem.replace("eip-", ""))
                raw_content = eip_file.read_text(encoding="utf-8")

                eips.append(LoadedEIP(
                    eip_number=eip_number,
                    file_path=eip_file,
                    raw_content=raw_content,
                    git_commit=git_commit,
                    loaded_at=loaded_at,
                ))
            except (OSError, ValueError) as e:
                logger.warning("failed_to_load_eip", file=str(eip_file), error=str(e))

        logger.info("loaded_eips", count=len(eips))
        return eips

    def load_eip(self, eip_number: int) -> LoadedEIP | None:
        """Load a specific EIP by number."""
        eip_file = self.data_dir / "EIPS" / f"eip-{eip_number}.md"
        if not eip_file.exists():
            return None

        git_commit = self.get_current_commit()
        raw_content = eip_file.read_text(encoding="utf-8")

        return LoadedEIP(
            eip_number=eip_number,
            file_path=eip_file,
            raw_content=raw_content,
            git_commit=git_commit,
            loaded_at=datetime.utcnow(),
        )
