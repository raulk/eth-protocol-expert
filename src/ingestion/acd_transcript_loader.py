"""ACD Transcript Loader - Load Ethereum AllCoreDevs call transcripts from ethereum/pm."""

import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import structlog

logger = structlog.get_logger()


@dataclass
class SpeakerSegment:
    """A segment of transcript attributed to a speaker."""

    speaker: str
    text: str
    timestamp: str | None = None


@dataclass
class ACDTranscript:
    """AllCoreDevs call transcript data."""

    call_number: int
    date: datetime | None
    title: str
    speakers: list[str]
    content: str
    raw_markdown: str
    segments: list[SpeakerSegment] = field(default_factory=list)
    file_path: Path | None = None


class ACDTranscriptLoader:
    """Load AllCoreDevs transcripts from ethereum/pm repository.

    The ethereum/pm repo contains meeting notes, agendas, and transcripts for
    various Ethereum governance calls including AllCoreDevs (ACD) calls.
    """

    REPO_URL = "https://github.com/ethereum/pm.git"

    def __init__(self, repo_path: str = "data/pm"):
        self.repo_path = Path(repo_path)

    def clone_repo(self) -> bool:
        """Clone or update the ethereum/pm repository."""
        if not self.repo_path.exists():
            logger.info("cloning_pm_repo", path=str(self.repo_path))
            try:
                subprocess.run(
                    ["git", "clone", "--depth", "1", self.REPO_URL, str(self.repo_path)],
                    check=True,
                    capture_output=True,
                )
                return True
            except subprocess.CalledProcessError as e:
                logger.error("failed_to_clone_pm_repo", error=e.stderr.decode())
                return False
        else:
            logger.info("updating_pm_repo", path=str(self.repo_path))
            try:
                subprocess.run(
                    ["git", "-C", str(self.repo_path), "pull", "--ff-only"],
                    check=True,
                    capture_output=True,
                )
                return True
            except subprocess.CalledProcessError as e:
                logger.warning("failed_to_update_pm_repo", error=e.stderr.decode())
                return True  # Repo exists, just couldn't update

    def list_transcripts(self) -> list[ACDTranscript]:
        """List all available ACD transcripts."""
        transcripts = []

        # ACD transcripts are in AllCoreDevs-EL-Meetings/ and AllCoreDevs-CL-Meetings/ directories
        acd_paths = [
            self.repo_path / "AllCoreDevs-EL-Meetings",
            self.repo_path / "AllCoreDevs-CL-Meetings",
            self.repo_path / "AllCoreDevs-Meetings",  # Legacy path
        ]

        found_any = False
        for acd_path in acd_paths:
            if not acd_path.exists():
                continue
            found_any = True

            # Find all markdown files
            for md_file in sorted(acd_path.glob("*.md")):
                transcript = self._parse_transcript_file(md_file)
                if transcript:
                    transcripts.append(transcript)

            # Also check subdirectories (some calls may be organized by year)
            for subdir in acd_path.iterdir():
                if subdir.is_dir():
                    for md_file in sorted(subdir.glob("*.md")):
                        transcript = self._parse_transcript_file(md_file)
                        if transcript:
                            transcripts.append(transcript)

        if not found_any:
            logger.warning("acd_directory_not_found", paths=[str(p) for p in acd_paths])

        logger.info("found_acd_transcripts", count=len(transcripts))
        return transcripts

    def load_transcript(self, path: str) -> ACDTranscript | None:
        """Load a specific transcript by path."""
        file_path = Path(path) if Path(path).is_absolute() else self.repo_path / path
        if not file_path.exists():
            logger.warning("transcript_not_found", path=str(file_path))
            return None
        return self._parse_transcript_file(file_path)

    def _parse_transcript_file(self, file_path: Path) -> ACDTranscript | None:
        """Parse a transcript markdown file."""
        try:
            raw_markdown = file_path.read_text(encoding="utf-8")
        except OSError as e:
            logger.warning("failed_to_read_transcript", path=str(file_path), error=str(e))
            return None

        # Extract call number from filename (e.g., "ACD-123.md" or "Meeting 123.md")
        call_number = self._extract_call_number(file_path.name)
        if call_number is None:
            # Skip non-transcript files like README.md
            return None

        # Parse metadata from content
        date = self._extract_date(raw_markdown)
        title = self._extract_title(raw_markdown, call_number)
        speakers = self._extract_speakers(raw_markdown)
        segments = self.extract_speaker_segments(raw_markdown)

        # Clean content (remove YAML frontmatter if present)
        content = self._clean_content(raw_markdown)

        return ACDTranscript(
            call_number=call_number,
            date=date,
            title=title,
            speakers=speakers,
            content=content,
            raw_markdown=raw_markdown,
            segments=segments,
            file_path=file_path,
        )

    def _extract_call_number(self, filename: str) -> int | None:
        """Extract ACD call number from filename."""
        # Common patterns: "ACD-123.md", "Meeting-123.md", "acd_123.md", "123.md"
        patterns = [
            r"ACD[_-]?(\d+)",
            r"Meeting[_-]?(\d+)",
            r"call[_-]?(\d+)",
            r"^(\d+)\.md$",
        ]

        for pattern in patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                return int(match.group(1))
        return None

    def _extract_date(self, content: str) -> datetime | None:
        """Extract date from transcript content or YAML frontmatter."""
        # Check YAML frontmatter
        frontmatter_match = re.search(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
        if frontmatter_match:
            yaml_content = frontmatter_match.group(1)
            date_match = re.search(r"date:\s*['\"]?(\d{4}-\d{2}-\d{2})", yaml_content)
            if date_match:
                try:
                    return datetime.strptime(date_match.group(1), "%Y-%m-%d")
                except ValueError:
                    pass

        # Look for date patterns in content
        date_patterns = [
            r"(\d{4}-\d{2}-\d{2})",  # ISO format
            r"(\d{1,2}/\d{1,2}/\d{4})",  # US format
            r"(\w+ \d{1,2},? \d{4})",  # Month Day, Year
        ]

        for pattern in date_patterns:
            match = re.search(pattern, content[:1000])  # Check first 1000 chars
            if match:
                date_str = match.group(1)
                for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%B %d, %Y", "%B %d %Y"]:
                    try:
                        return datetime.strptime(date_str, fmt)
                    except ValueError:
                        continue
        return None

    def _extract_title(self, content: str, call_number: int) -> str:
        """Extract title from transcript content."""
        # Check for first heading
        heading_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        if heading_match:
            return heading_match.group(1).strip()

        # Check YAML frontmatter for title
        frontmatter_match = re.search(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
        if frontmatter_match:
            yaml_content = frontmatter_match.group(1)
            title_match = re.search(r"title:\s*['\"]?(.+?)['\"]?\s*$", yaml_content, re.MULTILINE)
            if title_match:
                return title_match.group(1).strip()

        return f"AllCoreDevs Call #{call_number}"

    def _extract_speakers(self, content: str) -> list[str]:
        """Extract unique speaker names from transcript."""
        speakers = set()

        # Pattern for speaker attributions: "**Name:**", "Name:", "[Name]:"
        speaker_patterns = [
            r"\*\*([A-Za-z][A-Za-z\s\.]+?):\*\*",  # **Name:**
            r"^([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\s*:",  # Name: at line start
            r"\[([A-Za-z][A-Za-z\s\.]+?)\]:",  # [Name]:
        ]

        for pattern in speaker_patterns:
            for match in re.finditer(pattern, content, re.MULTILINE):
                speaker = match.group(1).strip()
                # Filter out common non-speaker patterns
                if speaker.lower() not in ["note", "notes", "link", "links", "action", "todo"]:
                    speakers.add(speaker)

        return sorted(speakers)

    def extract_speaker_segments(self, content: str) -> list[SpeakerSegment]:
        """Extract speaker-attributed segments from transcript content."""
        segments = []

        # Remove YAML frontmatter
        content = self._clean_content(content)

        # Pattern to match speaker lines with timestamps
        # Examples: "**Danny Ryan:** [00:05:23] Some text"
        #           "Tim Beiko: Some text"
        simple_pattern = re.compile(
            r"\*\*([A-Za-z][A-Za-z\s\.]+?):\*\*\s*(?:\[(\d{1,2}:\d{2}(?::\d{2})?)\])?\s*(.+?)(?=\*\*[A-Za-z]|\Z)",
            re.DOTALL,
        )

        matches = list(simple_pattern.finditer(content))
        if not matches:
            # Try line-by-line parsing as fallback
            return self._parse_segments_line_by_line(content)

        for match in matches:
            speaker = match.group(1).strip()
            timestamp = match.group(2)
            text = match.group(3).strip()

            if text and speaker.lower() not in ["note", "notes", "link", "action"]:
                segments.append(SpeakerSegment(speaker=speaker, text=text, timestamp=timestamp))

        return segments

    def _parse_segments_line_by_line(self, content: str) -> list[SpeakerSegment]:
        """Fallback line-by-line parsing for transcripts without bold speaker names."""
        segments = []
        current_speaker = None
        current_text = []
        current_timestamp = None

        for line in content.split("\n"):
            # Check for speaker line
            speaker_match = re.match(r"^([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\s*:\s*(.*)$", line)
            if speaker_match:
                # Save previous segment
                if current_speaker and current_text:
                    segments.append(
                        SpeakerSegment(
                            speaker=current_speaker,
                            text=" ".join(current_text).strip(),
                            timestamp=current_timestamp,
                        )
                    )
                current_speaker = speaker_match.group(1)
                current_text = [speaker_match.group(2)] if speaker_match.group(2) else []
                current_timestamp = None
            elif current_speaker:
                current_text.append(line)

        # Don't forget the last segment
        if current_speaker and current_text:
            segments.append(
                SpeakerSegment(
                    speaker=current_speaker,
                    text=" ".join(current_text).strip(),
                    timestamp=current_timestamp,
                )
            )

        return segments

    def _clean_content(self, content: str) -> str:
        """Remove YAML frontmatter and clean content."""
        # Remove YAML frontmatter
        content = re.sub(r"^---\s*\n.*?\n---\s*\n?", "", content, flags=re.DOTALL)
        return content.strip()
