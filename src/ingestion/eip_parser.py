"""EIP Parser - Extract frontmatter and sections from EIP markdown."""

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar

import structlog
import yaml

from .eip_loader import LoadedEIP

logger = structlog.get_logger()


@dataclass
class EIPSection:
    """A section within an EIP document."""
    name: str
    level: int  # Heading level (1-6)
    content: str
    start_offset: int
    end_offset: int
    subsections: list["EIPSection"] = field(default_factory=list)


@dataclass
class ParsedEIP:
    """Fully parsed EIP document."""
    eip_number: int
    title: str
    status: str
    type: str
    category: str | None
    author: str
    created: str
    requires: list[int]
    discussions_to: str | None

    # Full content
    raw_content: str
    abstract: str | None
    motivation: str | None
    specification: str | None
    rationale: str | None
    backwards_compatibility: str | None
    security_considerations: str | None

    # Structure
    sections: list[EIPSection]

    # Metadata
    git_commit: str
    loaded_at: datetime

    # All frontmatter fields (for extensibility)
    frontmatter: dict[str, Any] = field(default_factory=dict)


class EIPParser:
    """Parse EIP markdown files into structured data."""

    STANDARD_SECTIONS: ClassVar[list[str]] = [
        "abstract",
        "motivation",
        "specification",
        "rationale",
        "backwards compatibility",
        "test cases",
        "reference implementation",
        "security considerations",
        "copyright",
    ]

    def parse(self, loaded_eip: LoadedEIP) -> ParsedEIP:
        """Parse a loaded EIP into structured format."""
        content = loaded_eip.raw_content

        # Extract frontmatter
        frontmatter = self._extract_frontmatter(content)

        # Extract sections
        sections = self._extract_sections(content)

        # Get standard section contents
        section_map = {s.name.lower(): s.content for s in sections}

        return ParsedEIP(
            eip_number=loaded_eip.eip_number,
            title=frontmatter.get("title", ""),
            status=frontmatter.get("status", "Unknown"),
            type=frontmatter.get("type", "Unknown"),
            category=frontmatter.get("category"),
            author=frontmatter.get("author", ""),
            created=str(frontmatter.get("created", "")),
            requires=self._parse_requires(frontmatter.get("requires")),
            discussions_to=frontmatter.get("discussions-to"),
            raw_content=content,
            abstract=section_map.get("abstract"),
            motivation=section_map.get("motivation"),
            specification=section_map.get("specification"),
            rationale=section_map.get("rationale"),
            backwards_compatibility=section_map.get("backwards compatibility"),
            security_considerations=section_map.get("security considerations"),
            sections=sections,
            git_commit=loaded_eip.git_commit,
            loaded_at=loaded_eip.loaded_at,
            frontmatter=frontmatter,
        )

    def _extract_frontmatter(self, content: str) -> dict[str, Any]:
        """Extract YAML frontmatter from EIP content."""
        # EIP frontmatter is between --- markers
        match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
        if not match:
            return {}

        try:
            return yaml.safe_load(match.group(1)) or {}
        except yaml.YAMLError as e:
            logger.warning("failed_to_parse_frontmatter", error=str(e))
            return {}

    def _extract_sections(self, content: str) -> list[EIPSection]:
        """Extract markdown sections from content."""
        # Remove frontmatter
        content_without_frontmatter = re.sub(r"^---\n.*?\n---\n?", "", content, flags=re.DOTALL)

        sections = []
        # Match markdown headers
        header_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

        matches = list(header_pattern.finditer(content_without_frontmatter))

        for i, match in enumerate(matches):
            level = len(match.group(1))
            name = match.group(2).strip()
            start_offset = match.end() + 1  # After the newline

            # End is either the next header or end of content
            if i + 1 < len(matches):
                end_offset = matches[i + 1].start()
            else:
                end_offset = len(content_without_frontmatter)

            section_content = content_without_frontmatter[start_offset:end_offset].strip()

            sections.append(EIPSection(
                name=name,
                level=level,
                content=section_content,
                start_offset=start_offset,
                end_offset=end_offset,
            ))

        return sections

    def _parse_requires(self, requires_value: Any) -> list[int]:
        """Parse the 'requires' field into a list of EIP numbers."""
        if requires_value is None:
            return []

        if isinstance(requires_value, int):
            return [requires_value]

        if isinstance(requires_value, list):
            return [int(x) for x in requires_value if x is not None]

        if isinstance(requires_value, str):
            # Handle comma-separated or space-separated
            parts = re.split(r"[,\s]+", requires_value)
            result = []
            for part in parts:
                try:
                    result.append(int(part.strip()))
                except ValueError:
                    continue
            return result

        return []
