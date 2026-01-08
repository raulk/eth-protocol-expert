"""PDF Extractor - Extract text and structure from PDF documents using PyMuPDF."""

import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import httpx
import structlog

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

logger = structlog.get_logger()


@dataclass
class PDFSection:
    """A section extracted from a PDF document."""

    heading: str
    content: str
    page_numbers: list[int] = field(default_factory=list)
    level: int = 1


@dataclass
class PDFContent:
    """Extracted content from a PDF document."""

    title: str
    sections: list[PDFSection]
    references: list[str]
    metadata: dict[str, str | None] = field(default_factory=dict)
    page_count: int = 0
    full_text: str = ""


class PDFExtractor:
    """Extract structured content from PDF documents.

    Uses PyMuPDF (fitz) for efficient PDF parsing and text extraction.
    Attempts to preserve document structure including sections and references.
    """

    # Common section headings in academic papers
    SECTION_PATTERNS: ClassVar[list[str]] = [
        r"^\s*(?:\d+\.?\s+)?(Abstract)\s*$",
        r"^\s*(?:\d+\.?\s+)?(Introduction)\s*$",
        r"^\s*(?:\d+\.?\s+)?(Background)\s*$",
        r"^\s*(?:\d+\.?\s+)?(Related Work)\s*$",
        r"^\s*(?:\d+\.?\s+)?(Methodology|Methods)\s*$",
        r"^\s*(?:\d+\.?\s+)?(System Design|Design)\s*$",
        r"^\s*(?:\d+\.?\s+)?(Implementation)\s*$",
        r"^\s*(?:\d+\.?\s+)?(Evaluation|Experiments)\s*$",
        r"^\s*(?:\d+\.?\s+)?(Results)\s*$",
        r"^\s*(?:\d+\.?\s+)?(Discussion)\s*$",
        r"^\s*(?:\d+\.?\s+)?(Conclusion|Conclusions)\s*$",
        r"^\s*(?:\d+\.?\s+)?(References|Bibliography)\s*$",
        r"^\s*(?:\d+\.?\s+)?(Appendix|Appendices)\s*$",
        r"^\s*(?:\d+\.?\s+)?(Acknowledgments?)\s*$",
    ]

    def __init__(self, timeout: float = 60.0, max_pdf_size_bytes: int = 25 * 1024 * 1024):
        if fitz is None:
            raise ImportError("PyMuPDF (fitz) is required. Install with: pip install PyMuPDF")
        self.timeout = timeout
        self.max_pdf_size_bytes = max_pdf_size_bytes
        self._http_client: httpx.Client | None = None

    def _get_http_client(self) -> httpx.Client:
        if self._http_client is None:
            self._http_client = httpx.Client(timeout=self.timeout)
        return self._http_client

    def close(self):
        """Close HTTP client."""
        if self._http_client:
            self._http_client.close()
            self._http_client = None

    def extract(self, pdf_path: str | Path) -> PDFContent:
        """Extract structured content from a PDF file.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            PDFContent with extracted sections and metadata
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info("extracting_pdf", path=str(pdf_path))

        doc = fitz.open(pdf_path)
        try:
            return self._extract_from_doc(doc)
        finally:
            doc.close()

    def extract_from_url(self, url: str) -> PDFContent:
        """Extract content from a PDF at a URL.

        Downloads the PDF to a temporary file before extraction.

        Args:
            url: URL to the PDF file

        Returns:
            PDFContent with extracted sections and metadata
        """
        logger.info("downloading_pdf", url=url)

        try:
            with self._get_http_client().stream("GET", url, follow_redirects=True) as response:
                response.raise_for_status()
                content_type = response.headers.get("Content-Type", "").lower()
                if content_type and "pdf" not in content_type:
                    raise ValueError(f"Unexpected content type: {content_type}")

                content_length = response.headers.get("Content-Length")
                if content_length and int(content_length) > self.max_pdf_size_bytes:
                    raise ValueError(
                        f"PDF exceeds size limit ({content_length} bytes > {self.max_pdf_size_bytes})"
                    )

                # Write to temp file while enforcing size cap
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    total_bytes = 0
                    for chunk in response.iter_bytes(1024 * 64):
                        if not chunk:
                            continue
                        total_bytes += len(chunk)
                        if total_bytes > self.max_pdf_size_bytes:
                            raise ValueError(
                                f"PDF exceeds size limit ({total_bytes} bytes > {self.max_pdf_size_bytes})"
                            )
                        tmp.write(chunk)
                    tmp_path = Path(tmp.name)
        except (httpx.HTTPError, ValueError) as e:
            logger.error("pdf_download_failed", url=url, error=str(e))
            raise

        try:
            return self.extract(tmp_path)
        finally:
            tmp_path.unlink()

    def _extract_from_doc(self, doc: "fitz.Document") -> PDFContent:
        """Extract content from an open PyMuPDF document."""
        # Extract metadata
        metadata = self._extract_metadata(doc)

        # Extract full text page by page
        pages_text = []
        for page_num, page in enumerate(doc):
            text = page.get_text()
            pages_text.append((page_num + 1, text))

        full_text = "\n\n".join(text for _, text in pages_text)

        # Extract title (from metadata or first page)
        title = metadata.get("title", "")
        if not title:
            title = self._extract_title_from_text(pages_text[0][1] if pages_text else "")

        # Extract sections
        sections = self._extract_sections(pages_text)

        # Extract references
        references = self._extract_references(full_text)

        return PDFContent(
            title=title,
            sections=sections,
            references=references,
            metadata=metadata,
            page_count=len(doc),
            full_text=full_text,
        )

    def _extract_metadata(self, doc: "fitz.Document") -> dict[str, str | None]:
        """Extract PDF metadata."""
        meta = doc.metadata
        return {
            "title": meta.get("title"),
            "author": meta.get("author"),
            "subject": meta.get("subject"),
            "keywords": meta.get("keywords"),
            "creator": meta.get("creator"),
            "producer": meta.get("producer"),
            "creation_date": meta.get("creationDate"),
            "mod_date": meta.get("modDate"),
        }

    def _extract_title_from_text(self, first_page_text: str) -> str:
        """Extract title from first page text."""
        lines = first_page_text.strip().split("\n")

        # Usually the title is one of the first few non-empty lines
        for line in lines[:10]:
            line = line.strip()
            # Skip very short lines or lines that look like metadata
            if len(line) > 10 and not re.match(r"^(arXiv:|http|www\.|[0-9]+$)", line):
                return line

        return "Untitled"

    def extract_sections(self, doc: "fitz.Document") -> list[PDFSection]:
        """Extract sections from a PyMuPDF document (public method for external use)."""
        pages_text = []
        for page_num, page in enumerate(doc):
            text = page.get_text()
            pages_text.append((page_num + 1, text))
        return self._extract_sections(pages_text)

    def _extract_sections(self, pages_text: list[tuple[int, str]]) -> list[PDFSection]:
        """Extract sections from page text tuples."""
        sections = []
        current_section: PDFSection | None = None
        current_content: list[str] = []
        current_pages: list[int] = []

        # Combine patterns into single regex
        section_regex = re.compile("|".join(self.SECTION_PATTERNS), re.IGNORECASE | re.MULTILINE)

        for page_num, text in pages_text:
            lines = text.split("\n")

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Check if this line is a section heading
                match = section_regex.match(line)
                if match:
                    # Save previous section
                    if current_section:
                        current_section.content = "\n".join(current_content).strip()
                        current_section.page_numbers = current_pages
                        sections.append(current_section)

                    # Find which group matched for the heading name
                    heading = next((g for g in match.groups() if g), line)
                    current_section = PDFSection(heading=heading, content="", level=1)
                    current_content = []
                    current_pages = [page_num]
                else:
                    current_content.append(line)
                    if current_pages and page_num not in current_pages:
                        current_pages.append(page_num)

        # Don't forget the last section
        if current_section:
            current_section.content = "\n".join(current_content).strip()
            current_section.page_numbers = current_pages
            sections.append(current_section)

        # If no sections found, create one section with all content
        if not sections:
            all_text = "\n".join(text for _, text in pages_text)
            sections.append(
                PDFSection(
                    heading="Content",
                    content=all_text.strip(),
                    page_numbers=list(range(1, len(pages_text) + 1)),
                )
            )

        return sections

    def extract_references(self, doc_or_text: "fitz.Document | str") -> list[str]:
        """Extract references from document (public method for external use)."""
        if isinstance(doc_or_text, str):
            text = doc_or_text
        else:
            text = "\n".join(page.get_text() for page in doc_or_text)
        return self._extract_references(text)

    def _extract_references(self, text: str) -> list[str]:
        """Extract reference entries from the full text."""
        references = []

        # Find the references section
        ref_match = re.search(
            r"(?:^|\n)\s*(?:References|Bibliography)\s*\n(.*?)(?:\n\s*(?:Appendix|$))",
            text,
            re.IGNORECASE | re.DOTALL,
        )

        if not ref_match:
            return references

        ref_text = ref_match.group(1)

        # Try to split references by common patterns
        # Pattern 1: [1], [2], etc.
        numbered_refs = re.split(r"\n\s*\[\d+\]\s*", ref_text)
        if len(numbered_refs) > 2:
            for ref in numbered_refs[1:]:  # Skip first empty split
                ref = ref.strip()
                if ref and len(ref) > 10:
                    references.append(ref)
            return references

        # Pattern 2: 1., 2., etc. at line start
        numbered_refs = re.split(r"\n\s*\d+\.\s+", ref_text)
        if len(numbered_refs) > 2:
            for ref in numbered_refs[1:]:
                ref = ref.strip()
                if ref and len(ref) > 10:
                    references.append(ref)
            return references

        # Fallback: split by double newlines or recognize author patterns
        entries = re.split(r"\n\n+", ref_text)
        for entry in entries:
            entry = entry.strip()
            if entry and len(entry) > 20:
                references.append(entry)

        return references
