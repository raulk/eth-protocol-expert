"""ArXiv Fetcher - Fetch and parse academic papers from arXiv API."""

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar

import httpx
import structlog

logger = structlog.get_logger()


@dataclass
class ArxivPaper:
    """Academic paper metadata from arXiv."""

    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    categories: list[str]
    published: datetime
    updated: datetime | None = None
    pdf_url: str | None = None
    primary_category: str | None = None
    doi: str | None = None
    comment: str | None = None
    journal_ref: str | None = None


class ArxivFetcher:
    """Fetch papers from arXiv API.

    Uses the arXiv Atom API to search and retrieve paper metadata.
    Rate limited to 1 request per 3 seconds per arXiv guidelines.
    """

    BASE_URL = "https://export.arxiv.org/api/query"
    ATOM_NS: ClassVar[dict[str, str]] = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }

    # Ethereum-related search terms
    ETHEREUM_KEYWORDS: ClassVar[list[str]] = [
        "ethereum",
        "smart contract",
        "blockchain consensus",
        "proof of stake",
        "EVM",
        "solidity",
        "merkle patricia trie",
        "blob transaction",
        "data availability",
        "rollup",
        "layer 2 blockchain",
    ]

    # Relevant arXiv categories
    RELEVANT_CATEGORIES: ClassVar[list[str]] = [
        "cs.CR",  # Cryptography and Security
        "cs.DC",  # Distributed Computing
        "cs.NI",  # Networking and Internet Architecture
        "cs.SE",  # Software Engineering
        "cs.PL",  # Programming Languages
    ]

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        self._client: httpx.Client | None = None

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def close(self):
        """Close the HTTP client."""
        if self._client:
            self._client.close()
            self._client = None

    def search(self, query: str, max_results: int = 100) -> list[ArxivPaper]:
        """Search arXiv for papers matching the query.

        Args:
            query: Search query (supports arXiv search syntax)
            max_results: Maximum number of results to return

        Returns:
            List of ArxivPaper objects
        """
        params = {
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }

        logger.info("searching_arxiv", query=query, max_results=max_results)

        try:
            response = self._get_client().get(self.BASE_URL, params=params)
            response.raise_for_status()
        except httpx.HTTPError as e:
            logger.error("arxiv_search_failed", error=str(e))
            return []

        papers = self._parse_response(response.text)
        logger.info("arxiv_search_complete", results=len(papers))
        return papers

    def fetch_by_id(self, arxiv_id: str) -> ArxivPaper | None:
        """Fetch a specific paper by arXiv ID.

        Args:
            arxiv_id: arXiv paper ID (e.g., "2301.00001" or "2301.00001v1")

        Returns:
            ArxivPaper object or None if not found
        """
        # Clean the ID (remove version suffix for search)
        clean_id = re.sub(r"v\d+$", "", arxiv_id)

        params = {"id_list": clean_id, "max_results": 1}

        try:
            response = self._get_client().get(self.BASE_URL, params=params)
            response.raise_for_status()
        except httpx.HTTPError as e:
            logger.error("arxiv_fetch_failed", arxiv_id=arxiv_id, error=str(e))
            return None

        papers = self._parse_response(response.text)
        return papers[0] if papers else None

    def search_ethereum_papers(self, max_results: int = 100) -> list[ArxivPaper]:
        """Search for Ethereum and blockchain-related papers.

        Searches across relevant categories with Ethereum-specific keywords.

        Returns:
            List of ArxivPaper objects related to Ethereum
        """
        all_papers: dict[str, ArxivPaper] = {}

        # Build category filter
        cat_query = " OR ".join(f"cat:{cat}" for cat in self.RELEVANT_CATEGORIES)

        # Search for each keyword combination
        for keyword in self.ETHEREUM_KEYWORDS[:5]:  # Limit to avoid rate limiting
            query = f"all:{keyword} AND ({cat_query})"
            papers = self.search(query, max_results=max_results // len(self.ETHEREUM_KEYWORDS[:5]))

            for paper in papers:
                if paper.arxiv_id not in all_papers:
                    all_papers[paper.arxiv_id] = paper

        logger.info("ethereum_paper_search_complete", total_unique=len(all_papers))
        return list(all_papers.values())

    def search_by_category(
        self, categories: list[str] | None = None, max_results: int = 100
    ) -> list[ArxivPaper]:
        """Search for papers in specific arXiv categories.

        Args:
            categories: List of arXiv category codes (default: RELEVANT_CATEGORIES)
            max_results: Maximum results to return

        Returns:
            List of ArxivPaper objects
        """
        cats = categories or self.RELEVANT_CATEGORIES
        query = " OR ".join(f"cat:{cat}" for cat in cats)
        return self.search(query, max_results)

    def _parse_response(self, xml_content: str) -> list[ArxivPaper]:
        """Parse arXiv Atom XML response into ArxivPaper objects."""
        papers = []

        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            logger.error("failed_to_parse_arxiv_xml", error=str(e))
            return papers

        # Find all entry elements
        for entry in root.findall("atom:entry", self.ATOM_NS):
            paper = self._parse_entry(entry)
            if paper:
                papers.append(paper)

        return papers

    def _parse_entry(self, entry: ET.Element) -> ArxivPaper | None:
        """Parse a single Atom entry into an ArxivPaper."""
        try:
            # Extract ID
            id_elem = entry.find("atom:id", self.ATOM_NS)
            if id_elem is None or id_elem.text is None:
                return None
            arxiv_id = self._extract_arxiv_id(id_elem.text)

            # Extract title
            title_elem = entry.find("atom:title", self.ATOM_NS)
            title = self._clean_text(title_elem.text) if title_elem is not None else ""

            # Extract abstract (summary)
            abstract_elem = entry.find("atom:summary", self.ATOM_NS)
            abstract = self._clean_text(abstract_elem.text) if abstract_elem is not None else ""

            # Extract authors
            authors = []
            for author in entry.findall("atom:author", self.ATOM_NS):
                name_elem = author.find("atom:name", self.ATOM_NS)
                if name_elem is not None and name_elem.text:
                    authors.append(name_elem.text)

            # Extract categories
            categories = []
            primary_category = None
            for category in entry.findall("arxiv:primary_category", self.ATOM_NS):
                term = category.get("term")
                if term:
                    primary_category = term
                    categories.append(term)
            for category in entry.findall("atom:category", self.ATOM_NS):
                term = category.get("term")
                if term and term not in categories:
                    categories.append(term)

            # Extract dates
            published_elem = entry.find("atom:published", self.ATOM_NS)
            published = (
                self._parse_date(published_elem.text) if published_elem is not None else None
            )
            if published is None:
                return None  # Published date is required

            updated_elem = entry.find("atom:updated", self.ATOM_NS)
            updated = self._parse_date(updated_elem.text) if updated_elem is not None else None

            # Extract PDF URL
            pdf_url = None
            for link in entry.findall("atom:link", self.ATOM_NS):
                if link.get("title") == "pdf":
                    pdf_url = link.get("href")
                    break

            # Extract optional fields
            doi_elem = entry.find("arxiv:doi", self.ATOM_NS)
            doi = doi_elem.text if doi_elem is not None else None

            comment_elem = entry.find("arxiv:comment", self.ATOM_NS)
            comment = comment_elem.text if comment_elem is not None else None

            journal_elem = entry.find("arxiv:journal_ref", self.ATOM_NS)
            journal_ref = journal_elem.text if journal_elem is not None else None

            return ArxivPaper(
                arxiv_id=arxiv_id,
                title=title,
                authors=authors,
                abstract=abstract,
                categories=categories,
                published=published,
                updated=updated,
                pdf_url=pdf_url,
                primary_category=primary_category,
                doi=doi,
                comment=comment,
                journal_ref=journal_ref,
            )

        except Exception as e:
            logger.warning("failed_to_parse_arxiv_entry", error=str(e))
            return None

    def _extract_arxiv_id(self, url: str) -> str:
        """Extract arXiv ID from the full URL."""
        # URL format: http://arxiv.org/abs/2301.00001v1
        match = re.search(r"arxiv\.org/abs/(.+)$", url)
        return match.group(1) if match else url

    def _clean_text(self, text: str | None) -> str:
        """Clean whitespace from text."""
        if not text:
            return ""
        # Normalize whitespace
        return " ".join(text.split())

    def _parse_date(self, date_str: str | None) -> datetime | None:
        """Parse arXiv date string to datetime."""
        if not date_str:
            return None
        try:
            # arXiv uses ISO format: 2023-01-15T12:34:56Z
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except ValueError:
            return None
