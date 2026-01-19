"""Citation Graph - Track paper citations in FalkorDB."""

import re
from dataclasses import dataclass, field

import structlog

from .falkordb_store import FalkorDBStore, QueryResult

logger = structlog.get_logger()


@dataclass
class CitationEdge:
    """A citation relationship between papers."""

    citing_paper: str  # Paper ID that contains the citation
    cited_paper: str  # Paper ID being cited
    context: str | None = None  # Surrounding text where citation appears
    citation_number: int | None = None  # Reference number in citing paper


@dataclass
class PaperNode:
    """A paper node in the citation graph."""

    paper_id: str
    title: str
    authors: list[str] = field(default_factory=list)
    arxiv_id: str | None = None
    year: int | None = None
    venue: str | None = None


class CitationGraph:
    """Manage paper citations in FalkorDB.

    Stores papers and their citation relationships alongside the EIP graph.
    Enables queries like "what papers cite this paper" and "what papers
    does this paper cite".
    """

    def __init__(self, store: FalkorDBStore | None = None, graph_name: str = "citation_graph"):
        """Initialize citation graph.

        Args:
            store: FalkorDBStore instance (shared with EIP graph if desired)
            graph_name: Name for the citation graph
        """
        if store:
            self._store = store
        else:
            self._store = FalkorDBStore(graph_name=graph_name)

        self._initialized = False

    def connect(self):
        """Connect to FalkorDB and initialize schema."""
        self._store.connect()
        self._initialize_schema()
        self._initialized = True

    def close(self):
        """Close connection."""
        self._store.close()
        self._initialized = False

    def _initialize_schema(self):
        """Create indexes for efficient citation queries."""
        try:
            self._store.query("CREATE INDEX FOR (p:Paper) ON (p.paper_id)")
            logger.info("created_paper_id_index")
        except Exception as e:
            if "already indexed" not in str(e).lower():
                logger.warning("failed_to_create_index", error=str(e))

        try:
            self._store.query("CREATE INDEX FOR (p:Paper) ON (p.arxiv_id)")
            logger.info("created_arxiv_id_index")
        except Exception as e:
            if "already indexed" not in str(e).lower():
                logger.warning("failed_to_create_index", error=str(e))

    def add_paper(
        self,
        paper_id: str,
        title: str,
        references: list[str] | None = None,
        authors: list[str] | None = None,
        arxiv_id: str | None = None,
        year: int | None = None,
    ) -> QueryResult:
        """Add a paper to the graph with its references.

        Args:
            paper_id: Unique identifier for the paper
            title: Paper title
            references: List of reference strings (will attempt to extract paper IDs)
            authors: List of author names
            arxiv_id: arXiv identifier if available
            year: Publication year

        Returns:
            QueryResult from the creation operation
        """
        if not self._initialized:
            self.connect()

        # Create the paper node
        cypher = """
            MERGE (p:Paper {paper_id: $paper_id})
            ON CREATE SET
                p.title = $title,
                p.authors = $authors,
                p.arxiv_id = $arxiv_id,
                p.year = $year
            ON MATCH SET
                p.title = $title,
                p.authors = $authors,
                p.arxiv_id = $arxiv_id,
                p.year = $year
            RETURN p
        """
        result = self._store.query(
            cypher,
            params={
                "paper_id": paper_id,
                "title": title,
                "authors": authors or [],
                "arxiv_id": arxiv_id,
                "year": year,
            },
        )

        # Process references if provided
        if references:
            self._add_references(paper_id, references)

        logger.debug("added_paper", paper_id=paper_id, references=len(references or []))
        return result

    def _add_references(self, citing_paper: str, references: list[str]):
        """Add reference relationships from a paper to its citations."""
        for ref in references:
            # Try to extract an identifier from the reference
            cited_id = self._extract_paper_id(ref)
            if cited_id:
                self._create_citation_edge(citing_paper, cited_id, context=ref[:200])

    def _extract_paper_id(self, reference: str) -> str | None:
        """Extract a paper identifier from a reference string."""
        # Try to find arXiv ID
        arxiv_match = re.search(r"arXiv[:\s]*(\d{4}\.\d{4,5}(?:v\d+)?)", reference, re.IGNORECASE)
        if arxiv_match:
            return f"arxiv:{arxiv_match.group(1)}"

        # Try to find DOI
        doi_match = re.search(r"10\.\d{4,}/[^\s]+", reference)
        if doi_match:
            return f"doi:{doi_match.group(0)}"

        # Generate a hash-based ID from the reference text
        # Normalize and hash first ~100 chars of reference
        normalized = re.sub(r"\s+", " ", reference[:100].lower().strip())
        if len(normalized) > 20:
            import hashlib

            ref_hash = hashlib.md5(normalized.encode()).hexdigest()[:12]
            return f"ref:{ref_hash}"

        return None

    def _create_citation_edge(
        self, citing_paper: str, cited_paper: str, context: str | None = None
    ):
        """Create a CITES relationship between papers."""
        cypher = """
            MERGE (cited:Paper {paper_id: $cited_paper})
            WITH cited
            MATCH (citing:Paper {paper_id: $citing_paper})
            MERGE (citing)-[r:CITES]->(cited)
            ON CREATE SET r.context = $context
            RETURN citing, r, cited
        """
        self._store.query(
            cypher,
            params={
                "citing_paper": citing_paper,
                "cited_paper": cited_paper,
                "context": context,
            },
        )

    def add_citation(self, edge: CitationEdge) -> QueryResult:
        """Add a single citation relationship.

        Args:
            edge: CitationEdge with citing and cited paper IDs

        Returns:
            QueryResult from the operation
        """
        if not self._initialized:
            self.connect()

        cypher = """
            MERGE (citing:Paper {paper_id: $citing})
            MERGE (cited:Paper {paper_id: $cited})
            MERGE (citing)-[r:CITES]->(cited)
            ON CREATE SET
                r.context = $context,
                r.citation_number = $citation_number
            RETURN citing, r, cited
        """
        return self._store.query(
            cypher,
            params={
                "citing": edge.citing_paper,
                "cited": edge.cited_paper,
                "context": edge.context,
                "citation_number": edge.citation_number,
            },
        )

    def get_citing(self, paper_id: str) -> list[str]:
        """Get papers that this paper cites (outgoing citations).

        Args:
            paper_id: The paper to get citations from

        Returns:
            List of paper IDs that are cited by this paper
        """
        if not self._initialized:
            self.connect()

        result = self._store.query(
            """
            MATCH (p:Paper {paper_id: $paper_id})-[:CITES]->(cited:Paper)
            RETURN cited.paper_id
            """,
            params={"paper_id": paper_id},
        )
        return [row[0] for row in result.result_set]

    def get_cited_by(self, paper_id: str) -> list[str]:
        """Get papers that cite this paper (incoming citations).

        Args:
            paper_id: The paper to get citations for

        Returns:
            List of paper IDs that cite this paper
        """
        if not self._initialized:
            self.connect()

        result = self._store.query(
            """
            MATCH (citing:Paper)-[:CITES]->(p:Paper {paper_id: $paper_id})
            RETURN citing.paper_id
            """,
            params={"paper_id": paper_id},
        )
        return [row[0] for row in result.result_set]

    def get_citation_count(self, paper_id: str) -> dict[str, int]:
        """Get citation counts for a paper.

        Args:
            paper_id: The paper to get counts for

        Returns:
            Dict with 'cites' (outgoing) and 'cited_by' (incoming) counts
        """
        if not self._initialized:
            self.connect()

        # Outgoing citations
        cites_result = self._store.query(
            "MATCH (p:Paper {paper_id: $id})-[:CITES]->() RETURN count(*)",
            params={"id": paper_id},
        )
        cites = cites_result.result_set[0][0] if cites_result.result_set else 0

        # Incoming citations
        cited_by_result = self._store.query(
            "MATCH ()-[:CITES]->(p:Paper {paper_id: $id}) RETURN count(*)",
            params={"id": paper_id},
        )
        cited_by = cited_by_result.result_set[0][0] if cited_by_result.result_set else 0

        return {"cites": cites, "cited_by": cited_by}

    def get_paper(self, paper_id: str) -> PaperNode | None:
        """Get paper details by ID.

        Args:
            paper_id: The paper ID to look up

        Returns:
            PaperNode or None if not found
        """
        if not self._initialized:
            self.connect()

        result = self._store.query(
            "MATCH (p:Paper {paper_id: $id}) RETURN p",
            params={"id": paper_id},
        )

        if not result.result_set:
            return None

        node = result.result_set[0][0]
        props = dict(node.properties)

        return PaperNode(
            paper_id=props.get("paper_id", paper_id),
            title=props.get("title", ""),
            authors=props.get("authors", []),
            arxiv_id=props.get("arxiv_id"),
            year=props.get("year"),
            venue=props.get("venue"),
        )

    def find_common_citations(self, paper_ids: list[str]) -> list[str]:
        """Find papers cited by all given papers.

        Args:
            paper_ids: List of paper IDs

        Returns:
            List of paper IDs cited by all input papers
        """
        if not self._initialized:
            self.connect()

        if not paper_ids:
            return []

        cypher = """
            MATCH (p:Paper)-[:CITES]->(common:Paper)
            WHERE p.paper_id IN $paper_ids
            WITH common, count(DISTINCT p) as cite_count
            WHERE cite_count = $total
            RETURN common.paper_id
        """
        result = self._store.query(cypher, params={"paper_ids": paper_ids, "total": len(paper_ids)})
        return [row[0] for row in result.result_set]

    def get_citation_context(self, citing: str, cited: str) -> str | None:
        """Get the context text where a citation appears.

        Args:
            citing: Paper ID of the citing paper
            cited: Paper ID of the cited paper

        Returns:
            Context string or None if not found
        """
        if not self._initialized:
            self.connect()

        result = self._store.query(
            """
            MATCH (a:Paper {paper_id: $citing})-[r:CITES]->(b:Paper {paper_id: $cited})
            RETURN r.context
            """,
            params={"citing": citing, "cited": cited},
        )

        if result.result_set:
            return result.result_set[0][0]
        return None

    def count_papers(self) -> int:
        """Count total papers in the graph."""
        if not self._initialized:
            self.connect()

        result = self._store.query("MATCH (p:Paper) RETURN count(p)")
        return result.result_set[0][0] if result.result_set else 0

    def count_citations(self) -> int:
        """Count total citation relationships."""
        if not self._initialized:
            self.connect()

        result = self._store.query("MATCH ()-[r:CITES]->() RETURN count(r)")
        return result.result_set[0][0] if result.result_set else 0

    def clear_graph(self):
        """Delete all paper nodes and citations (for testing)."""
        self._store.query("MATCH (p:Paper) DETACH DELETE p")
        logger.warning("cleared_citation_graph")
