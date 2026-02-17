#!/usr/bin/env python3
"""Ingest Ethereum-related arXiv papers.

Usage:
    uv run python scripts/ingest_arxiv.py [--max-papers 300]
"""

import argparse
import asyncio

import structlog
from dotenv import load_dotenv
load_dotenv()  # Must run before any src.* imports

from src.chunking import PaperChunker, convert_chunks
from src.embeddings import create_embedder
from src.ingestion import ArxivFetcher, PDFExtractor, QualityScorer
from src.storage import PgVectorStore

logger = structlog.get_logger()


async def ingest_arxiv(max_papers: int = 300) -> None:
    """Ingest arXiv papers into the database."""
    fetcher = ArxivFetcher()
    extractor = PDFExtractor()
    scorer = QualityScorer()
    embedder = create_embedder()
    store = PgVectorStore()
    await store.connect()
    await store.initialize_schema()

    chunker = PaperChunker(max_tokens=512)
    papers_ingested = 0
    papers_skipped = 0

    try:
        papers = fetcher.search_ethereum_papers(max_results=max_papers)
        logger.info("found_papers", count=len(papers))

        for paper in papers:
            if not paper.pdf_url:
                logger.debug("skipping_paper_no_pdf", arxiv_id=paper.arxiv_id)
                papers_skipped += 1
                continue

            document_id = f"arxiv-{paper.arxiv_id}"

            try:
                pdf_content = extractor.extract_from_url(paper.pdf_url)

                quality = scorer.score(pdf_content)
                if not scorer.is_acceptable(quality, threshold=0.5):
                    logger.warning(
                        "low_quality_pdf",
                        arxiv_id=paper.arxiv_id,
                        score=round(quality.overall, 3),
                        issues=quality.issues[:3],
                    )
                    papers_skipped += 1
                    continue

            except Exception as e:
                logger.warning("pdf_extraction_failed", arxiv_id=paper.arxiv_id, error=str(e))
                papers_skipped += 1
                continue

            paper_chunks = chunker.chunk(pdf_content)
            if not paper_chunks:
                logger.debug("no_chunks_generated", arxiv_id=paper.arxiv_id)
                papers_skipped += 1
                continue

            standard_chunks = convert_chunks(paper_chunks, document_id)

            embedded = embedder.embed_chunks(standard_chunks)

            await store.store_generic_document(
                document_id=document_id,
                document_type="arxiv_paper",
                title=paper.title,
                source="arxiv",
                author=", ".join(paper.authors),
                raw_content=pdf_content.full_text,
                metadata={
                    "arxiv_id": paper.arxiv_id,
                    "categories": paper.categories,
                    "published": paper.published.isoformat() if paper.published else None,
                    "doi": paper.doi,
                    "abstract": paper.abstract,
                    "primary_category": paper.primary_category,
                    "quality_score": round(quality.overall, 3),
                },
            )

            await store.store_embedded_chunks(embedded)

            papers_ingested += 1
            logger.info(
                "ingested_paper",
                arxiv_id=paper.arxiv_id,
                title=paper.title[:50],
                chunks=len(embedded),
                quality=round(quality.overall, 3),
            )

        # Rebuild vector index after bulk insert
        await store.reindex_embeddings()

    finally:
        fetcher.close()
        extractor.close()
        await store.close()

    logger.info(
        "arxiv_ingestion_complete",
        ingested=papers_ingested,
        skipped=papers_skipped,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest arXiv papers")
    parser.add_argument("--max-papers", type=int, default=300)
    args = parser.parse_args()

    asyncio.run(ingest_arxiv(max_papers=args.max_papers))


if __name__ == "__main__":
    main()
