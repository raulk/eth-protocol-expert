"""Text Chunker - Chunk generic markdown/text documents."""

import hashlib
import re

import structlog
import tiktoken

from .fixed_chunker import Chunk

logger = structlog.get_logger()


class TextChunker:
    """Chunk arbitrary text into token-bounded chunks.

    Uses paragraph-aware splitting to avoid mid-sentence breaks where possible.
    """

    def __init__(
        self,
        max_tokens: int = 512,
        overlap_tokens: int = 64,
        encoding_name: str = "cl100k_base",
    ) -> None:
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.tokenizer = tiktoken.get_encoding(encoding_name)

    def chunk_text(
        self,
        text: str,
        document_id: str,
        *,
        section_path: str | None = None,
    ) -> list[Chunk]:
        """Chunk text into a list of Chunks."""
        if not text or not text.strip():
            return []

        tokens = self.tokenizer.encode(text)
        if len(tokens) <= self.max_tokens:
            return [
                Chunk(
                    chunk_id=self._hash_content(text),
                    document_id=document_id,
                    content=text.strip(),
                    token_count=len(tokens),
                    chunk_index=0,
                    section_path=section_path,
                )
            ]

        paragraphs = re.split(r"\n\n+", text)
        chunks: list[Chunk] = []
        current: list[str] = []
        current_tokens = 0
        chunk_index = 0

        for para in paragraphs:
            if not para.strip():
                continue
            para_tokens = len(self.tokenizer.encode(para))

            if current and current_tokens + para_tokens > self.max_tokens:
                chunk_text = "\n\n".join(current)
                chunks.append(
                    Chunk(
                        chunk_id=self._hash_content(chunk_text),
                        document_id=document_id,
                        content=chunk_text.strip(),
                        token_count=len(self.tokenizer.encode(chunk_text)),
                        chunk_index=chunk_index,
                        section_path=section_path,
                    )
                )
                chunk_index += 1
                current = [para]
                current_tokens = para_tokens
                continue

            current.append(para)
            current_tokens += para_tokens

        if current:
            chunk_text = "\n\n".join(current)
            chunks.append(
                Chunk(
                    chunk_id=self._hash_content(chunk_text),
                    document_id=document_id,
                    content=chunk_text.strip(),
                    token_count=len(self.tokenizer.encode(chunk_text)),
                    chunk_index=chunk_index,
                    section_path=section_path,
                )
            )

        logger.debug("chunked_text", document_id=document_id, chunks=len(chunks))
        return chunks

    def _hash_content(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
