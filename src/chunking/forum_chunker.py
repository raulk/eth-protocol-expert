"""Forum Chunker - Chunk forum posts respecting thread structure."""

import hashlib
import re

import structlog
import tiktoken

from ..ingestion.ethresearch_loader import LoadedForumPost, LoadedForumTopic
from .fixed_chunker import Chunk

logger = structlog.get_logger()


class ForumChunker:
    """Chunk forum posts while respecting thread structure.

    Key design decisions:
    - Each post is chunked independently (reply boundaries are preserved)
    - Long posts are split at paragraph boundaries
    - Post metadata (author, reply context) is included in chunks
    - Code blocks are kept atomic when possible
    """

    def __init__(
        self,
        max_tokens: int = 512,
        overlap_tokens: int = 64,
        encoding_name: str = "cl100k_base",
    ):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.tokenizer = tiktoken.get_encoding(encoding_name)

    def chunk_topic(self, topic: LoadedForumTopic) -> list[Chunk]:
        """Chunk all posts in a forum topic."""
        chunks = []
        chunk_index = 0

        # Create a topic header chunk
        header = self._create_topic_header(topic, chunk_index)
        chunks.append(header)
        chunk_index += 1

        # Chunk each post
        for post in topic.posts:
            post_chunks = self._chunk_post(post, topic, chunk_index)
            chunks.extend(post_chunks)
            chunk_index += len(post_chunks)

        logger.debug(
            "chunked_forum_topic",
            source=topic.source,
            topic_id=topic.topic_id,
            posts=len(topic.posts),
            chunks=len(chunks),
        )
        return chunks

    def chunk_post(self, post: LoadedForumPost) -> list[Chunk]:
        """Chunk a single post (when processing individually)."""
        # Create a minimal topic context for standalone processing
        return self._chunk_post(post, topic=None, start_index=0)

    def _create_topic_header(
        self, topic: LoadedForumTopic, chunk_index: int
    ) -> Chunk:
        """Create a header chunk with topic metadata."""
        document_id = self._make_document_id(topic.source, topic.topic_id)

        parts = [
            f"# {topic.title}",
            "",
            f"**Forum**: {topic.source}",
            f"**Topic ID**: {topic.topic_id}",
        ]

        if topic.category:
            parts.append(f"**Category**: {topic.category}")
        if topic.tags:
            parts.append(f"**Tags**: {', '.join(topic.tags)}")

        parts.extend([
            f"**Posts**: {topic.posts_count}",
            f"**Created**: {topic.created_at.isoformat() if topic.created_at else 'Unknown'}",
        ])

        content = "\n".join(parts)
        return Chunk(
            chunk_id=self._hash_content(content),
            document_id=document_id,
            content=content,
            token_count=self.count_tokens(content),
            chunk_index=chunk_index,
            section_path="Topic Header",
        )

    def _chunk_post(
        self,
        post: LoadedForumPost,
        topic: LoadedForumTopic | None,
        start_index: int,
    ) -> list[Chunk]:
        """Chunk a single post, keeping it atomic if possible."""
        if topic:
            document_id = self._make_document_id(topic.source, topic.topic_id)
            section_path = f"Post #{post.post_number}"
        else:
            document_id = self._make_document_id(post.source, post.topic_id)
            section_path = f"Post #{post.post_number}"

        # Build post header
        post_header = self._build_post_header(post)

        # Check if entire post fits in one chunk
        full_content = f"{post_header}\n\n{post.content}"
        full_tokens = self.count_tokens(full_content)

        if full_tokens <= self.max_tokens:
            return [Chunk(
                chunk_id=self._hash_content(full_content),
                document_id=document_id,
                content=full_content,
                token_count=full_tokens,
                chunk_index=start_index,
                section_path=section_path,
            )]

        # Need to split - extract code blocks first
        chunks = []
        code_blocks, prose_parts = self._extract_code_blocks(post.content)

        _ = self.count_tokens(post_header)  # reserved for future header budget tracking

        current_index = start_index
        first_chunk = True

        # Process prose parts
        for prose in prose_parts:
            if not prose.strip():
                continue

            prose_chunks = self._chunk_prose(
                prose=prose,
                document_id=document_id,
                section_path=section_path,
                start_index=current_index,
                prefix=post_header if first_chunk else None,
            )
            chunks.extend(prose_chunks)
            current_index += len(prose_chunks)
            first_chunk = False

        # Process code blocks as atomic units
        for i, code_block in enumerate(code_blocks):
            code_content = f"**{section_path} - Code Block {i + 1}**\n\n{code_block}"
            tokens = self.count_tokens(code_content)

            if tokens > self.max_tokens:
                logger.warning(
                    "oversized_code_block",
                    document_id=document_id,
                    post_number=post.post_number,
                    tokens=tokens,
                )

            chunks.append(Chunk(
                chunk_id=self._hash_content(code_content),
                document_id=document_id,
                content=code_content,
                token_count=tokens,
                chunk_index=current_index,
                section_path=f"{section_path} > Code Block {i + 1}",
            ))
            current_index += 1

        return chunks

    def _build_post_header(self, post: LoadedForumPost) -> str:
        """Build metadata header for a post."""
        parts = [f"## Post #{post.post_number} by @{post.username}"]

        if post.reply_to_post_number:
            parts.append(f"*Replying to Post #{post.reply_to_post_number}*")

        parts.append(f"*{post.created_at.strftime('%Y-%m-%d %H:%M UTC')}*")

        return "\n".join(parts)

    def _extract_code_blocks(self, content: str) -> tuple[list[str], list[str]]:
        """Extract code blocks and return (code_blocks, prose_parts)."""
        code_pattern = re.compile(r"```[\s\S]*?```", re.MULTILINE)
        code_blocks = code_pattern.findall(content)
        prose_parts = code_pattern.split(content)
        return code_blocks, prose_parts

    def _chunk_prose(
        self,
        prose: str,
        document_id: str,
        section_path: str,
        start_index: int,
        prefix: str | None = None,
    ) -> list[Chunk]:
        """Chunk prose content at paragraph boundaries."""
        tokens = self.tokenizer.encode(prose)
        prefix_tokens = self.count_tokens(prefix) if prefix else 0

        # Check if it fits with prefix
        if len(tokens) + prefix_tokens <= self.max_tokens:
            content = f"{prefix}\n\n{prose}" if prefix else prose
            return [Chunk(
                chunk_id=self._hash_content(content),
                document_id=document_id,
                content=content.strip(),
                token_count=len(tokens) + prefix_tokens,
                chunk_index=start_index,
                section_path=section_path,
            )]

        # Split at paragraph boundaries
        paragraphs = re.split(r"\n\n+", prose)
        chunks = []
        current_paragraphs = []
        current_tokens = prefix_tokens if prefix else 0
        include_prefix = prefix is not None

        for para in paragraphs:
            para_tokens = len(self.tokenizer.encode(para))

            if current_tokens + para_tokens > self.max_tokens and current_paragraphs:
                # Create chunk from accumulated paragraphs
                chunk_text = "\n\n".join(current_paragraphs)
                if include_prefix and prefix:
                    chunk_text = f"{prefix}\n\n{chunk_text}"
                    include_prefix = False

                chunks.append(Chunk(
                    chunk_id=self._hash_content(chunk_text),
                    document_id=document_id,
                    content=chunk_text,
                    token_count=current_tokens,
                    chunk_index=start_index + len(chunks),
                    section_path=section_path,
                ))
                current_paragraphs = []
                current_tokens = 0

            current_paragraphs.append(para)
            current_tokens += para_tokens

        # Don't forget the last chunk
        if current_paragraphs:
            chunk_text = "\n\n".join(current_paragraphs)
            if include_prefix and prefix:
                chunk_text = f"{prefix}\n\n{chunk_text}"

            chunks.append(Chunk(
                chunk_id=self._hash_content(chunk_text),
                document_id=document_id,
                content=chunk_text,
                token_count=current_tokens,
                chunk_index=start_index + len(chunks),
                section_path=section_path,
            ))

        return chunks

    def _make_document_id(self, source: str, topic_id: int) -> str:
        """Create a unique document ID for a forum topic."""
        return f"{source}-topic-{topic_id}"

    def _hash_content(self, content: str) -> str:
        """Generate a content-based hash for chunk ID."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def count_tokens(self, text: str | None) -> int:
        """Count tokens in text."""
        if not text:
            return 0
        return len(self.tokenizer.encode(text))
