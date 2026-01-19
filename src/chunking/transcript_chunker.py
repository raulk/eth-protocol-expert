"""Transcript Chunker - Speaker-aware chunking for ACD transcripts."""

import hashlib
from dataclasses import dataclass

import structlog
import tiktoken

from src.ingestion.acd_transcript_loader import ACDTranscript, SpeakerSegment

logger = structlog.get_logger()


@dataclass
class TranscriptChunk:
    """A chunk from a transcript with speaker attribution."""

    chunk_id: str
    content: str
    speaker: str | None
    call_number: int
    timestamp_start: str | None = None
    timestamp_end: str | None = None
    token_count: int = 0
    chunk_index: int = 0
    speakers_in_chunk: list[str] | None = None


class TranscriptChunker:
    """Speaker-aware chunking for AllCoreDevs transcripts.

    Preserves speaker boundaries when possible while respecting token limits.
    Includes speaker attribution in chunk metadata for proper sourcing.
    """

    def __init__(
        self,
        max_tokens: int = 512,
        overlap_tokens: int = 32,
        encoding_name: str = "cl100k_base",
    ):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.tokenizer = tiktoken.get_encoding(encoding_name)

    def chunk(
        self, transcript: ACDTranscript, max_tokens: int | None = None
    ) -> list[TranscriptChunk]:
        """Chunk a transcript while preserving speaker boundaries.

        Args:
            transcript: The ACD transcript to chunk
            max_tokens: Override default max tokens per chunk

        Returns:
            List of TranscriptChunk objects
        """
        max_tok = max_tokens or self.max_tokens
        chunks = []

        # If we have speaker segments, use speaker-aware chunking
        if transcript.segments:
            chunks = self._chunk_by_speakers(transcript, max_tok)
        else:
            # Fall back to simple content chunking
            chunks = self._chunk_plain_content(transcript, max_tok)

        logger.debug(
            "chunked_transcript",
            call_number=transcript.call_number,
            segments=len(transcript.segments),
            chunks=len(chunks),
        )
        return chunks

    def _chunk_by_speakers(
        self, transcript: ACDTranscript, max_tokens: int
    ) -> list[TranscriptChunk]:
        """Chunk transcript preserving speaker segment boundaries."""
        chunks = []
        chunk_index = 0

        # Group segments into chunks
        current_segments: list[SpeakerSegment] = []
        current_tokens = 0

        for segment in transcript.segments:
            segment_text = self._format_segment(segment)
            segment_tokens = self.count_tokens(segment_text)

            # If single segment exceeds max, split it
            if segment_tokens > max_tokens:
                # Flush current
                if current_segments:
                    chunk = self._create_chunk_from_segments(
                        current_segments, transcript, chunk_index
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    current_segments = []
                    current_tokens = 0

                # Split oversized segment
                split_chunks = self._split_large_segment(
                    segment, transcript, chunk_index, max_tokens
                )
                chunks.extend(split_chunks)
                chunk_index += len(split_chunks)
                continue

            # Check if adding this segment would exceed limit
            if current_tokens + segment_tokens > max_tokens and current_segments:
                # Create chunk from accumulated segments
                chunk = self._create_chunk_from_segments(current_segments, transcript, chunk_index)
                chunks.append(chunk)
                chunk_index += 1
                current_segments = []
                current_tokens = 0

            current_segments.append(segment)
            current_tokens += segment_tokens

        # Don't forget the last chunk
        if current_segments:
            chunk = self._create_chunk_from_segments(current_segments, transcript, chunk_index)
            chunks.append(chunk)

        return chunks

    def _chunk_plain_content(
        self, transcript: ACDTranscript, max_tokens: int
    ) -> list[TranscriptChunk]:
        """Chunk transcript content without speaker information."""
        chunks = []
        content = transcript.content

        # Add header
        header = f"# AllCoreDevs Call #{transcript.call_number}"
        if transcript.title:
            header += f": {transcript.title}"
        if transcript.date:
            header += f" ({transcript.date.strftime('%Y-%m-%d')})"
        header += "\n\n"

        full_content = header + content
        tokens = self.tokenizer.encode(full_content)

        if len(tokens) <= max_tokens:
            return [
                TranscriptChunk(
                    chunk_id=self._hash_content(full_content),
                    content=full_content,
                    speaker=None,
                    call_number=transcript.call_number,
                    token_count=len(tokens),
                    chunk_index=0,
                    speakers_in_chunk=transcript.speakers,
                )
            ]

        # Split into overlapping chunks
        chunk_index = 0
        start = 0

        while start < len(tokens):
            end = min(start + max_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)

            chunks.append(
                TranscriptChunk(
                    chunk_id=self._hash_content(chunk_text),
                    content=chunk_text,
                    speaker=None,
                    call_number=transcript.call_number,
                    token_count=len(chunk_tokens),
                    chunk_index=chunk_index,
                )
            )

            chunk_index += 1
            start = end - self.overlap_tokens

            if start >= len(tokens) - self.overlap_tokens:
                break

        return chunks

    def _create_chunk_from_segments(
        self,
        segments: list[SpeakerSegment],
        transcript: ACDTranscript,
        chunk_index: int,
    ) -> TranscriptChunk:
        """Create a chunk from a list of speaker segments."""
        # Format content with speaker attributions
        content_parts = []
        speakers_in_chunk = []
        timestamp_start = None
        timestamp_end = None

        for segment in segments:
            content_parts.append(self._format_segment(segment))
            if segment.speaker not in speakers_in_chunk:
                speakers_in_chunk.append(segment.speaker)
            if segment.timestamp:
                if timestamp_start is None:
                    timestamp_start = segment.timestamp
                timestamp_end = segment.timestamp

        content = "\n\n".join(content_parts)

        # Add chunk header
        header = f"[ACD Call #{transcript.call_number}]"
        if timestamp_start:
            header += f" [{timestamp_start}"
            if timestamp_end and timestamp_end != timestamp_start:
                header += f" - {timestamp_end}"
            header += "]"
        header += "\n\n"

        full_content = header + content

        return TranscriptChunk(
            chunk_id=self._hash_content(full_content),
            content=full_content,
            speaker=segments[0].speaker if len(segments) == 1 else None,
            call_number=transcript.call_number,
            timestamp_start=timestamp_start,
            timestamp_end=timestamp_end,
            token_count=self.count_tokens(full_content),
            chunk_index=chunk_index,
            speakers_in_chunk=speakers_in_chunk,
        )

    def _split_large_segment(
        self,
        segment: SpeakerSegment,
        transcript: ACDTranscript,
        start_index: int,
        max_tokens: int,
    ) -> list[TranscriptChunk]:
        """Split a single large segment into multiple chunks."""
        chunks = []
        text = segment.text
        tokens = self.tokenizer.encode(text)

        chunk_index = start_index
        start = 0

        while start < len(tokens):
            # Reserve tokens for header
            header_reserve = 50
            end = min(start + max_tokens - header_reserve, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)

            # Format with speaker and context
            formatted = f"[ACD Call #{transcript.call_number}]"
            if segment.timestamp:
                formatted += f" [{segment.timestamp}]"
            formatted += f"\n\n**{segment.speaker}:** {chunk_text}"

            if start > 0:
                formatted = f"[...continued]\n{formatted}"
            if end < len(tokens):
                formatted += "\n[continued...]"

            chunks.append(
                TranscriptChunk(
                    chunk_id=self._hash_content(formatted),
                    content=formatted,
                    speaker=segment.speaker,
                    call_number=transcript.call_number,
                    timestamp_start=segment.timestamp,
                    timestamp_end=segment.timestamp,
                    token_count=self.count_tokens(formatted),
                    chunk_index=chunk_index,
                    speakers_in_chunk=[segment.speaker],
                )
            )

            chunk_index += 1
            start = end - self.overlap_tokens

            if start >= len(tokens) - self.overlap_tokens:
                break

        return chunks

    def _format_segment(self, segment: SpeakerSegment) -> str:
        """Format a speaker segment for inclusion in a chunk."""
        result = f"**{segment.speaker}:**"
        if segment.timestamp:
            result += f" [{segment.timestamp}]"
        result += f" {segment.text}"
        return result

    def _hash_content(self, content: str) -> str:
        """Generate a content-based hash for chunk ID."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text))
