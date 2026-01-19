"""Code Chunker - Function-level chunking for code units."""

import hashlib
from dataclasses import dataclass

import structlog
import tiktoken

from src.parsing.code_unit_extractor import CodeUnit

logger = structlog.get_logger()


@dataclass
class CodeChunk:
    """A chunk of code suitable for embedding."""

    chunk_id: str
    content: str
    function_name: str
    file_path: str
    language: str
    dependencies: list[str]
    token_count: int


class CodeChunker:
    """Create embedable chunks from code units."""

    def __init__(
        self,
        max_tokens: int = 512,
        encoding_name: str = "cl100k_base",
    ):
        self.max_tokens = max_tokens
        self.tokenizer = tiktoken.get_encoding(encoding_name)

    def chunk(
        self,
        code_units: list[CodeUnit],
        language: str = "go",
    ) -> list[CodeChunk]:
        """Convert code units into embeddable chunks."""
        chunks: list[CodeChunk] = []

        for unit in code_units:
            token_count = self._count_tokens(unit.content)

            if token_count <= self.max_tokens:
                chunks.append(self._create_chunk(unit, unit.content, language, token_count))
            else:
                split_chunks = self._split_large_unit(unit, language)
                chunks.extend(split_chunks)

        logger.debug(
            "chunked_code_units",
            input_units=len(code_units),
            output_chunks=len(chunks),
        )

        return chunks

    def chunk_with_context(
        self,
        unit: CodeUnit,
        source_lines: list[str],
        context_lines: int = 10,
        language: str = "go",
    ) -> CodeChunk:
        """Create a chunk with surrounding context from the source file."""
        start_line, end_line = unit.line_range

        context_start = max(0, start_line - 1 - context_lines)
        context_end = min(len(source_lines), end_line + context_lines)

        before_context = source_lines[context_start : start_line - 1]
        after_context = source_lines[end_line:context_end]

        content_parts: list[str] = []

        if before_context:
            content_parts.append(f"// Context before ({len(before_context)} lines):")
            content_parts.extend(before_context)
            content_parts.append("")

        content_parts.append(unit.content)

        if after_context:
            content_parts.append("")
            content_parts.append(f"// Context after ({len(after_context)} lines):")
            content_parts.extend(after_context)

        full_content = "\n".join(content_parts)
        token_count = self._count_tokens(full_content)

        if token_count > self.max_tokens:
            return self._create_chunk(
                unit, unit.content, language, self._count_tokens(unit.content)
            )

        return self._create_chunk(unit, full_content, language, token_count)

    def _create_chunk(
        self,
        unit: CodeUnit,
        content: str,
        language: str,
        token_count: int,
    ) -> CodeChunk:
        chunk_id = self._hash_content(content)

        return CodeChunk(
            chunk_id=chunk_id,
            content=content,
            function_name=unit.name,
            file_path=unit.file_path,
            language=language,
            dependencies=unit.dependencies.copy(),
            token_count=token_count,
        )

    def _split_large_unit(
        self,
        unit: CodeUnit,
        language: str,
    ) -> list[CodeChunk]:
        """Split a large code unit into smaller chunks."""
        chunks: list[CodeChunk] = []
        lines = unit.content.split("\n")

        header_lines: list[str] = []
        body_lines: list[str] = []
        in_body = False
        brace_count = 0

        for line in lines:
            if not in_body:
                header_lines.append(line)
                if "{" in line:
                    brace_count += line.count("{") - line.count("}")
                    if brace_count > 0:
                        in_body = True
            else:
                body_lines.append(line)
                brace_count += line.count("{") - line.count("}")

        header = "\n".join(header_lines)
        header_tokens = self._count_tokens(header)

        available_tokens = self.max_tokens - header_tokens - 10

        if available_tokens <= 50:
            chunk_content = unit.content[: self._tokens_to_chars(self.max_tokens)]
            return [
                self._create_chunk(
                    unit,
                    chunk_content + "\n// ... truncated",
                    language,
                    self._count_tokens(chunk_content),
                )
            ]

        current_lines: list[str] = []
        current_tokens = 0
        chunk_index = 0

        for line in body_lines:
            line_tokens = self._count_tokens(line)

            if current_tokens + line_tokens > available_tokens and current_lines:
                chunk_content = header + "\n" + "\n".join(current_lines) + "\n    // ... continued"
                chunks.append(
                    CodeChunk(
                        chunk_id=self._hash_content(chunk_content + str(chunk_index)),
                        content=chunk_content,
                        function_name=f"{unit.name}_part{chunk_index + 1}",
                        file_path=unit.file_path,
                        language=language,
                        dependencies=unit.dependencies.copy(),
                        token_count=self._count_tokens(chunk_content),
                    )
                )
                chunk_index += 1
                current_lines = []
                current_tokens = 0

            current_lines.append(line)
            current_tokens += line_tokens

        if current_lines:
            chunk_content = header + "\n" + "\n".join(current_lines)
            if chunk_index > 0:
                chunk_content = "// ... continued from previous chunk\n" + chunk_content
            chunks.append(
                CodeChunk(
                    chunk_id=self._hash_content(chunk_content + str(chunk_index)),
                    content=chunk_content,
                    function_name=f"{unit.name}_part{chunk_index + 1}"
                    if chunk_index > 0
                    else unit.name,
                    file_path=unit.file_path,
                    language=language,
                    dependencies=unit.dependencies.copy(),
                    token_count=self._count_tokens(chunk_content),
                )
            )

        logger.debug(
            "split_large_unit",
            name=unit.name,
            original_tokens=self._count_tokens(unit.content),
            chunks=len(chunks),
        )

        return chunks

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def _tokens_to_chars(self, tokens: int) -> int:
        return tokens * 4

    def _hash_content(self, content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()[:16]
