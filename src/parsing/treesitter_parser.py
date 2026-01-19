"""Tree-sitter Parser - Parse Go and Rust source files."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import structlog
import tree_sitter_languages

logger = structlog.get_logger()


@dataclass
class ParsedFunction:
    """A parsed function or method from source code."""

    name: str
    signature: str
    body: str
    start_line: int
    end_line: int
    docstring: str | None = None


@dataclass
class ParsedStruct:
    """A parsed struct/type definition from source code."""

    name: str
    fields: list[tuple[str, str]]
    methods: list[str]
    start_line: int
    end_line: int


@dataclass
class ParsedFile:
    """Result of parsing a source file."""

    path: str
    language: str
    tree: object
    functions: list[ParsedFunction] = field(default_factory=list)
    structs: list[ParsedStruct] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)


class TreeSitterParser:
    """Parse Go and Rust source files using tree-sitter."""

    LANGUAGE_EXTENSIONS: ClassVar[dict[str, str]] = {
        ".go": "go",
        ".rs": "rust",
    }

    GO_FUNCTION_QUERY: ClassVar[str] = """
        (function_declaration
            name: (identifier) @func_name
            parameters: (parameter_list) @params
            result: (_)? @result
            body: (block) @body
        ) @function

        (method_declaration
            receiver: (parameter_list) @receiver
            name: (field_identifier) @method_name
            parameters: (parameter_list) @params
            result: (_)? @result
            body: (block) @body
        ) @method
    """

    GO_STRUCT_QUERY: ClassVar[str] = """
        (type_declaration
            (type_spec
                name: (type_identifier) @struct_name
                type: (struct_type
                    (field_declaration_list) @fields
                )
            )
        ) @struct
    """

    GO_IMPORT_QUERY: ClassVar[str] = """
        (import_declaration
            (import_spec_list
                (import_spec
                    path: (interpreted_string_literal) @import_path
                )
            )
        )

        (import_declaration
            (import_spec
                path: (interpreted_string_literal) @import_path
            )
        )
    """

    RUST_FUNCTION_QUERY: ClassVar[str] = """
        (function_item
            name: (identifier) @func_name
            parameters: (parameters) @params
            return_type: (_)? @return_type
            body: (block) @body
        ) @function

        (impl_item
            type: (_) @impl_type
            body: (declaration_list
                (function_item
                    name: (identifier) @method_name
                    parameters: (parameters) @params
                    return_type: (_)? @return_type
                    body: (block) @body
                ) @method
            )
        )
    """

    RUST_STRUCT_QUERY: ClassVar[str] = """
        (struct_item
            name: (type_identifier) @struct_name
            body: (field_declaration_list)? @fields
        ) @struct
    """

    RUST_IMPORT_QUERY: ClassVar[str] = """
        (use_declaration
            argument: (_) @use_path
        )
    """

    def __init__(self, languages: list[str] | None = None):
        self.languages = languages or ["go", "rust"]
        self._parsers: dict[str, object] = {}
        self._queries: dict[str, dict[str, object]] = {}

        for lang in self.languages:
            self._init_language(lang)

    def _init_language(self, language: str) -> None:
        parser = tree_sitter_languages.get_parser(language)
        self._parsers[language] = parser

        ts_language = tree_sitter_languages.get_language(language)
        self._queries[language] = {}

        if language == "go":
            self._queries[language]["function"] = ts_language.query(self.GO_FUNCTION_QUERY)
            self._queries[language]["struct"] = ts_language.query(self.GO_STRUCT_QUERY)
            self._queries[language]["import"] = ts_language.query(self.GO_IMPORT_QUERY)
        elif language == "rust":
            self._queries[language]["function"] = ts_language.query(self.RUST_FUNCTION_QUERY)
            self._queries[language]["struct"] = ts_language.query(self.RUST_STRUCT_QUERY)
            self._queries[language]["import"] = ts_language.query(self.RUST_IMPORT_QUERY)

        logger.debug("initialized_language_parser", language=language)

    def parse_file(self, path: str) -> ParsedFile:
        """Parse a source file and extract functions, structs, and imports."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        extension = file_path.suffix.lower()
        language = self.LANGUAGE_EXTENSIONS.get(extension)
        if not language:
            raise ValueError(f"Unsupported file extension: {extension}")

        if language not in self._parsers:
            raise ValueError(f"Language not initialized: {language}")

        content = file_path.read_text(encoding="utf-8")
        return self.parse_content(content, language, path)

    def parse_content(
        self,
        content: str,
        language: str,
        path: str = "<string>",
    ) -> ParsedFile:
        """Parse source content and extract code elements."""
        if language not in self._parsers:
            raise ValueError(f"Language not initialized: {language}")

        parser = self._parsers[language]
        tree = parser.parse(content.encode("utf-8"))

        parsed = ParsedFile(
            path=path,
            language=language,
            tree=tree,
        )

        content_bytes = content.encode("utf-8")
        lines = content.split("\n")

        self._extract_functions(parsed, tree, content_bytes, lines, language)
        self._extract_structs(parsed, tree, content_bytes, lines, language)
        self._extract_imports(parsed, tree, content_bytes, language)

        logger.debug(
            "parsed_file",
            path=path,
            language=language,
            functions=len(parsed.functions),
            structs=len(parsed.structs),
            imports=len(parsed.imports),
        )

        return parsed

    def _extract_functions(
        self,
        parsed: ParsedFile,
        tree: object,
        content_bytes: bytes,
        lines: list[str],
        language: str,
    ) -> None:
        query = self._queries[language]["function"]
        captures = query.captures(tree.root_node)

        capture_dict: dict[int, dict[str, object]] = {}

        for node, name in captures:
            node_id = id(node.parent) if node.parent else id(node)

            if name in ("function", "method"):
                capture_dict[id(node)] = {"node": node, "captures": {}}
                node_id = id(node)

            if node_id not in capture_dict:
                for parent_id, data in capture_dict.items():
                    if data["node"].start_byte <= node.start_byte <= data["node"].end_byte:
                        node_id = parent_id
                        break

            if node_id in capture_dict:
                capture_dict[node_id]["captures"][name] = node

        for data in capture_dict.values():
            caps = data["captures"]
            func_node = data["node"]

            name_node = caps.get("func_name") or caps.get("method_name")
            if not name_node:
                continue

            func_name = content_bytes[name_node.start_byte : name_node.end_byte].decode("utf-8")

            signature_parts = [func_name]
            if "receiver" in caps:
                receiver = content_bytes[
                    caps["receiver"].start_byte : caps["receiver"].end_byte
                ].decode("utf-8")
                signature_parts.insert(0, receiver)

            if "params" in caps:
                params = content_bytes[caps["params"].start_byte : caps["params"].end_byte].decode(
                    "utf-8"
                )
                signature_parts.append(params)

            if "result" in caps:
                result = content_bytes[caps["result"].start_byte : caps["result"].end_byte].decode(
                    "utf-8"
                )
                signature_parts.append(f" {result}")
            elif "return_type" in caps:
                ret = content_bytes[
                    caps["return_type"].start_byte : caps["return_type"].end_byte
                ].decode("utf-8")
                signature_parts.append(f" -> {ret}")

            signature = " ".join(signature_parts[:2]) + "".join(signature_parts[2:])

            body_node = caps.get("body")
            body = ""
            if body_node:
                body = content_bytes[body_node.start_byte : body_node.end_byte].decode("utf-8")

            start_line = func_node.start_point[0] + 1
            end_line = func_node.end_point[0] + 1

            docstring = self._extract_docstring(lines, start_line - 1, language)

            parsed.functions.append(
                ParsedFunction(
                    name=func_name,
                    signature=signature,
                    body=body,
                    start_line=start_line,
                    end_line=end_line,
                    docstring=docstring,
                )
            )

    def _extract_structs(
        self,
        parsed: ParsedFile,
        tree: object,
        content_bytes: bytes,
        lines: list[str],
        language: str,
    ) -> None:
        query = self._queries[language]["struct"]
        captures = query.captures(tree.root_node)

        struct_data: dict[int, dict[str, object]] = {}

        for node, name in captures:
            if name == "struct":
                struct_data[id(node)] = {"node": node, "name": None, "fields_node": None}
            elif name == "struct_name":
                for data in struct_data.values():
                    if data["node"].start_byte <= node.start_byte <= data["node"].end_byte:
                        data["name"] = content_bytes[node.start_byte : node.end_byte].decode(
                            "utf-8"
                        )
                        break
            elif name == "fields":
                for data in struct_data.values():
                    if data["node"].start_byte <= node.start_byte <= data["node"].end_byte:
                        data["fields_node"] = node
                        break

        for data in struct_data.values():
            if not data["name"]:
                continue

            struct_node = data["node"]
            struct_name = data["name"]

            fields: list[tuple[str, str]] = []
            if data["fields_node"]:
                fields = self._extract_fields(data["fields_node"], content_bytes, language)

            start_line = struct_node.start_point[0] + 1
            end_line = struct_node.end_point[0] + 1

            parsed.structs.append(
                ParsedStruct(
                    name=struct_name,
                    fields=fields,
                    methods=[],
                    start_line=start_line,
                    end_line=end_line,
                )
            )

    def _extract_fields(
        self,
        fields_node: object,
        content_bytes: bytes,
        language: str,
    ) -> list[tuple[str, str]]:
        fields: list[tuple[str, str]] = []

        for child in fields_node.children:
            if language == "go" and child.type == "field_declaration":
                field_name = ""
                field_type = ""
                for fc in child.children:
                    if fc.type == "field_identifier":
                        field_name = content_bytes[fc.start_byte : fc.end_byte].decode("utf-8")
                    elif fc.type not in (",", "comment"):
                        field_type = content_bytes[fc.start_byte : fc.end_byte].decode("utf-8")
                if field_name:
                    fields.append((field_name, field_type))

            elif language == "rust" and child.type == "field_declaration":
                field_name = ""
                field_type = ""
                for fc in child.children:
                    if fc.type == "field_identifier":
                        field_name = content_bytes[fc.start_byte : fc.end_byte].decode("utf-8")
                    elif fc.type not in (":", ",", "visibility_modifier"):
                        field_type = content_bytes[fc.start_byte : fc.end_byte].decode("utf-8")
                if field_name:
                    fields.append((field_name, field_type))

        return fields

    def _extract_imports(
        self,
        parsed: ParsedFile,
        tree: object,
        content_bytes: bytes,
        language: str,
    ) -> None:
        query = self._queries[language]["import"]
        captures = query.captures(tree.root_node)

        for node, name in captures:
            if name in ("import_path", "use_path"):
                import_str = content_bytes[node.start_byte : node.end_byte].decode("utf-8")
                import_str = import_str.strip('"').strip("'")
                if import_str not in parsed.imports:
                    parsed.imports.append(import_str)

    def _extract_docstring(
        self,
        lines: list[str],
        start_line_idx: int,
        language: str,
    ) -> str | None:
        doc_lines: list[str] = []

        idx = start_line_idx - 1
        while idx >= 0:
            line = lines[idx].strip()

            if language == "go":
                if line.startswith("//"):
                    doc_lines.insert(0, line[2:].strip())
                    idx -= 1
                else:
                    break
            elif language == "rust":
                if line.startswith("///"):
                    doc_lines.insert(0, line[3:].strip())
                    idx -= 1
                elif line.startswith("//!"):
                    doc_lines.insert(0, line[3:].strip())
                    idx -= 1
                else:
                    break
            else:
                break

        return "\n".join(doc_lines) if doc_lines else None
