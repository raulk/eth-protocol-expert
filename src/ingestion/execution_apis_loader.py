"""Loader for ethereum/execution-apis specifications (OpenRPC/JSON-RPC format).

The execution-apis repository defines the JSON-RPC interface for Ethereum
execution clients. It uses OpenRPC format with:

- Method definitions: JSON-RPC methods with params, result, errors, examples
- Schema definitions: Type definitions for parameters and results
- Markdown specs: Detailed human-readable documentation per fork (Paris, Shanghai, etc.)
"""

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import structlog
import yaml

logger = structlog.get_logger()


@dataclass
class JSONRPCMethod:
    """A parsed JSON-RPC method definition from OpenRPC spec."""

    name: str
    summary: str
    description: str
    params: list[dict]
    result: dict
    errors: list[dict]
    examples: list[dict]
    external_docs: dict | None
    source_file: Path
    namespace: str


@dataclass
class JSONRPCSchema:
    """A parsed schema definition from OpenRPC spec."""

    name: str
    title: str
    description: str
    schema_type: str
    properties: dict
    required: list[str]
    enum: list[str] | None
    pattern: str | None
    source_file: Path
    category: str


@dataclass
class MarkdownDoc:
    """A markdown specification document."""

    title: str
    content: str
    file_path: Path
    fork_name: str | None


@dataclass
class ParsedExecutionAPIs:
    """Complete parsed execution APIs specification."""

    methods: list[JSONRPCMethod] = field(default_factory=list)
    schemas: list[JSONRPCSchema] = field(default_factory=list)
    markdown_docs: list[MarkdownDoc] = field(default_factory=list)


class ExecutionAPIsLoader:
    """Load execution APIs from ethereum/execution-apis repository.

    The execution-apis repository defines the JSON-RPC interface for Ethereum
    execution clients. Structure:

    - src/engine/openrpc/methods/ - Engine API methods (consensus-execution interface)
    - src/engine/openrpc/schemas/ - Engine-specific schemas
    - src/eth/ - Standard eth_ namespace methods
    - src/debug/ - Debug namespace methods
    - src/schemas/ - Common schema definitions
    - src/engine/*.md - Detailed specs per fork (Paris, Shanghai, Cancun, etc.)
    """

    REPO_URL = "https://github.com/ethereum/execution-apis.git"

    METHOD_DIRS: ClassVar[list[tuple[str, str]]] = [
        ("src/engine/openrpc/methods", "engine"),
        ("src/eth", "eth"),
        ("src/debug", "debug"),
    ]

    SCHEMA_DIRS: ClassVar[list[tuple[str, str]]] = [
        ("src/schemas", "common"),
        ("src/engine/openrpc/schemas", "engine"),
    ]

    MARKDOWN_DIRS: ClassVar[list[str]] = [
        "src/engine",
    ]

    def __init__(self, repo_path: str | Path = "data/execution-apis") -> None:
        self.repo_url = self.REPO_URL
        self.repo_path = Path(repo_path)
        self.source_name = "execution-apis"

    def clone_or_update(self) -> str:
        """Clone or update the repository.

        Returns:
            Current git commit hash.
        """
        if not self.repo_path.exists():
            logger.info("cloning_repo", source=self.source_name, url=self.repo_url)
            self.repo_path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["git", "clone", "--depth", "1", self.repo_url, str(self.repo_path)],
                check=True,
                capture_output=True,
            )
        else:
            logger.info("updating_repo", source=self.source_name)
            subprocess.run(
                ["git", "-C", str(self.repo_path), "pull", "--ff-only"],
                check=True,
                capture_output=True,
            )

        result = subprocess.run(
            ["git", "-C", str(self.repo_path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    def get_current_commit(self) -> str:
        """Get current git commit SHA."""
        if not self.repo_path.exists():
            raise ValueError(f"Repo not found at {self.repo_path}")

        result = subprocess.run(
            ["git", "-C", str(self.repo_path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    def _load_yaml(self, file_path: Path) -> dict | list | None:
        """Load a YAML file."""
        try:
            with open(file_path, encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(
                "failed_to_load_yaml",
                source=self.source_name,
                path=str(file_path),
                error=str(e),
            )
            return None

    def _parse_method(
        self, method_def: dict, source_file: Path, namespace: str
    ) -> JSONRPCMethod | None:
        """Parse a single JSON-RPC method definition."""
        name = method_def.get("name")
        if not name:
            return None

        summary = method_def.get("summary", "")
        description = method_def.get("description", "")
        params = method_def.get("params", [])
        result = method_def.get("result", {})
        errors = method_def.get("errors", [])
        examples = method_def.get("examples", [])
        external_docs = method_def.get("externalDocs")

        return JSONRPCMethod(
            name=name,
            summary=summary,
            description=description,
            params=params,
            result=result,
            errors=errors,
            examples=examples,
            external_docs=external_docs,
            source_file=source_file,
            namespace=namespace,
        )

    def _parse_schema(
        self, name: str, schema_def: dict, source_file: Path, category: str
    ) -> JSONRPCSchema:
        """Parse a single schema definition."""
        title = schema_def.get("title", name)
        description = schema_def.get("description", "")
        schema_type = schema_def.get("type", "object")
        properties = schema_def.get("properties", {})
        required = schema_def.get("required", [])
        enum = schema_def.get("enum")
        pattern = schema_def.get("pattern")

        return JSONRPCSchema(
            name=name,
            title=title,
            description=description,
            schema_type=schema_type,
            properties=properties,
            required=required,
            enum=enum,
            pattern=pattern,
            source_file=source_file,
            category=category,
        )

    def _load_methods_from_file(self, file_path: Path, namespace: str) -> list[JSONRPCMethod]:
        """Load all methods from a single YAML file."""
        methods: list[JSONRPCMethod] = []
        data = self._load_yaml(file_path)

        if data is None:
            return methods

        if isinstance(data, list):
            for method_def in data:
                if isinstance(method_def, dict):
                    method = self._parse_method(method_def, file_path, namespace)
                    if method:
                        methods.append(method)
        elif isinstance(data, dict):
            for key, method_def in data.items():
                if isinstance(method_def, dict) and "name" not in method_def:
                    method_def["name"] = key
                if isinstance(method_def, dict):
                    method = self._parse_method(method_def, file_path, namespace)
                    if method:
                        methods.append(method)

        return methods

    def _load_schemas_from_file(self, file_path: Path, category: str) -> list[JSONRPCSchema]:
        """Load all schemas from a single YAML file."""
        schemas: list[JSONRPCSchema] = []
        data = self._load_yaml(file_path)

        if data is None:
            return schemas

        if isinstance(data, dict):
            for name, schema_def in data.items():
                if isinstance(schema_def, dict):
                    schema = self._parse_schema(name, schema_def, file_path, category)
                    schemas.append(schema)

        return schemas

    def _load_markdown_docs(self, md_dir: str) -> list[MarkdownDoc]:
        """Load markdown documentation files."""
        docs: list[MarkdownDoc] = []
        dir_path = self.repo_path / md_dir

        if not dir_path.exists():
            return docs

        for md_file in dir_path.glob("*.md"):
            if md_file.name in ["README.md"]:
                continue

            try:
                content = md_file.read_text(encoding="utf-8")
                if len(content.strip()) < 100:
                    continue

                title = md_file.stem.replace("-", " ").replace("_", " ").title()
                for line in content.split("\n"):
                    if line.startswith("# "):
                        title = line[2:].strip()
                        break

                fork_name = md_file.stem.lower()

                docs.append(
                    MarkdownDoc(
                        title=title,
                        content=content,
                        file_path=md_file,
                        fork_name=fork_name,
                    )
                )

            except Exception as e:
                logger.warning(
                    "failed_to_read_markdown",
                    source=self.source_name,
                    path=str(md_file),
                    error=str(e),
                )

        return docs

    def load_all(self) -> ParsedExecutionAPIs:
        """Load all execution APIs specifications."""
        result = ParsedExecutionAPIs()

        if not self.repo_path.exists():
            logger.warning("repo_not_found", path=str(self.repo_path))
            return result

        for method_dir, namespace in self.METHOD_DIRS:
            dir_path = self.repo_path / method_dir
            if not dir_path.exists():
                continue

            for yaml_file in dir_path.glob("*.yaml"):
                methods = self._load_methods_from_file(yaml_file, namespace)
                result.methods.extend(methods)
                logger.debug(
                    "loaded_methods",
                    file=str(yaml_file),
                    count=len(methods),
                    namespace=namespace,
                )

        for schema_dir, category in self.SCHEMA_DIRS:
            dir_path = self.repo_path / schema_dir
            if not dir_path.exists():
                continue

            for yaml_file in dir_path.glob("*.yaml"):
                schemas = self._load_schemas_from_file(yaml_file, category)
                result.schemas.extend(schemas)
                logger.debug(
                    "loaded_schemas",
                    file=str(yaml_file),
                    count=len(schemas),
                    category=category,
                )

        for md_dir in self.MARKDOWN_DIRS:
            docs = self._load_markdown_docs(md_dir)
            result.markdown_docs.extend(docs)

        logger.info(
            f"loaded_{self.source_name}",
            methods=len(result.methods),
            schemas=len(result.schemas),
            markdown_docs=len(result.markdown_docs),
        )

        return result

    def method_to_markdown(self, method: JSONRPCMethod) -> str:
        """Convert a JSON-RPC method to markdown documentation."""
        lines = [
            f"## {method.name}",
            "",
        ]

        if method.summary:
            lines.append(f"**Summary:** {method.summary}")
            lines.append("")

        lines.append(f"**Namespace:** {method.namespace}")
        lines.append("")

        if method.description:
            lines.append(method.description)
            lines.append("")

        if method.external_docs:
            url = method.external_docs.get("url", "")
            desc = method.external_docs.get("description", "Specification")
            if url:
                lines.append(f"**{desc}:** {url}")
                lines.append("")

        if method.params:
            lines.append("### Parameters")
            lines.append("")
            for i, param in enumerate(method.params):
                if isinstance(param, dict):
                    name = param.get("name", f"param{i}")
                    required = param.get("required", False)
                    schema = param.get("schema", {})
                    param_desc = self._describe_schema(schema)

                    req_marker = " (required)" if required else " (optional)"
                    lines.append(f"- **{name}**{req_marker}: {param_desc}")
            lines.append("")

        if method.result:
            lines.append("### Result")
            lines.append("")
            if isinstance(method.result, dict):
                result_name = method.result.get("name", "Result")
                result_schema = method.result.get("schema", {})
                result_desc = self._describe_schema(result_schema)
                lines.append(f"**{result_name}:** {result_desc}")
            lines.append("")

        if method.errors:
            lines.append("### Errors")
            lines.append("")
            for error in method.errors:
                if isinstance(error, dict):
                    code = error.get("code", "")
                    message = error.get("message", "")
                    lines.append(f"- **{code}**: {message}")
            lines.append("")

        if method.examples:
            lines.append("### Examples")
            lines.append("")
            for example in method.examples:
                if isinstance(example, dict):
                    example_name = example.get("name", "Example")
                    lines.append(f"**{example_name}**")
                    lines.append("")

                    example_params = example.get("params", [])
                    if example_params:
                        lines.append("Request params:")
                        lines.append("```json")
                        for p in example_params:
                            if isinstance(p, dict):
                                param_name = p.get("name", "")
                                param_value = p.get("value")
                                lines.append(f"// {param_name}")
                                if param_value is not None:
                                    lines.append(
                                        yaml.dump(param_value, default_flow_style=True).strip()
                                    )
                        lines.append("```")
                        lines.append("")

                    example_result = example.get("result", {})
                    if example_result:
                        result_value = example_result.get("value")
                        if result_value is not None:
                            lines.append("Response:")
                            lines.append("```json")
                            lines.append(yaml.dump(result_value, default_flow_style=False).strip())
                            lines.append("```")
                            lines.append("")

        return "\n".join(lines)

    def _describe_schema(self, schema: dict) -> str:
        """Create a brief description of a schema."""
        if not isinstance(schema, dict):
            return "any"

        if "$ref" in schema:
            ref = schema["$ref"]
            return f"-> {ref.split('/')[-1]}" if "/" in ref else f"-> {ref}"

        if "oneOf" in schema or "anyOf" in schema:
            union_key = "oneOf" if "oneOf" in schema else "anyOf"
            types = []
            for opt in schema[union_key]:
                if isinstance(opt, dict):
                    if "$ref" in opt:
                        types.append(opt["$ref"].split("/")[-1])
                    elif "type" in opt:
                        types.append(opt["type"])
                    elif "title" in opt:
                        types.append(opt["title"])
            return " | ".join(types) if types else "union"

        if "type" in schema:
            base_type = schema["type"]
            if base_type == "array" and "items" in schema:
                items = schema["items"]
                if isinstance(items, dict):
                    if "$ref" in items:
                        item_type = items["$ref"].split("/")[-1]
                        return f"array of {item_type}"
                    elif "type" in items:
                        return f"array of {items['type']}"
            return base_type

        return "object"

    def schema_to_markdown(self, schema: JSONRPCSchema) -> str:
        """Convert a schema to markdown documentation."""
        lines = [
            f"## {schema.name}",
            "",
        ]

        if schema.title and schema.title != schema.name:
            lines.append(f"**Title:** {schema.title}")
            lines.append("")

        if schema.description:
            lines.append(schema.description)
            lines.append("")

        lines.append(f"**Type:** {schema.schema_type}")
        lines.append(f"**Category:** {schema.category}")
        lines.append("")

        if schema.enum:
            lines.append("### Enum values")
            lines.append("")
            for value in schema.enum:
                lines.append(f"- `{value}`")
            lines.append("")

        if schema.pattern:
            lines.append(f"**Pattern:** `{schema.pattern}`")
            lines.append("")

        if schema.properties:
            lines.append("### Properties")
            lines.append("")
            for prop_name, prop_def in schema.properties.items():
                if isinstance(prop_def, dict):
                    prop_type = prop_def.get("type", "any")
                    prop_desc = prop_def.get("description", "")
                    prop_title = prop_def.get("title", "")

                    required_marker = " (required)" if prop_name in schema.required else ""

                    if "$ref" in prop_def:
                        ref = prop_def["$ref"]
                        ref_name = ref.split("/")[-1] if "/" in ref else ref
                        prop_type = f"-> {ref_name}"

                    display_desc = prop_desc if prop_desc else prop_title
                    lines.append(
                        f"- **{prop_name}** ({prop_type}){required_marker}: {display_desc}"
                    )
            lines.append("")

        return "\n".join(lines)

    def get_all_markdown(self) -> list[tuple[str, str, str, Path]]:
        """Get all content as markdown documents for chunking.

        Returns:
            List of (doc_type, title, content, source_path) tuples.
        """
        result: list[tuple[str, str, str, Path]] = []
        parsed = self.load_all()

        for method in parsed.methods:
            title = method.name
            content = self.method_to_markdown(method)
            result.append(("method", title, content, method.source_file))

        for schema in parsed.schemas:
            title = schema.name
            content = self.schema_to_markdown(schema)
            result.append(("schema", title, content, schema.source_file))

        for doc in parsed.markdown_docs:
            result.append(("spec", doc.title, doc.content, doc.file_path))

        return result
