"""Generic loader for OpenAPI specification repositories."""

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import structlog
import yaml

logger = structlog.get_logger()


@dataclass
class APIEndpoint:
    """A parsed API endpoint from OpenAPI spec."""

    path: str
    method: str
    summary: str
    description: str
    tags: list[str]
    parameters: list[dict]
    request_body: dict | None
    responses: dict
    operation_id: str | None
    source_file: Path


@dataclass
class APISchema:
    """A parsed schema/type definition from OpenAPI spec."""

    name: str
    title: str
    description: str
    schema_type: str
    properties: dict
    required: list[str]
    source_file: Path


@dataclass
class ParsedOpenAPI:
    """Complete parsed OpenAPI specification."""

    title: str
    version: str
    description: str
    endpoints: list[APIEndpoint]
    schemas: list[APISchema]
    source_file: Path
    tags: list[dict]


class OpenAPILoader:
    """Generic loader for OpenAPI specification repositories.

    Loads OpenAPI YAML files and extracts endpoints and schemas
    in a format suitable for chunking and embedding.
    """

    def __init__(
        self,
        repo_url: str,
        repo_path: str | Path,
        source_name: str,
        spec_file: str | None = None,
        spec_patterns: list[str] | None = None,
    ) -> None:
        """Initialize the OpenAPI loader.

        Args:
            repo_url: Git URL to clone from
            repo_path: Local path to clone/store the repo
            source_name: Name for logging and document source field
            spec_file: Single OpenAPI spec file (e.g., "openapi.yaml")
            spec_patterns: Glob patterns to find spec files
        """
        self.repo_url = repo_url
        self.repo_path = Path(repo_path)
        self.source_name = source_name
        self.spec_file = spec_file
        self.spec_patterns = spec_patterns or ["*.yaml", "*.yml"]

    def clone_or_update(self) -> str:
        """Clone or update the repository.

        Returns:
            Current git commit hash.
        """
        if not self.repo_path.exists():
            logger.info(
                "cloning_repo", source=self.source_name, url=self.repo_url
            )
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

    def _load_yaml(self, file_path: Path) -> dict | None:
        """Load a YAML file, handling $ref resolution within the same file."""
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

    def _resolve_refs(
        self, obj: dict | list, spec: dict, base_path: Path
    ) -> dict | list:
        """Resolve $ref references in OpenAPI spec."""
        if isinstance(obj, dict):
            if "$ref" in obj:
                ref = obj["$ref"]
                if ref.startswith("#/"):
                    parts = ref[2:].split("/")
                    resolved = spec
                    for part in parts:
                        if isinstance(resolved, dict) and part in resolved:
                            resolved = resolved[part]
                        else:
                            return obj
                    return resolved
                elif ref.startswith("./") or not ref.startswith("#"):
                    ref_path = base_path.parent / ref.split("#")[0]
                    if ref_path.exists():
                        ref_spec = self._load_yaml(ref_path)
                        if ref_spec and "#/" in ref:
                            parts = ref.split("#/")[1].split("/")
                            resolved = ref_spec
                            for part in parts:
                                if isinstance(resolved, dict) and part in resolved:
                                    resolved = resolved[part]
                                else:
                                    return obj
                            return resolved
                        return ref_spec or obj
            return {k: self._resolve_refs(v, spec, base_path) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._resolve_refs(item, spec, base_path) for item in obj]
        return obj

    def _parse_endpoint(
        self, path: str, method: str, operation: dict, source_file: Path
    ) -> APIEndpoint:
        """Parse a single API endpoint."""
        summary = operation.get("summary", "")
        description = operation.get("description", "")
        tags = operation.get("tags", [])
        parameters = operation.get("parameters", [])
        request_body = operation.get("requestBody")
        responses = operation.get("responses", {})
        operation_id = operation.get("operationId")

        return APIEndpoint(
            path=path,
            method=method.upper(),
            summary=summary,
            description=description,
            tags=tags,
            parameters=parameters,
            request_body=request_body,
            responses=responses,
            operation_id=operation_id,
            source_file=source_file,
        )

    def _parse_schema(
        self, name: str, schema: dict, source_file: Path
    ) -> APISchema:
        """Parse a single schema definition."""
        return APISchema(
            name=name,
            title=schema.get("title", name),
            description=schema.get("description", ""),
            schema_type=schema.get("type", "object"),
            properties=schema.get("properties", {}),
            required=schema.get("required", []),
            source_file=source_file,
        )

    def _parse_openapi_file(self, file_path: Path) -> ParsedOpenAPI | None:
        """Parse a single OpenAPI specification file."""
        spec = self._load_yaml(file_path)
        if not spec:
            return None

        if "openapi" not in spec and "swagger" not in spec:
            return None

        info = spec.get("info", {})
        title = info.get("title", file_path.stem)
        version = info.get("version", "unknown")
        description = info.get("description", "")

        endpoints: list[APIEndpoint] = []
        schemas: list[APISchema] = []

        paths = spec.get("paths", {})
        for path, path_item in paths.items():
            if isinstance(path_item, dict):
                if "$ref" in path_item:
                    path_item = self._resolve_refs(path_item, spec, file_path)
                    if not isinstance(path_item, dict):
                        continue

                for method in ["get", "post", "put", "delete", "patch", "options", "head"]:
                    if method in path_item:
                        operation = path_item[method]
                        if "$ref" in operation:
                            operation = self._resolve_refs(operation, spec, file_path)
                        if isinstance(operation, dict):
                            endpoint = self._parse_endpoint(
                                path, method, operation, file_path
                            )
                            endpoints.append(endpoint)

        components = spec.get("components", {})
        schema_defs = components.get("schemas", {})
        for schema_name, schema_def in schema_defs.items():
            if isinstance(schema_def, dict):
                schema = self._parse_schema(schema_name, schema_def, file_path)
                schemas.append(schema)

        definitions = spec.get("definitions", {})
        for schema_name, schema_def in definitions.items():
            if isinstance(schema_def, dict):
                schema = self._parse_schema(schema_name, schema_def, file_path)
                schemas.append(schema)

        tags = spec.get("tags", [])

        return ParsedOpenAPI(
            title=title,
            version=version,
            description=description,
            endpoints=endpoints,
            schemas=schemas,
            source_file=file_path,
            tags=tags,
        )

    def load_specs(self) -> list[ParsedOpenAPI]:
        """Load all OpenAPI specifications from the repository."""
        specs: list[ParsedOpenAPI] = []

        if not self.repo_path.exists():
            logger.warning("repo_not_found", path=str(self.repo_path))
            return specs

        if self.spec_file:
            spec_path = self.repo_path / self.spec_file
            if spec_path.exists():
                parsed = self._parse_openapi_file(spec_path)
                if parsed:
                    specs.append(parsed)
        else:
            for pattern in self.spec_patterns:
                for yaml_file in self.repo_path.rglob(pattern):
                    if ".github" in str(yaml_file):
                        continue
                    parsed = self._parse_openapi_file(yaml_file)
                    if parsed and (parsed.endpoints or parsed.schemas):
                        specs.append(parsed)

        logger.info(f"loaded_{self.source_name}_specs", count=len(specs))
        return specs

    def endpoint_to_markdown(self, endpoint: APIEndpoint) -> str:
        """Convert an endpoint to markdown documentation."""
        lines = [
            f"## {endpoint.method} {endpoint.path}",
            "",
        ]

        if endpoint.summary:
            lines.append(f"**Summary:** {endpoint.summary}")
            lines.append("")

        if endpoint.tags:
            lines.append(f"**Tags:** {', '.join(endpoint.tags)}")
            lines.append("")

        if endpoint.description:
            lines.append(endpoint.description)
            lines.append("")

        if endpoint.parameters:
            lines.append("### Parameters")
            lines.append("")
            for param in endpoint.parameters:
                if isinstance(param, dict):
                    name = param.get("name", "unknown")
                    location = param.get("in", "query")
                    required = param.get("required", False)
                    desc = param.get("description", "")
                    schema = param.get("schema", {})
                    param_type = schema.get("type", "string") if isinstance(schema, dict) else "string"

                    req_marker = " (required)" if required else ""
                    lines.append(f"- **{name}** ({location}, {param_type}){req_marker}: {desc}")
            lines.append("")

        if endpoint.request_body:
            lines.append("### Request Body")
            lines.append("")
            if isinstance(endpoint.request_body, dict):
                desc = endpoint.request_body.get("description", "")
                if desc:
                    lines.append(desc)
                content = endpoint.request_body.get("content", {})
                for media_type, media_spec in content.items():
                    lines.append(f"**Content-Type:** {media_type}")
                    if isinstance(media_spec, dict):
                        schema = media_spec.get("schema", {})
                        if isinstance(schema, dict):
                            if "type" in schema:
                                lines.append(f"**Type:** {schema['type']}")
            lines.append("")

        if endpoint.responses:
            lines.append("### Responses")
            lines.append("")
            for code, response in endpoint.responses.items():
                if isinstance(response, dict):
                    desc = response.get("description", "")
                    lines.append(f"- **{code}**: {desc}")
            lines.append("")

        return "\n".join(lines)

    def schema_to_markdown(self, schema: APISchema) -> str:
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
                        prop_type = f"→ {ref_name}"

                    display_name = prop_title if prop_title else prop_name
                    lines.append(f"- **{prop_name}** ({prop_type}){required_marker}: {prop_desc or display_name}")
            lines.append("")

        return "\n".join(lines)
