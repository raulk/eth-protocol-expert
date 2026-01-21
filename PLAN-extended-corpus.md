# Extended corpus ingestion plan

## Overview

Expand the Ethereum Protocol Expert corpus by ingesting additional high-value repositories from the ethereum GitHub organization. This builds on the existing ingestion infrastructure.

## Current corpus (baseline)

| Source | Documents | Chunks |
|--------|-----------|--------|
| EIPs | 881 | 8,538 |
| Consensus specs | 33 | 608 |
| Execution specs | 602 | 1,893 |
| ACD transcripts | 148 | 2,234 |
| arXiv papers | 48 | ~500 |
| ethresear.ch | 50 | 754 |
| **Total** | **1,762** | **14,527** |

## Target repositories

### High priority (protocol specs & design)

| Repo | Stars | Content | Est. Documents |
|------|-------|---------|----------------|
| research | 1,909 | Python implementations, PDFs, design rationale | ~300 |
| devp2p | 1,099 | P2P networking specs (discv4/5, RLPx, ENR) | ~17 |
| execution-apis | 1,083 | JSON-RPC API specs (OpenAPI) | ~50 methods |
| beacon-APIs | 373 | Beacon node REST API specs | ~80 endpoints |
| builder-specs | 217 | MEV/PBS builder specifications | ~20 |

### Medium priority (standards & extensions)

| Repo | Stars | Content | Est. Documents |
|------|-------|---------|----------------|
| ERCs | 636 | Application-level standards (ERC-20, etc.) | ~565 |
| RIPs | 140 | Rollup Improvement Proposals | ~17 |
| portal-network-specs | 354 | Light client / portal network | ~25 |

---

## Phase 1: ERCs and RIPs (trivial - reuse EIP infrastructure)

### 1.1 Create ERC loader

**File:** `src/ingestion/erc_loader.py`

```python
"""Loader for ethereum/ERCs repository - reuses EIP infrastructure."""

from pathlib import Path
from src.ingestion.eip_loader import EIPLoader
from src.ingestion.eip_parser import EIPParser, ParsedEIP

class ERCLoader(EIPLoader):
    """Load ERCs from ethereum/ERCs repository."""

    REPO_URL = "https://github.com/ethereum/ERCs.git"

    def __init__(self, repo_path: str | Path = "data/ERCs") -> None:
        super().__init__(repo_path)
        self.repo_url = self.REPO_URL

    def list_ercs(self) -> list[Path]:
        """List all ERC markdown files."""
        ercs_dir = self.repo_path / "ERCS"
        if not ercs_dir.exists():
            return []
        return sorted(ercs_dir.glob("erc-*.md"))
```

**Chunking:** Reuse existing `SectionChunker`
**Document type:** `erc`
**Source:** `ethereum/ERCs`

### 1.2 Create RIP loader

**File:** `src/ingestion/rip_loader.py`

Same pattern as ERC loader, pointing to `ethereum/RIPs` repo.

**Document type:** `rip`
**Source:** `ethereum/RIPs`

### 1.3 Ingestion scripts

- `scripts/ingest_ercs.py`
- `scripts/ingest_rips.py`

### 1.4 Validation

```bash
uv run python scripts/ingest_ercs.py --limit 50
uv run python scripts/ingest_rips.py
```

**Expected output:** ~565 ERCs, ~17 RIPs

---

## Phase 2: Markdown specification repos (devp2p, portal-network-specs)

### 2.1 Create generic markdown spec loader

**File:** `src/ingestion/markdown_spec_loader.py`

```python
"""Generic loader for markdown specification repositories."""

from dataclasses import dataclass
from pathlib import Path
import subprocess
import re

@dataclass
class MarkdownSpec:
    """A markdown specification document."""
    name: str
    title: str
    content: str
    file_path: Path
    category: str | None = None  # e.g., "discovery", "transport", "capability"

class MarkdownSpecLoader:
    """Load markdown specs from a git repository."""

    def __init__(
        self,
        repo_url: str,
        repo_path: str | Path,
        spec_patterns: list[str] = ["**/*.md"],
        exclude_patterns: list[str] = ["**/README.md", "**/CHANGELOG.md"],
    ) -> None:
        self.repo_url = repo_url
        self.repo_path = Path(repo_path)
        self.spec_patterns = spec_patterns
        self.exclude_patterns = exclude_patterns

    def clone_or_update(self) -> str:
        """Clone or update repository, return commit hash."""
        # Same pattern as ConsensusSpecLoader
        ...

    def load_all_specs(self) -> list[MarkdownSpec]:
        """Load all markdown spec files."""
        specs = []
        for pattern in self.spec_patterns:
            for md_file in self.repo_path.glob(pattern):
                if self._should_exclude(md_file):
                    continue
                spec = self._parse_spec_file(md_file)
                if spec:
                    specs.append(spec)
        return specs

    def _parse_spec_file(self, file_path: Path) -> MarkdownSpec | None:
        """Parse a markdown spec file."""
        content = file_path.read_text(encoding="utf-8")

        # Extract title from first heading
        title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        title = title_match.group(1) if title_match else file_path.stem

        # Derive category from directory structure
        rel_path = file_path.relative_to(self.repo_path)
        category = rel_path.parts[0] if len(rel_path.parts) > 1 else None

        return MarkdownSpec(
            name=file_path.stem,
            title=title,
            content=content,
            file_path=file_path,
            category=category,
        )
```

### 2.2 DevP2P loader

**File:** `src/ingestion/devp2p_loader.py`

```python
class DevP2PLoader(MarkdownSpecLoader):
    """Load devp2p specifications."""

    REPO_URL = "https://github.com/ethereum/devp2p.git"

    # Category mapping based on directory/filename
    CATEGORIES = {
        "discv4": "discovery",
        "discv5": "discovery",
        "dnsdisc": "discovery",
        "enr": "discovery",
        "rlpx": "transport",
        "caps": "capability",
    }

    def __init__(self, repo_path: str | Path = "data/devp2p") -> None:
        super().__init__(
            repo_url=self.REPO_URL,
            repo_path=repo_path,
            spec_patterns=["*.md", "caps/*.md", "discv5/*.md", "enr-entries/*.md"],
            exclude_patterns=["README.md"],
        )
```

**Document type:** `devp2p_spec`
**Metadata:** `protocol_layer` (discovery/transport/capability)

### 2.3 Portal network specs loader

**File:** `src/ingestion/portal_spec_loader.py`

```python
class PortalSpecLoader(MarkdownSpecLoader):
    """Load portal network specifications."""

    REPO_URL = "https://github.com/ethereum/portal-network-specs.git"

    def __init__(self, repo_path: str | Path = "data/portal-network-specs") -> None:
        super().__init__(
            repo_url=self.REPO_URL,
            repo_path=repo_path,
            spec_patterns=[
                "*.md",
                "history/*.md",
                "utp/*.md",
                "jsonrpc/*.md",
                "ping-extensions/**/*.md",
            ],
            exclude_patterns=["README.md", "SPECIFICATION_TEMPLATE.md"],
        )
```

**Document type:** `portal_spec`
**Metadata:** `network_type` (history/beacon/state/verkle)

### 2.4 Ingestion scripts

- `scripts/ingest_devp2p.py`
- `scripts/ingest_portal_specs.py`

### 2.5 Validation

```bash
uv run python scripts/ingest_devp2p.py
uv run python scripts/ingest_portal_specs.py
```

**Expected output:** ~17 devp2p specs, ~25 portal specs

---

## Phase 3: OpenAPI specification repos (execution-apis, beacon-APIs)

### 3.1 Create OpenAPI spec chunker

**File:** `src/chunking/api_spec_chunker.py`

```python
"""Chunker for OpenAPI/YAML API specifications."""

from dataclasses import dataclass
import yaml

@dataclass
class APIMethodChunk:
    """A single API method as a chunk."""
    chunk_id: str
    method_name: str
    namespace: str  # eth_, engine_, debug_, beacon_, etc.
    summary: str
    params: list[dict]
    result: dict
    examples: list[dict]
    content: str  # Formatted text representation
    token_count: int

class APISpecChunker:
    """Chunk OpenAPI specs by method/endpoint."""

    def __init__(self, max_tokens: int = 1024) -> None:
        self.max_tokens = max_tokens

    def chunk_openapi_file(self, yaml_content: str, source: str) -> list[APIMethodChunk]:
        """Parse OpenAPI YAML and create one chunk per method."""
        data = yaml.safe_load(yaml_content)
        chunks = []

        # Handle different YAML structures
        if isinstance(data, list):
            # execution-apis style: list of method definitions
            for method in data:
                chunk = self._method_to_chunk(method, source)
                if chunk:
                    chunks.append(chunk)
        elif "paths" in data:
            # Standard OpenAPI style: paths object
            for path, methods in data.get("paths", {}).items():
                for http_method, spec in methods.items():
                    if http_method.startswith("$"):
                        continue  # Skip $ref
                    chunk = self._endpoint_to_chunk(path, http_method, spec, source)
                    if chunk:
                        chunks.append(chunk)

        return chunks

    def _method_to_chunk(self, method: dict, source: str) -> APIMethodChunk | None:
        """Convert a JSON-RPC method definition to a chunk."""
        name = method.get("name", "unknown")
        namespace = name.split("_")[0] if "_" in name else ""

        # Build human-readable content
        content_parts = [
            f"# {name}",
            f"\n{method.get('summary', '')}",
            "\n## Parameters",
        ]

        for param in method.get("params", []):
            content_parts.append(f"- **{param.get('name')}** ({param.get('required', False) and 'required' or 'optional'})")

        content_parts.append("\n## Result")
        content_parts.append(f"{method.get('result', {}).get('name', 'unknown')}")

        if method.get("examples"):
            content_parts.append("\n## Examples")
            for ex in method["examples"]:
                content_parts.append(f"- {ex.get('name', 'example')}")

        content = "\n".join(content_parts)

        return APIMethodChunk(
            chunk_id=f"{source}-{name}",
            method_name=name,
            namespace=namespace,
            summary=method.get("summary", ""),
            params=method.get("params", []),
            result=method.get("result", {}),
            examples=method.get("examples", []),
            content=content,
            token_count=len(content.split()),  # Approximate
        )
```

### 3.2 Execution APIs loader

**File:** `src/ingestion/execution_apis_loader.py`

```python
class ExecutionAPIsLoader:
    """Load execution layer JSON-RPC API specs."""

    REPO_URL = "https://github.com/ethereum/execution-apis.git"

    def __init__(self, repo_path: str | Path = "data/execution-apis") -> None:
        self.repo_path = Path(repo_path)

    def clone_or_update(self) -> str:
        """Clone or update repository."""
        ...

    def load_all_methods(self) -> list[dict]:
        """Load all JSON-RPC method definitions from YAML files."""
        methods = []
        src_dir = self.repo_path / "src"

        for yaml_file in src_dir.rglob("*.yaml"):
            content = yaml_file.read_text()
            data = yaml.safe_load(content)
            if isinstance(data, list):
                methods.extend(data)

        return methods
```

**Document type:** `execution_api`
**Metadata:** `namespace` (eth, engine, debug, web3)

### 3.3 Beacon APIs loader

**File:** `src/ingestion/beacon_apis_loader.py`

```python
class BeaconAPIsLoader:
    """Load beacon node REST API specs."""

    REPO_URL = "https://github.com/ethereum/beacon-APIs.git"

    def __init__(self, repo_path: str | Path = "data/beacon-APIs") -> None:
        self.repo_path = Path(repo_path)

    def load_all_endpoints(self) -> list[dict]:
        """Load all endpoint definitions."""
        # Parse beacon-node-oapi.yaml and resolve $refs
        ...
```

**Document type:** `beacon_api`
**Metadata:** `api_category` (beacon, validator, node, events, config)

### 3.4 Ingestion scripts

- `scripts/ingest_execution_apis.py`
- `scripts/ingest_beacon_apis.py`

### 3.5 Validation

```bash
uv run python scripts/ingest_execution_apis.py
uv run python scripts/ingest_beacon_apis.py
```

**Expected output:** ~50 execution API methods, ~80 beacon API endpoints

---

## Phase 4: Builder specs

### 4.1 Builder specs loader

**File:** `src/ingestion/builder_specs_loader.py`

Combines patterns from Phase 2 (markdown) and Phase 3 (OpenAPI):
- Load `/specs/{fork}/*.md` as markdown specs
- Load `/apis/*.yaml` as API specs
- Store fork version in metadata

**Document type:** `builder_spec`
**Metadata:** `fork` (bellatrix, capella, deneb), `content_type` (spec, api)

### 4.2 Ingestion script

- `scripts/ingest_builder_specs.py`

### 4.3 Validation

```bash
uv run python scripts/ingest_builder_specs.py
```

**Expected output:** ~20 documents

---

## Phase 5: Research repository

### 5.1 Research repo loader

**File:** `src/ingestion/research_loader.py`

```python
"""Loader for ethereum/research repository."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

@dataclass
class ResearchDocument:
    """A research document (code, paper, or note)."""
    name: str
    topic: str  # Directory name (verkle_trie_eip, casper4, etc.)
    content_type: Literal["python", "pdf", "markdown"]
    content: str
    file_path: Path

class ResearchLoader:
    """Load research code, papers, and notes."""

    REPO_URL = "https://github.com/ethereum/research.git"

    # High-value research topics
    PRIORITY_TOPICS = [
        "verkle_trie_eip",
        "binius",
        "casper4",
        "sharding",
        "erasure_code",
        "mimc_stark",
        "beacon_chain_impl",
        "verkle",
        "ssz",
    ]

    def __init__(self, repo_path: str | Path = "data/research") -> None:
        self.repo_path = Path(repo_path)

    def load_python_files(self) -> list[ResearchDocument]:
        """Load Python implementation files."""
        docs = []
        for py_file in self.repo_path.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            topic = self._extract_topic(py_file)
            docs.append(ResearchDocument(
                name=py_file.stem,
                topic=topic,
                content_type="python",
                content=py_file.read_text(),
                file_path=py_file,
            ))
        return docs

    def load_pdfs(self) -> list[ResearchDocument]:
        """Load PDF papers (requires PDFExtractor)."""
        ...

    def _extract_topic(self, file_path: Path) -> str:
        """Extract research topic from file path."""
        rel_path = file_path.relative_to(self.repo_path)
        return rel_path.parts[0] if len(rel_path.parts) > 1 else "general"
```

### 5.2 Chunking strategy

- **Python files:** Use `CodeChunker` - extract function/class docstrings + signatures
- **PDFs:** Use existing `PDFExtractor` + `PaperChunker`
- **Markdown:** Use `SectionChunker`

### 5.3 Ingestion script

- `scripts/ingest_research.py`

Options:
- `--python-only` - Skip PDF processing
- `--topics verkle,casper` - Filter to specific topics

### 5.4 Validation

```bash
uv run python scripts/ingest_research.py --python-only --limit 100
```

**Expected output:** ~250 Python files, ~50 PDFs (if enabled)

---

## Implementation order

| Phase | Effort | New Components | Documents Added |
|-------|--------|----------------|-----------------|
| 1. ERCs + RIPs | 1 hour | Trivial loader wrappers | ~580 |
| 2. devp2p + portal | 2 hours | `MarkdownSpecLoader` | ~42 |
| 3. execution + beacon APIs | 3 hours | `APISpecChunker` | ~130 |
| 4. builder-specs | 1 hour | Combines above | ~20 |
| 5. research | 3 hours | `ResearchLoader` | ~300 |
| **Total** | **~10 hours** | | **~1,070** |

## Expected final corpus

| Source | Documents | Est. Chunks |
|--------|-----------|-------------|
| EIPs | 881 | 8,500 |
| ERCs | 565 | 5,500 |
| RIPs | 17 | 200 |
| Consensus specs | 33 | 600 |
| Execution specs | 602 | 1,900 |
| Execution APIs | 50 | 300 |
| Beacon APIs | 80 | 500 |
| Builder specs | 20 | 200 |
| devp2p specs | 17 | 200 |
| Portal specs | 25 | 300 |
| ACD transcripts | 148 | 2,200 |
| arXiv papers | 100 | 1,000 |
| ethresear.ch | 200 | 3,000 |
| Research repo | 300 | 2,000 |
| **Total** | **~3,000** | **~26,000** |

## Success criteria

1. All loaders can clone/update repositories without errors
2. All chunkers produce valid chunks with proper metadata
3. Validation queries return results from new document types:
   - "How does discv5 differ from discv4?" → devp2p_spec sources
   - "What is the eth_call API?" → execution_api sources
   - "What is ERC-721?" → erc sources
   - "How does verkle tree work?" → research sources
