"""Spec-Impl Linker - Link EIPs to their implementations in client codebases."""

import asyncio
import os
import re
from dataclasses import dataclass
from typing import ClassVar

import anthropic
import structlog

from src.graph.falkordb_store import FalkorDBStore

logger = structlog.get_logger()


@dataclass
class SpecImplLink:
    """A link between an EIP specification and its implementation."""

    eip_number: int
    file_path: str
    function_name: str
    confidence: float
    evidence: str


class SpecImplLinker:
    """Link EIP specifications to their implementations in client codebases."""

    EIP_PATTERNS: ClassVar[list[str]] = [
        r"EIP[-_]?(\d+)",
        r"eip[-_]?(\d+)",
        r"EIP\s*#?\s*(\d+)",
    ]

    IMPLEMENTATION_KEYWORDS: ClassVar[dict[int, list[str]]] = {
        1559: ["basefee", "base_fee", "BaseFee", "EIP1559", "eip1559", "dynamic fee"],
        4844: ["blob", "Blob", "BLOB", "kzg", "KZG", "EIP4844", "eip4844", "data_gas", "blob_gas"],
        4895: ["withdrawal", "Withdrawal", "EIP4895", "eip4895", "beacon_root"],
        2718: ["typed_transaction", "TypedTransaction", "EIP2718", "tx_type", "TransactionType"],
        2930: ["access_list", "AccessList", "EIP2930", "accessList"],
        3675: ["merge", "Merge", "pos", "proof_of_stake", "terminal_total_difficulty"],
        7516: ["blobbasefee", "blob_base_fee", "BLOBBASEFEE"],
    }

    def __init__(
        self,
        api_key: str | None = None,
        graph_store: FalkorDBStore | None = None,
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.graph_store = graph_store

    async def find_implementations(
        self,
        eip_number: int,
        codebase_files: dict[str, str],
    ) -> list[SpecImplLink]:
        """Find implementations of an EIP in a codebase."""
        candidate_files = self._filter_candidates(eip_number, codebase_files)

        if not candidate_files:
            logger.debug("no_candidates_found", eip=eip_number)
            return []

        links: list[SpecImplLink] = []

        for file_path, content in candidate_files.items():
            file_links = await self._analyze_file(eip_number, file_path, content)
            links.extend(file_links)

        links.sort(key=lambda x: x.confidence, reverse=True)

        logger.info(
            "found_implementations",
            eip=eip_number,
            count=len(links),
        )

        return links

    async def link_all_eips(
        self,
        eip_numbers: list[int],
        codebase_files: dict[str, str],
    ) -> dict[int, list[SpecImplLink]]:
        """Find implementations for multiple EIPs."""
        results: dict[int, list[SpecImplLink]] = {}

        for eip_number in eip_numbers:
            links = await self.find_implementations(eip_number, codebase_files)
            results[eip_number] = links

            if self.graph_store and links:
                self._store_links_in_graph(eip_number, links)

        return results

    def find_eip_references(self, content: str) -> list[int]:
        """Find all EIP numbers referenced in code content."""
        eip_numbers: set[int] = set()

        for pattern in self.EIP_PATTERNS:
            matches = re.findall(pattern, content)
            for match in matches:
                eip_numbers.add(int(match))

        return sorted(eip_numbers)

    def _filter_candidates(
        self,
        eip_number: int,
        codebase_files: dict[str, str],
    ) -> dict[str, str]:
        candidates: dict[str, str] = {}

        eip_str = str(eip_number)
        keywords = self.IMPLEMENTATION_KEYWORDS.get(eip_number, [])

        for file_path, content in codebase_files.items():
            if "_test.go" in file_path or "_test.rs" in file_path:
                continue
            if "/test/" in file_path or "/tests/" in file_path:
                continue

            content_lower = content.lower()

            has_eip_reference = (
                f"eip{eip_str}" in content_lower
                or f"eip-{eip_str}" in content_lower
                or f"eip_{eip_str}" in content_lower
                or f"eip {eip_str}" in content_lower
            )

            has_keyword = any(kw.lower() in content_lower for kw in keywords)

            if has_eip_reference or has_keyword:
                candidates[file_path] = content

        return candidates

    async def _analyze_file(
        self,
        eip_number: int,
        file_path: str,
        content: str,
    ) -> list[SpecImplLink]:
        max_content_len = 8000
        if len(content) > max_content_len:
            content = content[:max_content_len] + "\n... (truncated)"

        prompt = f"""Analyze this code file and identify functions/methods that implement EIP-{eip_number}.

File: {file_path}

```
{content}
```

For each implementation found, respond in this exact format:
FUNCTION: <function_name>
CONFIDENCE: <0.0-1.0>
EVIDENCE: <one line explanation>
---

If no implementations are found, respond with:
NO_IMPLEMENTATIONS_FOUND

Focus on core implementation logic, not just references or imports."""

        try:
            response = await asyncio.to_thread(
                self.client.messages.create,
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )

            return self._parse_llm_response(eip_number, file_path, response.content[0].text)

        except Exception as e:
            logger.warning("llm_analysis_failed", file=file_path, error=str(e))
            return []

    def _parse_llm_response(
        self,
        eip_number: int,
        file_path: str,
        response: str,
    ) -> list[SpecImplLink]:
        if "NO_IMPLEMENTATIONS_FOUND" in response:
            return []

        links: list[SpecImplLink] = []
        entries = response.split("---")

        for entry in entries:
            entry = entry.strip()
            if not entry:
                continue

            func_match = re.search(r"FUNCTION:\s*(.+)", entry)
            conf_match = re.search(r"CONFIDENCE:\s*([\d.]+)", entry)
            evid_match = re.search(r"EVIDENCE:\s*(.+)", entry)

            if func_match and conf_match:
                try:
                    confidence = float(conf_match.group(1))
                    confidence = max(0.0, min(1.0, confidence))
                except ValueError:
                    confidence = 0.5

                links.append(
                    SpecImplLink(
                        eip_number=eip_number,
                        file_path=file_path,
                        function_name=func_match.group(1).strip(),
                        confidence=confidence,
                        evidence=evid_match.group(1).strip() if evid_match else "",
                    )
                )

        return links

    def _store_links_in_graph(self, eip_number: int, links: list[SpecImplLink]) -> None:
        if not self.graph_store:
            return

        for link in links:
            cypher = """
                MERGE (e:EIP {number: $eip_number})
                MERGE (f:CodeFile {path: $file_path})
                MERGE (fn:Function {name: $function_name, file: $file_path})
                MERGE (e)-[r:IMPLEMENTED_BY]->(fn)
                SET r.confidence = $confidence,
                    r.evidence = $evidence
                MERGE (fn)-[:DEFINED_IN]->(f)
            """
            try:
                self.graph_store.query(
                    cypher,
                    params={
                        "eip_number": link.eip_number,
                        "file_path": link.file_path,
                        "function_name": link.function_name,
                        "confidence": link.confidence,
                        "evidence": link.evidence,
                    },
                )
            except Exception as e:
                logger.warning(
                    "failed_to_store_link",
                    eip=eip_number,
                    function=link.function_name,
                    error=str(e),
                )

        logger.debug("stored_implementation_links", eip=eip_number, count=len(links))
