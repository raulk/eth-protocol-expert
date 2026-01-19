"""Alias Table - Store canonical concept to aliases mapping (Phase 7)."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar


@dataclass
class AliasEntry:
    """An entry mapping a canonical concept to its aliases."""

    canonical_id: str
    canonical_name: str
    aliases: list[str] = field(default_factory=list)
    eip_numbers: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "canonical_id": self.canonical_id,
            "canonical_name": self.canonical_name,
            "aliases": self.aliases,
            "eip_numbers": self.eip_numbers,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AliasEntry":
        return cls(
            canonical_id=data["canonical_id"],
            canonical_name=data["canonical_name"],
            aliases=data.get("aliases", []),
            eip_numbers=data.get("eip_numbers", []),
        )


class AliasTable:
    """Bidirectional mapping between canonical concepts and their aliases.

    Supports:
    - Adding aliases to canonical concepts
    - Resolving any alias to its canonical name
    - Retrieving all aliases for a canonical concept
    - Persistence to JSON file
    """

    DEFAULT_ALIASES: ClassVar[dict[str, tuple[str, list[str], list[int]]]] = {
        "EIP-4844": (
            "Proto-Danksharding",
            ["proto-danksharding", "blob transactions", "danksharding-lite", "blobs", "blob tx"],
            [4844],
        ),
        "EIP-1559": (
            "London Fee Market",
            ["london fee market", "base fee", "eip1559", "fee burn", "type 2 transactions"],
            [1559],
        ),
        "EIP-3675": (
            "The Merge",
            ["the merge", "paris upgrade", "pos transition", "proof of stake", "ethereum 2.0"],
            [3675],
        ),
        "EIP-4337": (
            "Account Abstraction",
            ["account abstraction", "aa", "smart accounts", "bundler", "useroperation"],
            [4337],
        ),
        "EIP-721": (
            "NFT Standard",
            ["nft standard", "nfts", "non-fungible tokens", "erc721", "erc-721"],
            [721],
        ),
        "EIP-20": (
            "Token Standard",
            ["erc20", "erc-20", "token standard", "fungible tokens", "eip20"],
            [20],
        ),
        "EIP-2930": (
            "Access List Transaction",
            ["access list", "type 1 transaction", "berlin transaction"],
            [2930],
        ),
        "EIP-4895": (
            "Beacon Chain Withdrawals",
            ["beacon withdrawals", "staking withdrawals", "shanghai withdrawals"],
            [4895],
        ),
        "EIP-6780": (
            "SELFDESTRUCT Restriction",
            ["selfdestruct", "self destruct", "dencun selfdestruct"],
            [6780],
        ),
        "EIP-7702": (
            "EOA Code Delegation",
            ["eoa delegation", "set code transaction", "eoa to smart account"],
            [7702],
        ),
    }

    def __init__(self) -> None:
        self._entries: dict[str, AliasEntry] = {}
        self._alias_index: dict[str, str] = {}
        self._populate_defaults()

    def _populate_defaults(self) -> None:
        for canonical_id, (canonical_name, aliases, eip_numbers) in self.DEFAULT_ALIASES.items():
            entry = AliasEntry(
                canonical_id=canonical_id,
                canonical_name=canonical_name,
                aliases=list(aliases),
                eip_numbers=list(eip_numbers),
            )
            self._entries[canonical_id] = entry
            self._index_entry(entry)

    def _index_entry(self, entry: AliasEntry) -> None:
        canonical_lower = entry.canonical_id.lower()
        self._alias_index[canonical_lower] = entry.canonical_id
        self._alias_index[entry.canonical_name.lower()] = entry.canonical_id

        for alias in entry.aliases:
            self._alias_index[alias.lower()] = entry.canonical_id

        for eip_num in entry.eip_numbers:
            self._alias_index[f"eip{eip_num}"] = entry.canonical_id
            self._alias_index[f"eip-{eip_num}"] = entry.canonical_id
            self._alias_index[f"erc{eip_num}"] = entry.canonical_id
            self._alias_index[f"erc-{eip_num}"] = entry.canonical_id

    def add_alias(self, canonical: str, alias: str) -> None:
        """Add an alias for a canonical concept.

        Creates a new entry if the canonical doesn't exist.
        """
        canonical_upper = canonical.upper() if canonical.startswith(("eip", "EIP")) else canonical

        if canonical_upper not in self._entries:
            eip_num = self._extract_eip_number(canonical_upper)
            eip_numbers = [eip_num] if eip_num else []
            entry = AliasEntry(
                canonical_id=canonical_upper,
                canonical_name=canonical_upper,
                aliases=[],
                eip_numbers=eip_numbers,
            )
            self._entries[canonical_upper] = entry
            self._index_entry(entry)

        entry = self._entries[canonical_upper]
        alias_lower = alias.lower()
        if alias_lower not in [a.lower() for a in entry.aliases]:
            entry.aliases.append(alias)
            self._alias_index[alias_lower] = canonical_upper

    def _extract_eip_number(self, text: str) -> int | None:
        import re

        match = re.search(r"(?:eip|erc)-?(\d+)", text, re.IGNORECASE)
        return int(match.group(1)) if match else None

    def resolve(self, term: str) -> str | None:
        """Resolve a term to its canonical name.

        Returns None if the term is not recognized.
        """
        return self._alias_index.get(term.lower())

    def get_aliases(self, canonical: str) -> list[str]:
        """Get all aliases for a canonical concept."""
        entry = self._entries.get(
            canonical.upper() if canonical.startswith(("eip", "EIP")) else canonical
        )
        return list(entry.aliases) if entry else []

    def get_entry(self, canonical: str) -> AliasEntry | None:
        """Get the full entry for a canonical concept."""
        return self._entries.get(
            canonical.upper() if canonical.startswith(("eip", "EIP")) else canonical
        )

    def get_eip_numbers(self, term: str) -> list[int]:
        """Get EIP numbers associated with a term."""
        canonical = self.resolve(term)
        if not canonical:
            eip_num = self._extract_eip_number(term)
            return [eip_num] if eip_num else []

        entry = self._entries.get(canonical)
        return list(entry.eip_numbers) if entry else []

    def all_entries(self) -> list[AliasEntry]:
        """Get all entries in the table."""
        return list(self._entries.values())

    def save(self, path: Path | str) -> None:
        """Save the alias table to a JSON file."""
        path = Path(path)
        data = {
            "entries": [entry.to_dict() for entry in self._entries.values()],
        }
        path.write_text(json.dumps(data, indent=2))

    def load(self, path: Path | str) -> None:
        """Load the alias table from a JSON file.

        Merges loaded data with existing entries.
        """
        path = Path(path)
        if not path.exists():
            return

        data = json.loads(path.read_text())
        for entry_data in data.get("entries", []):
            entry = AliasEntry.from_dict(entry_data)
            self._entries[entry.canonical_id] = entry
            self._index_entry(entry)

    @classmethod
    def from_file(cls, path: Path | str) -> "AliasTable":
        """Create an alias table from a JSON file."""
        table = cls()
        table.load(path)
        return table
