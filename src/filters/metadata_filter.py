"""Metadata Filter - Filter EIPs by status, category, author, and other fields."""

from dataclasses import dataclass, field
from typing import ClassVar

import structlog

logger = structlog.get_logger()


@dataclass
class MetadataQuery:
    """Query parameters for metadata filtering.

    All fields are optional. When multiple fields are set, they are ANDed together.
    List fields (like statuses) are ORed within the field.
    """

    statuses: list[str] = field(default_factory=list)
    types: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    authors: list[str] = field(default_factory=list)
    eip_numbers: list[int] = field(default_factory=list)
    requires_eip: int | None = None
    superseded_by: int | None = None
    created_after: str | None = None
    created_before: str | None = None

    def is_empty(self) -> bool:
        """Check if no filters are set."""
        return (
            not self.statuses
            and not self.types
            and not self.categories
            and not self.authors
            and not self.eip_numbers
            and self.requires_eip is None
            and self.superseded_by is None
            and self.created_after is None
            and self.created_before is None
        )


class MetadataFilter:
    """Build SQL WHERE clauses from metadata queries.

    Works with the documents table schema:
    - status VARCHAR(64)
    - type VARCHAR(64)
    - category VARCHAR(64)
    - author TEXT
    - eip_number INTEGER
    - requires INTEGER[]
    - superseded_by INTEGER (optional, may need migration)
    - created_date VARCHAR(32)
    """

    VALID_STATUSES: ClassVar[list[str]] = [
        "Draft",
        "Review",
        "Last Call",
        "Final",
        "Stagnant",
        "Withdrawn",
        "Living",
    ]

    VALID_TYPES: ClassVar[list[str]] = [
        "Standards Track",
        "Meta",
        "Informational",
    ]

    VALID_CATEGORIES: ClassVar[list[str]] = [
        "Core",
        "Networking",
        "Interface",
        "ERC",
    ]

    def build_where_clause(
        self,
        query: MetadataQuery,
        param_offset: int = 1,
    ) -> tuple[str, list]:
        """Build a SQL WHERE clause from a metadata query.

        Args:
            query: The metadata filter parameters
            param_offset: Starting index for SQL parameters ($1, $2, ...)

        Returns:
            Tuple of (where_clause_string, parameter_values)
            The where_clause includes "WHERE" if conditions exist, empty string otherwise.
        """
        conditions = []
        params = []
        param_idx = param_offset

        if query.statuses:
            placeholders = ", ".join(f"${param_idx + i}" for i in range(len(query.statuses)))
            conditions.append(f"status IN ({placeholders})")
            params.extend(query.statuses)
            param_idx += len(query.statuses)

        if query.types:
            placeholders = ", ".join(f"${param_idx + i}" for i in range(len(query.types)))
            conditions.append(f"type IN ({placeholders})")
            params.extend(query.types)
            param_idx += len(query.types)

        if query.categories:
            placeholders = ", ".join(f"${param_idx + i}" for i in range(len(query.categories)))
            conditions.append(f"category IN ({placeholders})")
            params.extend(query.categories)
            param_idx += len(query.categories)

        if query.authors:
            # Author field can contain multiple authors, use ILIKE for partial matching
            author_conditions = []
            for _ in query.authors:
                author_conditions.append(f"author ILIKE '%' || ${param_idx} || '%'")
                param_idx += 1
            conditions.append(f"({' OR '.join(author_conditions)})")
            params.extend(query.authors)

        if query.eip_numbers:
            placeholders = ", ".join(f"${param_idx + i}" for i in range(len(query.eip_numbers)))
            conditions.append(f"eip_number IN ({placeholders})")
            params.extend(query.eip_numbers)
            param_idx += len(query.eip_numbers)

        if query.requires_eip is not None:
            # requires is an INTEGER[] array, use @> for contains
            conditions.append(f"requires @> ARRAY[${param_idx}]::integer[]")
            params.append(query.requires_eip)
            param_idx += 1

        if query.superseded_by is not None:
            conditions.append(f"superseded_by = ${param_idx}")
            params.append(query.superseded_by)
            param_idx += 1

        if query.created_after:
            conditions.append(f"created_date >= ${param_idx}")
            params.append(query.created_after)
            param_idx += 1

        if query.created_before:
            conditions.append(f"created_date <= ${param_idx}")
            params.append(query.created_before)
            param_idx += 1

        if not conditions:
            return "", []

        where_clause = "WHERE " + " AND ".join(conditions)

        logger.debug(
            "built_metadata_filter",
            conditions=len(conditions),
            where_clause=where_clause,
        )

        return where_clause, params

    def build_document_ids_subquery(
        self,
        query: MetadataQuery,
        param_offset: int = 1,
    ) -> tuple[str, list]:
        """Build a subquery to get document_ids matching the metadata filter.

        Useful for filtering chunks by their parent document's metadata.

        Returns:
            Tuple of (subquery_string, parameter_values)
        """
        where_clause, params = self.build_where_clause(query, param_offset)

        if not where_clause:
            return "", []

        subquery = f"SELECT document_id FROM documents {where_clause}"
        return subquery, params

    def validate_status(self, status: str) -> bool:
        """Check if a status value is valid."""
        return status in self.VALID_STATUSES

    def validate_type(self, type_: str) -> bool:
        """Check if a type value is valid."""
        return type_ in self.VALID_TYPES

    def validate_category(self, category: str) -> bool:
        """Check if a category value is valid."""
        return category in self.VALID_CATEGORIES
