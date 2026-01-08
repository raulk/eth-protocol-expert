"""Deduplication module - Detect near-duplicate content across sources."""

from .dedup_service import (
    DedupService,
    DuplicatePair,
    MinHasher,
    SimHasher,
)

__all__ = [
    "DedupService",
    "DuplicatePair",
    "MinHasher",
    "SimHasher",
]
