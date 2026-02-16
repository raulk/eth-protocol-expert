"""Tests for EIP parser."""

from datetime import datetime
from pathlib import Path

import pytest

from src.ingestion.eip_loader import LoadedEIP
from src.ingestion.eip_parser import EIPParser

# Sample EIP content for testing
SAMPLE_EIP_CONTENT = """---
eip: 4844
title: Shard Blob Transactions
description: Shard Blob Transactions scale data-availability of Ethereum in a simple, forwards-compatible manner.
author: Vitalik Buterin (@vbuterin), Dankrad Feist (@dankrad)
discussions-to: https://ethereum-magicians.org/t/eip-4844-shard-blob-transactions/8430
status: Final
type: Standards Track
category: Core
created: 2022-02-25
requires: 1559, 2718, 2930, 4895
---

## Abstract

This EIP introduces a new transaction type for "blob-carrying transactions".

## Motivation

Rollups are the dominant scaling solution for Ethereum.

## Specification

We introduce a new transaction type with the following format:

```python
class BlobTransaction:
    chain_id: uint256
    nonce: uint64
    max_priority_fee_per_gas: uint256
```

### Parameters

The following parameters are defined:

| Parameter | Value |
|-----------|-------|
| BLOB_TX_TYPE | 0x03 |
| MAX_BLOB_GAS_PER_BLOCK | 786432 |

## Rationale

The rationale for this design is to provide a simple mechanism.

## Backwards Compatibility

This EIP is backwards compatible.

## Security Considerations

There are no known security issues.

## Copyright

Copyright and related rights waived via CC0.
"""


@pytest.fixture
def parser():
    return EIPParser()


@pytest.fixture
def sample_loaded_eip():
    return LoadedEIP(
        eip_number=4844,
        file_path=Path("test/eip-4844.md"),
        raw_content=SAMPLE_EIP_CONTENT,
        git_commit="abc123",
        loaded_at=datetime.utcnow(),
    )


def test_parse_eip_number(parser, sample_loaded_eip):
    """Test that EIP number is correctly parsed."""
    parsed = parser.parse(sample_loaded_eip)
    assert parsed.eip_number == 4844


def test_parse_title(parser, sample_loaded_eip):
    """Test that title is correctly extracted."""
    parsed = parser.parse(sample_loaded_eip)
    assert parsed.title == "Shard Blob Transactions"


def test_parse_status(parser, sample_loaded_eip):
    """Test that status is correctly extracted."""
    parsed = parser.parse(sample_loaded_eip)
    assert parsed.status == "Final"


def test_parse_type(parser, sample_loaded_eip):
    """Test that type is correctly extracted."""
    parsed = parser.parse(sample_loaded_eip)
    assert parsed.type == "Standards Track"


def test_parse_category(parser, sample_loaded_eip):
    """Test that category is correctly extracted."""
    parsed = parser.parse(sample_loaded_eip)
    assert parsed.category == "Core"


def test_parse_author(parser, sample_loaded_eip):
    """Test that author is correctly extracted."""
    parsed = parser.parse(sample_loaded_eip)
    assert "Vitalik Buterin" in parsed.author


def test_parse_requires(parser, sample_loaded_eip):
    """Test that requires field is correctly parsed."""
    parsed = parser.parse(sample_loaded_eip)
    assert parsed.requires == [1559, 2718, 2930, 4895]


def test_parse_sections(parser, sample_loaded_eip):
    """Test that sections are correctly extracted."""
    parsed = parser.parse(sample_loaded_eip)

    section_names = [s.name for s in parsed.sections]
    assert "Abstract" in section_names
    assert "Motivation" in section_names
    assert "Specification" in section_names
    assert "Rationale" in section_names


def test_parse_abstract_content(parser, sample_loaded_eip):
    """Test that abstract section content is extracted."""
    parsed = parser.parse(sample_loaded_eip)
    assert parsed.abstract is not None
    assert "blob-carrying transactions" in parsed.abstract


def test_parse_specification_content(parser, sample_loaded_eip):
    """Test that specification section content is extracted."""
    parsed = parser.parse(sample_loaded_eip)
    assert parsed.specification is not None
    assert "BlobTransaction" in parsed.specification


def test_git_commit_preserved(parser, sample_loaded_eip):
    """Test that git commit is preserved."""
    parsed = parser.parse(sample_loaded_eip)
    assert parsed.git_commit == "abc123"


def test_parse_requires_as_int():
    """Test parsing requires field when it's a single int."""
    parser = EIPParser()
    assert parser._parse_requires(1559) == [1559]


def test_parse_requires_as_list():
    """Test parsing requires field when it's a list."""
    parser = EIPParser()
    assert parser._parse_requires([1559, 2718]) == [1559, 2718]


def test_parse_requires_as_string():
    """Test parsing requires field when it's a string."""
    parser = EIPParser()
    assert parser._parse_requires("1559, 2718") == [1559, 2718]


def test_parse_requires_none():
    """Test parsing requires field when it's None."""
    parser = EIPParser()
    assert parser._parse_requires(None) == []
