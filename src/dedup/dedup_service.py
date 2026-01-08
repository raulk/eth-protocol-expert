"""Dedup Service - MinHash and SimHash based near-duplicate detection."""

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass

import structlog

from ..ingestion.ethresearch_loader import LoadedForumPost

logger = structlog.get_logger()

# Large prime for MinHash
LARGE_PRIME = 2**61 - 1
MAX_HASH = 2**64 - 1


@dataclass
class DuplicatePair:
    """A pair of posts identified as near-duplicates."""
    post_a_id: str  # Format: "{source}-{topic_id}-{post_id}"
    post_b_id: str
    similarity: float
    method: str  # "minhash" or "simhash"


def _tokenize(text: str) -> list[str]:
    """Tokenize text into normalized words."""
    # Lowercase and extract words
    text = text.lower()
    words = re.findall(r"\b\w+\b", text)
    return words


def _shingle(tokens: list[str], n: int = 3) -> set[str]:
    """Create n-gram shingles from tokens."""
    if len(tokens) < n:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


class MinHasher:
    """MinHash implementation for estimating Jaccard similarity.

    MinHash creates a compact signature of a set that can be used to
    efficiently estimate Jaccard similarity between sets.
    """

    def __init__(self, num_hashes: int = 128, shingle_size: int = 3):
        self.num_hashes = num_hashes
        self.shingle_size = shingle_size
        # Generate random hash function parameters
        # Using a deterministic seed for reproducibility
        self._a_params = [
            (i * 0x5bd1e995 + 0x1b873593) % LARGE_PRIME
            for i in range(num_hashes)
        ]
        self._b_params = [
            (i * 0x1b873593 + 0x5bd1e995) % LARGE_PRIME
            for i in range(num_hashes)
        ]

    def _hash_shingle(self, shingle: str) -> int:
        """Hash a shingle to a 64-bit integer."""
        return int(hashlib.md5(shingle.encode()).hexdigest(), 16) % MAX_HASH

    def compute_signature(self, text: str) -> list[int]:
        """Compute MinHash signature for text."""
        tokens = _tokenize(text)
        shingles = _shingle(tokens, self.shingle_size)

        if not shingles:
            return [MAX_HASH] * self.num_hashes

        # Compute MinHash signature
        signature = []
        for i in range(self.num_hashes):
            min_hash = MAX_HASH
            for shingle in shingles:
                h = self._hash_shingle(shingle)
                # Apply hash function: (a * h + b) % LARGE_PRIME
                val = (self._a_params[i] * h + self._b_params[i]) % LARGE_PRIME
                min_hash = min(min_hash, val)
            signature.append(min_hash)

        return signature

    def estimate_similarity(self, sig_a: list[int], sig_b: list[int]) -> float:
        """Estimate Jaccard similarity from two signatures."""
        if len(sig_a) != len(sig_b):
            raise ValueError("Signatures must have same length")

        matches = sum(1 for a, b in zip(sig_a, sig_b, strict=True) if a == b)
        return matches / len(sig_a)


class SimHasher:
    """SimHash implementation for near-duplicate detection.

    SimHash creates a locality-sensitive hash where similar documents
    have similar hashes (small Hamming distance).
    """

    def __init__(self, hash_bits: int = 64):
        self.hash_bits = hash_bits

    def _hash_token(self, token: str) -> int:
        """Hash a token to hash_bits."""
        h = hashlib.sha256(token.encode()).hexdigest()
        return int(h, 16) % (2 ** self.hash_bits)

    def compute_hash(
        self, text: str, weight_fn: Callable[[str], float] | None = None
    ) -> int:
        """Compute SimHash for text.

        Args:
            text: Input text
            weight_fn: Optional function to weight tokens (default: equal weight)
        """
        tokens = _tokenize(text)
        if not tokens:
            return 0

        # Initialize bit counters
        bit_counts = [0.0] * self.hash_bits

        for token in tokens:
            weight = weight_fn(token) if weight_fn else 1.0
            h = self._hash_token(token)

            for i in range(self.hash_bits):
                bit = (h >> i) & 1
                if bit:
                    bit_counts[i] += weight
                else:
                    bit_counts[i] -= weight

        # Convert to hash
        simhash = 0
        for i in range(self.hash_bits):
            if bit_counts[i] > 0:
                simhash |= (1 << i)

        return simhash

    def hamming_distance(self, hash_a: int, hash_b: int) -> int:
        """Compute Hamming distance between two hashes."""
        xor = hash_a ^ hash_b
        return bin(xor).count("1")

    def similarity(self, hash_a: int, hash_b: int) -> float:
        """Compute similarity (1 - normalized Hamming distance)."""
        distance = self.hamming_distance(hash_a, hash_b)
        return 1.0 - (distance / self.hash_bits)


class DedupService:
    """Service for detecting near-duplicate forum posts.

    Uses both MinHash and SimHash for robust duplicate detection.
    Designed to find crossposts between ethresear.ch and ethereum-magicians.org.
    """

    def __init__(
        self,
        minhash_threshold: float = 0.7,
        simhash_threshold: float = 0.85,
        num_minhash: int = 128,
        simhash_bits: int = 64,
    ):
        self.minhash_threshold = minhash_threshold
        self.simhash_threshold = simhash_threshold
        self.minhasher = MinHasher(num_hashes=num_minhash)
        self.simhasher = SimHasher(hash_bits=simhash_bits)

        # Storage for computed signatures
        self._minhash_signatures: dict[str, list[int]] = {}
        self._simhash_values: dict[str, int] = {}
        self._post_content: dict[str, str] = {}

    def _make_post_id(self, post: LoadedForumPost) -> str:
        """Create unique ID for a post."""
        return f"{post.source}-{post.topic_id}-{post.post_id}"

    def index_post(self, post: LoadedForumPost):
        """Index a post for deduplication."""
        post_id = self._make_post_id(post)

        # Skip if already indexed
        if post_id in self._minhash_signatures:
            return

        # Compute and store signatures
        self._minhash_signatures[post_id] = self.minhasher.compute_signature(post.content)
        self._simhash_values[post_id] = self.simhasher.compute_hash(post.content)
        self._post_content[post_id] = post.content

        logger.debug("indexed_post_for_dedup", post_id=post_id)

    def index_posts(self, posts: list[LoadedForumPost]):
        """Index multiple posts."""
        for post in posts:
            self.index_post(post)
        logger.info("indexed_posts_for_dedup", count=len(posts))

    def find_duplicates_for_post(
        self, post: LoadedForumPost, cross_source_only: bool = True
    ) -> list[DuplicatePair]:
        """Find near-duplicates for a specific post.

        Args:
            post: The post to find duplicates for
            cross_source_only: If True, only consider posts from different sources
        """
        self.index_post(post)
        post_id = self._make_post_id(post)
        post_sig = self._minhash_signatures[post_id]
        post_simhash = self._simhash_values[post_id]

        duplicates = []

        for other_id, other_sig in self._minhash_signatures.items():
            if other_id == post_id:
                continue

            # Check cross-source constraint
            if cross_source_only:
                post_source = post_id.split("-")[0]
                other_source = other_id.split("-")[0]
                if post_source == other_source:
                    continue

            # Check MinHash similarity
            minhash_sim = self.minhasher.estimate_similarity(post_sig, other_sig)
            if minhash_sim >= self.minhash_threshold:
                duplicates.append(DuplicatePair(
                    post_a_id=post_id,
                    post_b_id=other_id,
                    similarity=minhash_sim,
                    method="minhash",
                ))
                continue

            # Fallback to SimHash for borderline cases
            other_simhash = self._simhash_values[other_id]
            simhash_sim = self.simhasher.similarity(post_simhash, other_simhash)
            if simhash_sim >= self.simhash_threshold:
                duplicates.append(DuplicatePair(
                    post_a_id=post_id,
                    post_b_id=other_id,
                    similarity=simhash_sim,
                    method="simhash",
                ))

        return duplicates

    def find_all_duplicates(
        self, cross_source_only: bool = True
    ) -> list[DuplicatePair]:
        """Find all near-duplicate pairs among indexed posts."""
        duplicates = []
        seen_pairs: set[tuple[str, str]] = set()

        post_ids = list(self._minhash_signatures.keys())

        for i, post_id_a in enumerate(post_ids):
            sig_a = self._minhash_signatures[post_id_a]
            simhash_a = self._simhash_values[post_id_a]

            for post_id_b in post_ids[i + 1:]:
                # Check cross-source constraint
                if cross_source_only:
                    source_a = post_id_a.split("-")[0]
                    source_b = post_id_b.split("-")[0]
                    if source_a == source_b:
                        continue

                # Avoid duplicate pairs
                pair_key = tuple(sorted([post_id_a, post_id_b]))
                if pair_key in seen_pairs:
                    continue

                sig_b = self._minhash_signatures[post_id_b]

                # Check MinHash similarity
                minhash_sim = self.minhasher.estimate_similarity(sig_a, sig_b)
                if minhash_sim >= self.minhash_threshold:
                    duplicates.append(DuplicatePair(
                        post_a_id=post_id_a,
                        post_b_id=post_id_b,
                        similarity=minhash_sim,
                        method="minhash",
                    ))
                    seen_pairs.add(pair_key)
                    continue

                # Fallback to SimHash
                simhash_b = self._simhash_values[post_id_b]
                simhash_sim = self.simhasher.similarity(simhash_a, simhash_b)
                if simhash_sim >= self.simhash_threshold:
                    duplicates.append(DuplicatePair(
                        post_a_id=post_id_a,
                        post_b_id=post_id_b,
                        similarity=simhash_sim,
                        method="simhash",
                    ))
                    seen_pairs.add(pair_key)

        logger.info("found_duplicates", count=len(duplicates))
        return duplicates

    def get_post_content(self, post_id: str) -> str | None:
        """Get indexed post content by ID."""
        return self._post_content.get(post_id)

    def clear_index(self):
        """Clear all indexed data."""
        self._minhash_signatures.clear()
        self._simhash_values.clear()
        self._post_content.clear()
        logger.info("cleared_dedup_index")

    def index_size(self) -> int:
        """Get number of indexed posts."""
        return len(self._minhash_signatures)
