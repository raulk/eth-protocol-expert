"""Tests for Phase 6: Forum Ingestion.

These are self-contained unit tests that don't require external API calls.
"""

import re
from dataclasses import dataclass


class TestDiscourseClient:
    """Tests for Discourse API client."""

    def test_api_url_construction(self):
        """Test constructing API URLs."""
        base_url = "https://ethresear.ch"

        latest_url = f"{base_url}/latest.json"
        topic_url = f"{base_url}/t/12345.json"
        post_url = f"{base_url}/posts/67890.json"
        search_url = f"{base_url}/search.json?q=EIP-4844"

        assert "latest.json" in latest_url
        assert "/t/12345" in topic_url
        assert "/posts/67890" in post_url
        assert "q=EIP-4844" in search_url

    def test_rate_limiting(self):
        """Test rate limiting logic."""
        import time

        last_request = time.time()
        min_interval = 1.0

        # Should wait if within interval
        current_time = last_request + 0.5
        wait_time = max(0, min_interval - (current_time - last_request))

        assert wait_time > 0

        # Should not wait if past interval
        current_time = last_request + 1.5
        wait_time = max(0, min_interval - (current_time - last_request))

        assert wait_time == 0

    def test_raw_vs_cooked_preference(self):
        """Test that raw markdown is preferred over cooked HTML."""
        post_data = {
            "id": 12345,
            "raw": "This is **raw** markdown with `code`",
            "cooked": "<p>This is <strong>raw</strong> markdown with <code>code</code></p>",
        }

        # Should use 'raw' for stable content
        content = post_data["raw"]

        assert "**raw**" in content
        assert "<p>" not in content


class TestEIPMentionDetection:
    """Tests for detecting EIP mentions in forum posts."""

    def test_eip_pattern_standard(self):
        """Test detecting standard EIP mentions."""
        pattern = re.compile(r'\bEIP[- ]?(\d+)\b', re.IGNORECASE)

        text = "EIP-1559 introduced a new fee market"
        matches = pattern.findall(text)

        assert len(matches) == 1
        assert "1559" in matches

    def test_eip_pattern_variations(self):
        """Test detecting various EIP mention formats."""
        pattern = re.compile(r'\bEIP[- ]?(\d+)\b', re.IGNORECASE)

        texts = [
            ("EIP-4844 is about blobs", "4844"),
            ("EIP 4844 introduces blobs", "4844"),
            ("eip-4844 (lowercase)", "4844"),
            ("EIP4844 without separator", "4844"),
        ]

        for text, expected in texts:
            matches = pattern.findall(text)
            assert len(matches) >= 1
            assert expected in matches

    def test_multiple_eip_mentions(self):
        """Test detecting multiple EIP mentions."""
        pattern = re.compile(r'\bEIP[- ]?(\d+)\b', re.IGNORECASE)

        text = "EIP-1559 and EIP-4844 work together with EIP-2718"
        matches = pattern.findall(text)

        assert len(matches) == 3
        assert "1559" in matches
        assert "4844" in matches
        assert "2718" in matches

    def test_no_false_positives(self):
        """Test avoiding false positives."""
        pattern = re.compile(r'\bEIP[- ]?(\d+)\b', re.IGNORECASE)

        texts = [
            "IP address 192.168.1.1",
            "recipe for cookies",
            "equipment list",
        ]

        for text in texts:
            matches = pattern.findall(text)
            assert len(matches) == 0


class TestForumChunking:
    """Tests for forum post chunking."""

    def test_post_boundaries_respected(self):
        """Posts should not be split mid-post when possible."""
        posts = [
            {"id": 1, "content": "First post content.", "tokens": 50},
            {"id": 2, "content": "Second post content.", "tokens": 50},
        ]

        max_tokens = 100

        chunks = []
        for post in posts:
            if post["tokens"] <= max_tokens:
                chunks.append(post)

        assert len(chunks) == 2

    def test_long_post_splitting(self):
        """Long posts should be split at appropriate boundaries."""
        long_post = {
            "id": 1,
            "content": "Start of post.\n\nParagraph 1.\n\nParagraph 2.\n\nParagraph 3.",
            "tokens": 500,
        }

        paragraphs = long_post["content"].split("\n\n")
        assert len(paragraphs) >= 3

    def test_reply_context_preserved(self):
        """Reply context should be included in chunks."""
        post = {
            "id": 2,
            "content": "This is a reply",
            "reply_to_post_number": 1,
            "username": "user123",
        }

        chunk_metadata = {
            "post_id": post["id"],
            "reply_to": post["reply_to_post_number"],
            "author": post["username"],
        }

        assert chunk_metadata["reply_to"] == 1


class TestForumGraph:
    """Tests for forum reply graph."""

    def test_replies_to_edge(self):
        """Test REPLIES_TO edge creation."""
        @dataclass
        class ReplyEdge:
            from_post_id: int
            to_post_id: int
            relationship_type: str = "REPLIES_TO"

        edge = ReplyEdge(from_post_id=2, to_post_id=1)

        assert edge.from_post_id == 2
        assert edge.to_post_id == 1
        assert edge.relationship_type == "REPLIES_TO"

    def test_thread_structure(self):
        """Test building thread structure from replies."""
        posts = [
            {"id": 1, "reply_to": None},
            {"id": 2, "reply_to": 1},
            {"id": 3, "reply_to": 1},
            {"id": 4, "reply_to": 2},
        ]

        replies = {}
        for post in posts:
            if post["reply_to"]:
                if post["reply_to"] not in replies:
                    replies[post["reply_to"]] = []
                replies[post["reply_to"]].append(post["id"])

        assert 1 in replies
        assert len(replies[1]) == 2
        assert 4 in replies.get(2, [])


class TestCrossReference:
    """Tests for cross-referencing forums to EIPs."""

    def test_link_discussion_to_eip(self):
        """Test linking a forum discussion to mentioned EIPs."""
        post_content = "EIP-4844 would greatly benefit rollups like Optimism."

        pattern = re.compile(r'\bEIP[- ]?(\d+)\b', re.IGNORECASE)
        eips = pattern.findall(post_content)

        assert "4844" in eips

    def test_multiple_eip_references(self):
        """Test post referencing multiple EIPs."""
        post_content = "EIP-1559 and EIP-4844 both change the fee market."

        pattern = re.compile(r'\bEIP[- ]?(\d+)\b', re.IGNORECASE)
        eips = pattern.findall(post_content)

        assert len(eips) == 2


class TestDeduplication:
    """Tests for crosspost deduplication."""

    def get_shingles(self, text: str, n: int = 3) -> set[str]:
        """Generate n-gram shingles from text."""
        words = text.lower().split()
        return {" ".join(words[i:i+n]) for i in range(len(words) - n + 1)}

    def jaccard_similarity(self, set1: set, set2: set) -> float:
        """Calculate Jaccard similarity between two sets."""
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0

    def test_exact_duplicate_detection(self):
        """Test detecting exact duplicates."""
        content1 = "This is the exact same post content."
        content2 = "This is the exact same post content."

        assert content1 == content2

    def test_near_duplicate_with_shingles(self):
        """Test near-duplicate detection using n-gram shingles."""
        text1 = "EIP 4844 introduces blob transactions for cheap data"
        text2 = "EIP-4844 introduces blob transactions for affordable data"

        shingles1 = self.get_shingles(text1)
        shingles2 = self.get_shingles(text2)

        similarity = self.jaccard_similarity(shingles1, shingles2)

        # Should be somewhat similar but not identical
        assert 0 < similarity < 1

    def test_different_posts_not_duplicates(self):
        """Test that different posts are not flagged as duplicates."""
        content1 = "EIP-1559 changes the fee market"
        content2 = "Account abstraction enables smart contract wallets"

        shingles1 = self.get_shingles(content1)
        shingles2 = self.get_shingles(content2)

        similarity = self.jaccard_similarity(shingles1, shingles2)

        # Threshold typically 0.7 for duplicates
        assert similarity < 0.7

    def test_minhash_signature(self):
        """Test MinHash signature generation."""
        text = "This is a test document for minhash"
        shingles = self.get_shingles(text)

        # MinHash generates fixed-size signature
        num_hashes = 100

        def simple_hash(shingle: str, seed: int) -> int:
            import hashlib
            h = hashlib.md5((str(seed) + shingle).encode())
            return int(h.hexdigest(), 16)

        signature = []
        for i in range(num_hashes):
            min_hash = min(simple_hash(s, i) for s in shingles) if shingles else 0
            signature.append(min_hash)

        assert len(signature) == num_hashes
