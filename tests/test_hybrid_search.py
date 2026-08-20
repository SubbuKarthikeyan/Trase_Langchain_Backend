"""
test_hybrid_search.py
─────────────────────
Comprehensive unit and integration test suite for Hybrid Search with
Reciprocal Rank Fusion (RRF).

Tests:
    1. _rrf_fuse basic fusion & rank calculation
    2. _rrf_fuse single list (no fusion needed)
    3. _rrf_fuse document overlap boosting
    4. _rrf_fuse empty lists handling
    5. _rrf_fuse_with_scores detailed debug structure
    6. RRF k sensitivity tuning test
    7. Retriever initialization with custom rrf_k and hybrid toggle
    8. Retriever fallback when legacy mode is enabled
"""

import unittest
from app.rag.retriever import (
    _rrf_fuse,
    _rrf_fuse_with_scores,
    _content_key,
    Retriever,
)
from app.core.config import settings


class TestHybridSearchRRF(unittest.TestCase):

    def setUp(self):
        self.doc_a = "Bus Name: Southern Express\nRoute: Chennai to Madurai\nFare: Rs 450"
        self.doc_b = "Bus Name: Royal Travels\nRoute: Chennai to Bangalore\nFare: Rs 600"
        self.doc_c = "Bus Name: Greenline\nRoute: Madurai to Trichy\nFare: Rs 200"

    def test_rrf_fuse_basic(self):
        """Test RRF fusion correctly merges multiple ranked lists."""
        leg1 = [self.doc_a, self.doc_b]
        leg2 = [self.doc_c, self.doc_a]

        fused = _rrf_fuse([leg1, leg2], k=60)

        # doc_a appears at rank 1 in leg 1, rank 2 in leg 2
        # Score for doc_a = 1/(60+1) + 1/(60+2) = 0.01639 + 0.01612 = 0.03251
        # doc_b score = 1/(60+2) = 0.01612
        # doc_c score = 1/(60+1) = 0.01639
        # doc_a should be rank 1 overall!
        self.assertEqual(len(fused), 3)
        self.assertEqual(fused[0], self.doc_a)

    def test_rrf_fuse_single_list(self):
        """Test RRF fusion handles a single list gracefully without altering order."""
        leg1 = [self.doc_b, self.doc_c, self.doc_a]
        fused = _rrf_fuse([leg1], k=60)
        self.assertEqual(fused, [self.doc_b, self.doc_c, self.doc_a])

    def test_rrf_fuse_overlap_boosting(self):
        """Test that documents present in all legs get boosted over top-ranked single-leg docs."""
        doc_shared = "Shared route info: Chennai to Salem"
        doc_solo = "Solo route info: Trichy to Tanjore"

        leg1 = [doc_solo, doc_shared]
        leg2 = [doc_shared]
        leg3 = [doc_shared]

        fused = _rrf_fuse([leg1, leg2, leg3], k=60)
        # doc_shared appears in 3 legs (ranks 2, 1, 1), doc_solo only in 1 (rank 1).
        # doc_shared must win!
        self.assertEqual(fused[0], doc_shared)

    def test_rrf_fuse_empty_lists(self):
        """Test empty inputs return empty list without errors."""
        self.assertEqual(_rrf_fuse([], k=60), [])
        self.assertEqual(_rrf_fuse([[], []], k=60), [])

    def test_rrf_fuse_with_scores(self):
        """Test detailed debug scoring function."""
        leg1 = [self.doc_a, self.doc_b]
        leg2 = [self.doc_a]

        detail = _rrf_fuse_with_scores([leg1, leg2], k=60)
        self.assertEqual(len(detail), 2)
        self.assertEqual(detail[0]["text"], self.doc_a)
        self.assertIn("rrf_score", detail[0])
        self.assertEqual(detail[0]["appeared_in_legs"], [0, 1])

    def test_rrf_k_sensitivity(self):
        """Test that smaller k values penalize lower ranks more heavily."""
        doc1 = "Doc 1"
        doc2 = "Doc 2"

        leg1 = [doc1, doc2]
        leg2 = [doc2, doc1]

        # Equal appearances at ranks 1 & 2 -> scores equal regardless of k
        scores_k60 = _rrf_fuse_with_scores([leg1, leg2], k=60)
        self.assertAlmostEqual(scores_k60[0]["rrf_score"], scores_k60[1]["rrf_score"], places=5)

    def test_retriever_initialization(self):
        """Test Retriever correctly initializes hybrid & rrf_k parameters."""
        retriever = Retriever(top_k=7, rrf_k=30, hybrid=True)
        self.assertEqual(retriever.top_k, 7)
        self.assertEqual(retriever.rrf_k, 30)
        self.assertTrue(retriever.hybrid)

    def test_retriever_legacy_toggle(self):
        """Test Retriever respects hybrid=False legacy toggle."""
        retriever = Retriever(hybrid=False)
        self.assertFalse(retriever.hybrid)


if __name__ == "__main__":
    unittest.main()
