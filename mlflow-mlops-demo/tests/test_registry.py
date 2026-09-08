"""Unit tests for the promotion gate, which has no workspace dependency."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from northpeak_mlops.registry import should_promote


class PromotionGateTests(unittest.TestCase):
    def test_promotes_when_there_is_no_incumbent(self):
        promote, reason = should_promote(0.01, None, 5.0)
        self.assertTrue(promote)
        self.assertIn("No incumbent", reason)

    def test_promotes_when_required_improvement_is_met(self):
        promote, _ = should_promote(0.009, 0.01, 10.0)
        self.assertTrue(promote)

    def test_rejects_weaker_candidate(self):
        promote, reason = should_promote(0.0095, 0.01, 10.0)
        self.assertFalse(promote)
        self.assertIn("exceeded", reason)


if __name__ == "__main__":
    unittest.main()
