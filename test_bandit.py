"""Test for EpsilonGreedyBandit class."""
import unittest
from bandit import EpsilonGreedyBandit

class TestEpsilonGreedyBandit(unittest.TestCase):
    def test_bandit_basic(self):
        arms = [0, 1, 2]
        bandit = EpsilonGreedyBandit(arms, eps=0)
        # Initially all values are 0, so any arm can be selected
        self.assertIn(bandit.select(), arms)
        # Update arm 1 with high reward
        bandit.update(1, 10)
        # Now arm 1 should be preferred
        self.assertEqual(bandit.select(), 1)
        # Update arm 2 with even higher reward
        bandit.update(2, 20)
        self.assertEqual(bandit.select(), 2)
        # Test epsilon randomness
        bandit = EpsilonGreedyBandit(arms, eps=1)
        results = set(bandit.select() for _ in range(100))
        self.assertEqual(results, set(arms))

if __name__ == "__main__":
    unittest.main()
