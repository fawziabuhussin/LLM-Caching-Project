"""ε‑Greedy bandit over discrete arms (candidate w values)."""
import random
class EpsilonGreedyBandit:
    def __init__(self, arms, eps: float = 0.05):
        self.arms = list(arms)
        self.eps = eps
        self.counts = {a: 0 for a in self.arms}
        self.values = {a: 0.0 for a in self.arms}

    def select(self):
        if random.random() < self.eps:
            return random.choice(self.arms)
        return max(self.arms, key=self.values.get)

    def update(self, arm, reward):
        c = self.counts[arm] = self.counts[arm] + 1
        v = self.values[arm]
        self.values[arm] = v + (reward - v) / c
