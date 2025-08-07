"""CA‑LRU cache implementation — plug‑and‑play.
`get(key, fetch_fn)` is the main API. Compatible with GPTCache
by wrapping `fetch_fn` around LLM client."""
import time
from collections import OrderedDict
import numpy as np
from cost_estimator import OnlineLinearRegressor
from bandit import EpsilonGreedyBandit

class CALRUCache:
    def __init__(self, capacity=512,
                 arms=(0.0, 0.25, 0.5, 0.75, 1.0),
                 eps=0.05, norm_window=1000):
        self.capacity = capacity
        self.model = OnlineLinearRegressor(n_features=2)
        self.bandit = EpsilonGreedyBandit(arms, eps)
        self.cache = OrderedDict()  # key -> (value, cost_pred, ts)
        self.cost_hist = []
        self.norm_window = norm_window
        self.w = self.bandit.select()

    # -------- public API --------
    def get(self, key, fetch_fn):
        now = time.time()
        if key in self.cache:  # HIT
            val, cost_pred, _ = self.cache.pop(key)
            self.cache[key] = (val, cost_pred, now)
            reward = 1.0
            hit = True
        else:                  # MISS
            val = fetch_fn(key)
            feats = np.array([len(key), len(val)])
            cost_pred = self.model.predict(feats)
            self.model.partial_fit(feats, len(val.split()))
            self._consider_eviction(cost_pred)
            self.cache[key] = (val, cost_pred, now)
            reward = -1.0
            hit = False
        self.bandit.update(self.w, reward)
        self.w = self.bandit.select()
        return val, hit

    # -------- helpers --------
    def _consider_eviction(self, new_cost):
        if len(self.cache) < self.capacity:
            self.cost_hist.append(new_cost)
            return
        mn, mx = min(self.cost_hist), max(self.cost_hist)
        norm = (lambda c: 0.5) if mn == mx else (lambda c: (c - mn) / (mx - mn))
        rec = {k: (i + 1) / len(self.cache) for i, k in enumerate(reversed(self.cache))}
        score = {k: self.w * rec[k] - (1 - self.w) * norm(v[1])
                 for k, v in self.cache.items()}
        victim = min(score, key=score.get)
        self.cache.pop(victim)
        if len(self.cost_hist) >= self.norm_window:
            self.cost_hist.pop(0)
        self.cost_hist.append(new_cost)
