# ================================================
# Checkpoint 2 – Baseline vs Extended + Token-aware backend
# Files in this bundle:
#   1) policy.py         (adds LRU baseline; CALRU with optional bandit)
#   2) replay_driver.py  (adds --policy flag and token-aware fake LLM)
# Copy each block into its file path.
# ================================================

# ======================== policy.py ========================
"""Cache policies: LRU (baseline) and CA-LRU (extended).
Both expose the same API:
    get(key, fetch_fn) -> (value, hit: bool)

Usage in driver:
    if args.policy == 'lru':
        cache = LRUCache(capacity=args.capacity)
    elif args.policy == 'calru':
        cache = CALRUCache(capacity=args.capacity, fixed_w=0.5)
    elif args.policy == 'calru_bandit':
        cache = CALRUCache(capacity=args.capacity, arms=(0.0,0.25,0.5,0.75,1.0), eps=0.05)
"""
from __future__ import annotations
import time
from collections import OrderedDict
import numpy as np
from typing import Callable, Optional, Tuple

from cost_estimator import OnlineLinearRegressor
from bandit import EpsilonGreedyBandit

CacheFetchFn = Callable[[str], str]


class LRUCache:
    """Vanilla LRU baseline using OrderedDict.
    Evicts the least-recently-used key (head of the dict) when full.
    """
    def __init__(self, capacity: int = 512):
        self.capacity = int(capacity)
        self.cache: OrderedDict[str, Tuple[str, float]] = OrderedDict()

    def get(self, key: str, fetch_fn: CacheFetchFn) -> Tuple[str, bool]:
        now = time.time()
        if key in self.cache:
            val, _ = self.cache.pop(key)
            self.cache[key] = (val, now)  # move to MRU
            return val, True
        # miss
        val = fetch_fn(key)
        if len(self.cache) >= self.capacity:
            # pop LRU
            self.cache.popitem(last=False)
        self.cache[key] = (val, now)
        return val, False


class CALRUCache:
    """Cost-Aware LRU with optional online bandit for weight w.

    If `fixed_w` is provided, bandit is disabled and w is constant.
    Otherwise provide `arms` and `eps` to enable ε-greedy learning.
    """
    def __init__(
        self,
        capacity: int = 512,
        *,
        fixed_w: Optional[float] = None,
        arms: Tuple[float, ...] = (0.5,),
        eps: float = 0.0,
        norm_window: int = 1000,
    ) -> None:
        self.capacity = int(capacity)
        self.model = OnlineLinearRegressor(2)
        self.cache: OrderedDict[str, Tuple[str, float, float]] = OrderedDict()  # key -> (value, cost_pred, ts)
        self.cost_hist: list[float] = []
        self.norm_window = int(norm_window)

        if fixed_w is not None:
            self.bandit: Optional[EpsilonGreedyBandit] = None
            self.w = float(fixed_w)
        else:
            self.bandit = EpsilonGreedyBandit(arms=arms, eps=eps)
            self.w = self.bandit.select()

    def get(self, key: str, fetch_fn: CacheFetchFn) -> Tuple[str, bool]:
        now = time.time()
        if key in self.cache:  # HIT
            val, cost_pred, _ = self.cache.pop(key)
            self.cache[key] = (val, cost_pred, now)
            self._bandit_update(+1.0)
            return val, True

        # MISS → fetch and learn
        val = fetch_fn(key)
        feats = np.array([len(key), len(val)], dtype=float)
        cost_pred = self.model.predict(feats)
        # use whitespace tokens as a rough cost target for now
        self.model.partial_fit(feats, float(max(1, len(val.split()))))
        self._consider_eviction(cost_pred)
        self.cache[key] = (val, cost_pred, now)
        self._bandit_update(-1.0)
        return val, False

    # ----------------- helpers -----------------
    def _bandit_update(self, reward: float) -> None:
        if self.bandit is None:
            return
        self.bandit.update(self.w, reward)
        self.w = self.bandit.select()

    def _consider_eviction(self, new_cost: float) -> None:
        # room available
        if len(self.cache) < self.capacity:
            self._push_cost(new_cost)
            return
        # build score for each resident key
        mn, mx = (min(self.cost_hist), max(self.cost_hist)) if self.cost_hist else (0.0, 1.0)
        def norm(c: float) -> float:
            if mx <= mn:
                return 0.5
            return (c - mn) / (mx - mn)
        # recency rank: 1/N .. 1 for LRU..MRU
        n = len(self.cache)
        rec = {k: (i + 1) / n for i, k in enumerate(reversed(self.cache))}
        score = {
            k: self.w * rec[k] - (1.0 - self.w) * norm(v[1])
            for k, v in self.cache.items()
        }
        victim = min(score, key=score.get)
        self.cache.pop(victim)
        self._push_cost(new_cost)

    def _push_cost(self, c: float) -> None:
        self.cost_hist.append(float(c))
        if len(self.cost_hist) > self.norm_window:
            self.cost_hist.pop(0)

