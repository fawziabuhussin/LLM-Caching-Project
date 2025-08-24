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
from typing import Callable, Optional, Tuple

import numpy as np

from cost_estimator import OnlineLinearRegressor
from bandit import EpsilonGreedyBandit

CacheFetchFn = Callable[[str], str]


class LRUCache:
    """Vanilla LRU baseline using OrderedDict.
    Evicts the least-recently-used key (head of the dict) when full.
    """
    def __init__(self, capacity: int = 512):
        self.capacity = int(capacity)
        # key -> (value, ts)
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

    Implementation notes:
    - We model "age" so that higher means *older* (more evictable).
    - Predicted cost is normalized to [0,1]; higher means *more expensive*
      to recompute (we prefer to KEEP expensive entries).
    - Eviction score = w*age - (1-w)*cost; evict the MAX score.
      * w=1.0  -> pure LRU (evict oldest)
      * w=0.0  -> pure cost (evict cheapest)
    - Optional admission gate: if cache is full and predicted cost < admit_tau,
      do not insert on a miss (serve-and-drop).
    """
    def __init__(
        self,
        capacity: int = 512,
        *,
        fixed_w: Optional[float] = None,
        arms: Tuple[float, ...] = (0.5,),
        eps: float = 0.0,
        norm_window: int = 1000,
        admit_tau: float = 0.0,     # 0.0 keeps prior behavior; try 0.3–0.5 to skip cheap one-shots
    ) -> None:
        self.capacity = int(capacity)
        self.model = OnlineLinearRegressor(2)
        # key -> (value, cost_pred, ts)
        self.cache: OrderedDict[str, Tuple[str, float, float]] = OrderedDict()
        self.cost_hist: list[float] = []
        self.norm_window = int(norm_window)
        self.admit_tau = float(admit_tau)

        if fixed_w is not None:
            self.bandit: Optional[EpsilonGreedyBandit] = None
            self.w = float(fixed_w)
        else:
            self.bandit = EpsilonGreedyBandit(arms=arms, eps=eps)
            self.w = self.bandit.select()

    def get(self, key: str, fetch_fn: CacheFetchFn) -> Tuple[str, bool]:
        now = time.time()

        # HIT
        if key in self.cache:
            val, cost_pred, _ = self.cache.pop(key)
            self.cache[key] = (val, cost_pred, now)  # move to MRU
            self._bandit_update(+1.0)
            return val, True

        # MISS → fetch and learn
        val = fetch_fn(key)

        # simple feature vector; keep float64 for stability
        feats = np.array([len(key), len(val)], dtype=np.float64)

        # predict cost BEFORE learning (policy uses the prediction for eviction/admission)
        cost_pred = float(self.model.predict(feats))

        # optional admission control when full: skip cheap predicted items
        if len(self.cache) >= self.capacity and cost_pred < self.admit_tau:
            # learn from the observation anyway
            self.model.partial_fit(feats, float(max(1, len(val.split()))))
            self._bandit_update(-1.0)
            return val, False

        # learn a simple target (whitespace tokens ≈ cost proxy)
        self.model.partial_fit(feats, float(max(1, len(val.split()))))

        # make room if needed, tracking cost range statistics
        self._consider_eviction(cost_pred)

        # insert at MRU
        self.cache[key] = (val, cost_pred, now)

        # bandit: negative reward for a miss
        self._bandit_update(-1.0)
        return val, False

    # ----------------- helpers -----------------
    def _bandit_update(self, reward: float) -> None:
        if self.bandit is None:
            return
        self.bandit.update(self.w, reward)
        self.w = self.bandit.select()

    def _consider_eviction(self, new_cost: float) -> None:
        """Evict one item if cache is full.
        Eviction score = w*AGE - (1-w)*COST01, evict MAX score.
        """
        # room available
        if len(self.cache) < self.capacity:
            self._push_cost(new_cost)
            return

        # ---- normalize predicted cost to [0,1] over recent history ----
        mn, mx = (min(self.cost_hist), max(self.cost_hist)) if self.cost_hist else (0.0, 1.0)

        def norm(c: float) -> float:
            # clamp and normalize; avoid div-by-zero
            c = float(c)
            if mx <= mn:
                return 0.5
            if c < mn:
                c = mn
            elif c > mx:
                c = mx
            return (c - mn) / (mx - mn)

        # ---- AGE (older => larger): OrderedDict is LRU..MRU; reversed(...) is MRU..LRU ----
        n = len(self.cache)
        age = {k: (i + 1) / n for i, k in enumerate(reversed(self.cache))}
        # MRU gets ~1/n (small), LRU gets 1.0 (largest)

        # ---- eviction score & victim selection (evict MAX score) ----
        def score_key(k: str) -> float:
            cost01 = norm(self.cache[k][1])  # cached predicted cost
            return self.w * age[k] - (1.0 - self.w) * cost01

        victim = max(self.cache, key=score_key)
        self.cache.pop(victim)

        # track cost range with the newly observed prediction
        self._push_cost(new_cost)

    def _push_cost(self, c: float) -> None:
        self.cost_hist.append(float(c))
        if len(self.cost_hist) > self.norm_window:
            self.cost_hist.pop(0)
