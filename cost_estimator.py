"""Online linear regression that predicts token-level cost.
`predict(x)` returns a float. `partial_fit(x, y)` updates
weights with SGD. Features expected: [prompt_tokens, resp_tokens]."""

import numpy as np


class OnlineLinearRegressor:
    def __init__(
        self,
        n_features: int,
        lr: float = 1e-3,
        l2: float = 1e-5,
        max_grad_norm: float = 10.0,
    ):
        # Use float64 for headroom
        self.w = np.zeros(n_features, dtype=np.float64)
        self.b = 0.0
        self.lr = float(lr)
        self.l2 = float(l2)
        self.max_grad_norm = float(max_grad_norm)

    def predict(self, x):
        x = np.asarray(x, dtype=np.float64)
        y = float(np.dot(self.w, x) + self.b)
        # Guard against any numerical weirdness
        if not np.isfinite(y):
            y = float(np.dot(np.nan_to_num(self.w), np.nan_to_num(x)) + float(self.b))
        return y

    def partial_fit(self, x, y):
        x = np.asarray(x, dtype=np.float64)
        y = float(y)

        # Forward & error
        y_pred = float(np.dot(self.w, x) + self.b)
        err = y_pred - y

        # Gradients
        g_w = err * x + self.l2 * self.w     # dL/dw
        g_b = err                             # dL/db

        # Replace NaN/Inf with large finite values (then clip)
        if not np.all(np.isfinite(g_w)):
            g_w = np.nan_to_num(g_w, nan=0.0, posinf=1e6, neginf=-1e6)
        if not np.isfinite(g_b):
            g_b = 0.0 if np.isnan(g_b) else (1e6 if g_b > 0 else -1e6)

        # Clip gradient by global norm to avoid explosions
        gn = np.linalg.norm(g_w)
        if gn > self.max_grad_norm:
            scale = self.max_grad_norm / (gn + 1e-12)
            g_w *= scale
            g_b *= scale

        # SGD step
        self.w -= self.lr * g_w
        self.b -= self.lr * g_b

        # Keep parameters finite/reasonable
        self.w = np.clip(self.w, -1e6, 1e6)
        if not np.isfinite(self.b):
            self.b = float(np.clip(self.b, -1e6, 1e6))
