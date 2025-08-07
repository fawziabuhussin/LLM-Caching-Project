"""Online linear regression that predicts token‑level cost.
`predict(x)` returns a float. `partial_fit(x, y)` updates
weights with SGD. Features expected: [prompt_tokens, resp_tokens]."""
import numpy as np
class OnlineLinearRegressor:
    def __init__(self, n_features: int, lr: float = 1e-2, l2: float = 1e-5):
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.lr = lr
        self.l2 = l2

    def predict(self, x):
        return float(np.dot(self.w, x) + self.b)

    def partial_fit(self, x, y):
        y_pred = self.predict(x)
        err = y_pred - y
        self.w -= self.lr * (err * x + self.l2 * self.w)
        self.b -= self.lr * err
