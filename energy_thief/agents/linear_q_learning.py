from __future__ import annotations
from typing import Optional
import numpy as np


class LinearQLearningAgent:
    def __init__(
        self,
        feature_extractor,
        n_actions: int,
        alpha: float = 0.05,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.999,
        initial_w: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        self.feature_extractor = feature_extractor
        self.n_features = int(feature_extractor.n_features)
        self.n_actions = int(n_actions)

        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon_start)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)

        self.W = np.full((self.n_actions, self.n_features), float(initial_w), dtype=np.float64)
        self._rng = np.random.default_rng(seed)

    def q_values(self, state) -> np.ndarray:
        return self.W @ self.feature_extractor(state)

    def select_action(self, state, *, greedy: bool = False) -> int:
        if not greedy and self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.n_actions))
        q = self.q_values(state)
        return int(self._rng.choice(np.flatnonzero(q == q.max())))

    def update(self, state, action, reward, next_state, terminated, next_action=None) -> None:
        x = self.feature_extractor(state)
        q_sa = float(self.W[action] @ x)
        if terminated:
            target = float(reward)
        else:
            q_next_max = float((self.W @ self.feature_extractor(next_state)).max())
            target = float(reward) + self.gamma * q_next_max
        self.W[action] += self.alpha * (target - q_sa) * x

    def end_episode(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
