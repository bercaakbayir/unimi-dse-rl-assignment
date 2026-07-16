"""Tabular Q-learning agent (off-policy TD control)."""

from __future__ import annotations

from typing import Optional

import numpy as np


class QLearningAgent:
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: Optional[int] = None,
    ) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.Q = np.zeros((n_states, n_actions))
        self._rng = np.random.default_rng(seed)

    def select_action(self, state: int, *, greedy: bool = False) -> int:
        # epsilon-greedy: explore with prob. epsilon, otherwise act greedily.
        if not greedy and self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.n_actions))
        # random tie-breaking avoids a systematic bias toward low-index actions.
        q = self.Q[state]
        return int(self._rng.choice(np.flatnonzero(q == q.max())))

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        terminated: bool,
        next_action: Optional[int] = None,  # unused; keeps the SARSA interface
    ) -> None:
        # Off-policy target uses max_a Q(s', a): the greedy value regardless of
        # the (exploratory) action actually taken next. No bootstrap on a
        # terminal transition, since there is no future return.
        bootstrap = 0.0 if terminated else self.Q[next_state].max()
        td_target = reward + self.gamma * bootstrap
        self.Q[state, action] += self.alpha * (td_target - self.Q[state, action])

    def end_episode(self) -> None:
        # Anneal exploration toward exploitation as the estimates improve.
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def greedy_policy(self) -> np.ndarray:
        return self.Q.argmax(axis=1)

    def state_values(self) -> np.ndarray:
        return self.Q.max(axis=1)
