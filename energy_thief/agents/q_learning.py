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
        if not greedy and self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.n_actions))
        q = self.Q[state]
        return int(self._rng.choice(np.flatnonzero(q == q.max())))

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        terminated: bool) -> None:
        bootstrap = 0.0 if terminated else self.Q[next_state].max() # boostrap = 0 if terminated, maxQ(s',a) otherwise
        td_target = reward + self.gamma * bootstrap # y - r + gamma * boostrap
        self.Q[state, action] += self.alpha * (td_target - self.Q[state, action]) # Q <- Q + alpha * (y - Q)
        
        # Q(s,a) <- Q(s,a) + alpha * (r + gamma * maxQ(s', a') - Q(s,a))

    def end_episode(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        #epsilon <- max(epsilon_min, epsilon * epsilon_decay)

    def greedy_policy(self) -> np.ndarray:
        # pi(s,a) = argmaxQ(s,a)
        return self.Q.argmax(axis=1)
    

    def state_values(self) -> np.ndarray:
        # V(s,a) = maxQ(s,a)
        return self.Q.max(axis=1)
