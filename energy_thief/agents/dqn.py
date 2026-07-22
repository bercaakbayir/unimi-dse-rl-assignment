from __future__ import annotations
import copy
import random
from collections import deque, namedtuple
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, n_state_features: int, n_actions: int, hidden_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_state_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, state):
        return self.net(state)


Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


class ReplayBuffer:
    def __init__(self, capacity: int, *, seed: Optional[int] = None):
        self._buffer: deque = deque(maxlen=int(capacity))
        self._rng = random.Random(seed)

    def push(self, state, action, reward, next_state, done) -> None:
        self._buffer.append(Transition(
            np.asarray(state, dtype=np.float32), int(action), float(reward),
            np.asarray(next_state, dtype=np.float32), bool(done),
        ))

    def sample(self, batch_size: int):
        batch = self._rng.sample(self._buffer, batch_size)
        states = torch.from_numpy(np.stack([t.state for t in batch]))
        actions = torch.tensor([t.action for t in batch], dtype=torch.int64)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32)
        next_states = torch.from_numpy(np.stack([t.next_state for t in batch]))
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self._buffer)


class DQNAgent:
    def __init__(self, n_state_features: int, n_actions: int, hidden_size: int = 64,
                 lr: float = 1e-3, gamma: float = 0.99, epsilon_start: float = 1.0,
                 epsilon_min: float = 0.05, epsilon_decay: float = 0.995,
                 buffer_capacity: int = 50_000, batch_size: int = 64,
                 min_buffer_size: int = 1000, target_sync_every: int = 500,
                 seed: Optional[int] = None):
        self.n_actions = int(n_actions)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon_start)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)

        if seed is not None:
            torch.manual_seed(seed)
        self._rng = np.random.default_rng(seed)

        self.q_net = QNetwork(n_state_features, n_actions, hidden_size)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.target_net = copy.deepcopy(self.q_net)
        for p in self.target_net.parameters():
            p.requires_grad_(False)
        self.target_sync_every = int(target_sync_every)
        self._update_count = 0

        self.batch_size = int(batch_size)
        self.min_buffer_size = int(min_buffer_size)
        self.buffer = ReplayBuffer(buffer_capacity, seed=seed)

    def q_values(self, state) -> np.ndarray:
        state_t = torch.as_tensor(np.asarray(state, dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            return self.q_net(state_t).squeeze(0).numpy()

    def select_action(self, state, *, greedy: bool = False) -> int:
        if not greedy and self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.n_actions))
        q = self.q_values(state)
        return int(self._rng.choice(np.flatnonzero(q == q.max())))

    def update(self, state, action, reward, next_state, terminated, next_action=None) -> None:
        self.buffer.push(state, action, reward, next_state, terminated)
        if len(self.buffer) < self.min_buffer_size:
            return
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        q_sa = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next_max = self.target_net(next_states).max(dim=1).values
            target = rewards + self.gamma * (1.0 - dones) * q_next_max
        loss = ((q_sa - target) ** 2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._update_count += 1
        if self._update_count % self.target_sync_every == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def end_episode(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
