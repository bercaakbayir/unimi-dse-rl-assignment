"""Energy Thief agents."""

from energy_thief.agents.q_learning import QLearningAgent
from energy_thief.agents.linear_q_learning import LinearQLearningAgent

__all__ = ["QLearningAgent", "LinearQLearningAgent"]

try:  # DQN needs torch; keep the package importable without it.
    from energy_thief.agents.dqn import DQNAgent, QNetwork, ReplayBuffer
    __all__ += ["DQNAgent", "QNetwork", "ReplayBuffer"]
except ModuleNotFoundError:  # pragma: no cover
    pass
