"""Energy Thief environment -- Level 1 (small discrete grid, tabular-friendly).

This is the first of three complexity levels of the VFA-4 "Energy Thief"
project. The agent is a thief that moves on a small 2D grid representing a
power grid. Some cells are "tappable" power nodes with a fixed *value*; the
thief siphons energy at its current cell into an (unbanked) *surplus*. The
harder it siphons, the more energy it gets, but the higher the chance a
monitoring system raises an alarm. An alarm wipes the accumulated surplus.
The thief only keeps its loot if it escapes through the *exit* cell.

Level 1 is deliberately simple so that a **tabular** agent can solve it:

* the grid is small (default 5x5);
* the value map is fixed for the whole run (generated once at construction),
  so the thief can *memorise* where the rich cells are;
* the monitoring probability depends only on the siphon intensity, not on a
  per-cell suspicion history (that adaptive "heat" mechanic is introduced in
  Level 2, where it blows up the state space and motivates function
  approximation).

The observation is a single integer, so the environment plugs directly into
``rlc``'s tabular ``QLearningAgent`` / ``SarsaAgent`` and the shared
``train`` / ``evaluate`` loop.

MDP summary
-----------
State   : (agent position, unbanked surplus U in {0, ..., surplus_max})
Actions : 0=up 1=down 2=left 3=right 4=siphon-low 5=siphon-high
Reward  : 0 on ordinary steps;
          +U when the thief steps onto the exit (cash out, episode ends);
          -alarm_penalty when an alarm fires (and U is reset to 0).
Horizon : the episode truncates after ``max_steps`` steps; surplus not cashed
          out by then is lost.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

try:  # gymnasium is the standard interface; fall back gracefully if absent.
    import gymnasium as gym
    from gymnasium import spaces
    _Base = gym.Env
except ModuleNotFoundError:  # pragma: no cover - lets the file import bare.
    gym = None
    spaces = None
    _Base = object


# Action constants (module-level so notebooks/tests can refer to them by name).
UP, DOWN, LEFT, RIGHT, SIPHON_LOW, SIPHON_HIGH = range(6)
_MOVES = {
    UP: (-1, 0),
    DOWN: (1, 0),
    LEFT: (0, -1),
    RIGHT: (0, 1),
}
ACTION_NAMES = {
    UP: "up",
    DOWN: "down",
    LEFT: "left",
    RIGHT: "right",
    SIPHON_LOW: "siphon-low",
    SIPHON_HIGH: "siphon-high",
}


class GridThiefEnv(_Base):
    """Level-1 Energy Thief grid world.

    Parameters
    ----------
    size : int
        Side length of the square grid. Default 5.
    surplus_max : int
        Cap on the (integer) unbanked surplus carried by the thief. The
        discrete state space has ``size * size * (surplus_max + 1)`` states.
    max_steps : int
        Episode length before truncation.
    n_rich_cells : int
        Number of tappable cells to place at random. Their values are drawn
        uniformly from ``{1, ..., max_cell_value}``; all other cells have
        value 0 (moving-only corridors).
    max_cell_value : int
        Largest per-cell value. ``siphon-low`` yields ``value``; ``siphon-high``
        yields ``2 * value`` (both capped so that surplus never exceeds
        ``surplus_max``).
    p_low, p_high : float
        Alarm probability per siphon at low / high intensity.
    alarm_penalty : float
        Reward subtracted when an alarm fires.
    step_cost : float
        Small reward subtracted on every step, to encourage efficient heists.
        Default 0.0 (kept simple for the first experiments).
    seed : int, optional
        Seed for the fixed value-map layout and the internal dynamics RNG.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        size: int = 5,
        surplus_max: int = 8,
        max_steps: int = 50,
        n_rich_cells: int = 5,
        max_cell_value: int = 2,
        p_low: float = 0.05,
        p_high: float = 0.20,
        alarm_penalty: float = 2.0,
        step_cost: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        if size < 2:
            raise ValueError(f"size must be >= 2, got {size}.")
        if surplus_max < 1:
            raise ValueError(f"surplus_max must be >= 1, got {surplus_max}.")
        if not 0.0 <= p_low <= p_high <= 1.0:
            raise ValueError(
                f"alarm probabilities must satisfy 0 <= p_low <= p_high <= 1, "
                f"got p_low={p_low}, p_high={p_high}."
            )

        self.size = int(size)
        self.surplus_max = int(surplus_max)
        self.max_steps = int(max_steps)
        self.max_cell_value = int(max_cell_value)
        self.p_low = float(p_low)
        self.p_high = float(p_high)
        self.alarm_penalty = float(alarm_penalty)
        self.step_cost = float(step_cost)

        # RNG for the (fixed) layout and for the (per-episode) dynamics.
        self._rng = np.random.default_rng(seed)

        # Fixed start / exit at opposite corners.
        self.start_pos: tuple[int, int] = (0, 0)
        self.exit_pos: tuple[int, int] = (self.size - 1, self.size - 1)

        # Value map is generated ONCE and stays fixed for the whole run, so the
        # tabular agent can memorise where the rich cells are.
        self.value_map = self._make_value_map(int(n_rich_cells))

        # Spaces: a single integer observation, six discrete actions.
        self.n_states = self.size * self.size * (self.surplus_max + 1)
        if spaces is not None:
            self.observation_space = spaces.Discrete(self.n_states)
            self.action_space = spaces.Discrete(6)

        # Episode state (initialised in reset()).
        self.pos: tuple[int, int] = self.start_pos
        self.surplus: int = 0
        self.t: int = 0

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _make_value_map(self, n_rich_cells: int) -> np.ndarray:
        """Place ``n_rich_cells`` valued cells at random; corners stay empty."""
        value_map = np.zeros((self.size, self.size), dtype=np.int64)
        forbidden = {self.start_pos, self.exit_pos}
        free = [
            (r, c)
            for r in range(self.size)
            for c in range(self.size)
            if (r, c) not in forbidden
        ]
        n = min(n_rich_cells, len(free))
        idx = self._rng.choice(len(free), size=n, replace=False)
        for k in idx:
            r, c = free[int(k)]
            value_map[r, c] = int(self._rng.integers(1, self.max_cell_value + 1))
        return value_map

    # ------------------------------------------------------------------
    # Observation encoding
    # ------------------------------------------------------------------
    def _encode(self, pos: tuple[int, int], surplus: int) -> int:
        """Map (position, surplus) to a single integer state index."""
        pos_idx = pos[0] * self.size + pos[1]
        return pos_idx * (self.surplus_max + 1) + int(surplus)

    def decode(self, obs: int) -> tuple[tuple[int, int], int]:
        """Inverse of :meth:`_encode` -- handy for plotting/debugging."""
        surplus = obs % (self.surplus_max + 1)
        pos_idx = obs // (self.surplus_max + 1)
        return (pos_idx // self.size, pos_idx % self.size), surplus

    def _get_obs(self) -> int:
        return self._encode(self.pos, self.surplus)

    def _info(self, alarm: bool = False, siphoned: int = 0) -> dict[str, Any]:
        return {
            "pos": self.pos,
            "surplus": self.surplus,
            "cell_value": int(self.value_map[self.pos]),
            "alarm": alarm,
            "siphoned": siphoned,
            "t": self.t,
        }

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[int, dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.pos = self.start_pos
        self.surplus = 0
        self.t = 0
        return self._get_obs(), self._info()

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        if self.action_space is not None and not self.action_space.contains(action):
            raise ValueError(f"invalid action {action!r}")

        reward = -self.step_cost
        terminated = False
        alarm = False
        siphoned = 0

        if action in _MOVES:
            dr, dc = _MOVES[action]
            nr, nc = self.pos[0] + dr, self.pos[1] + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                self.pos = (nr, nc)
            # Cash out on reaching the exit.
            if self.pos == self.exit_pos:
                reward += float(self.surplus)
                self.surplus = 0
                terminated = True
        else:  # a siphon action
            value = int(self.value_map[self.pos])
            if value > 0:  # nothing to steal (and no alarm risk) on empty cells
                intensity_high = action == SIPHON_HIGH
                gain = value * (2 if intensity_high else 1)
                p = self.p_high if intensity_high else self.p_low
                if self._rng.random() < p:
                    alarm = True
                    self.surplus = 0
                    reward -= self.alarm_penalty
                else:
                    siphoned = min(gain, self.surplus_max - self.surplus)
                    self.surplus += siphoned

        self.t += 1
        truncated = (not terminated) and (self.t >= self.max_steps)

        return self._get_obs(), reward, terminated, truncated, self._info(alarm, siphoned)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render(self) -> str:
        """Return an ASCII picture of the grid (``A``=thief, ``E``=exit)."""
        rows = []
        for r in range(self.size):
            cells = []
            for c in range(self.size):
                if (r, c) == self.pos:
                    cells.append("A")
                elif (r, c) == self.exit_pos:
                    cells.append("E")
                elif self.value_map[r, c] > 0:
                    cells.append(str(int(self.value_map[r, c])))
                else:
                    cells.append(".")
            rows.append(" ".join(cells))
        header = f"t={self.t}  surplus={self.surplus}/{self.surplus_max}"
        return header + "\n" + "\n".join(rows)
