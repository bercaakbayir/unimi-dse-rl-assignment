"""Energy Thief environment -- Level 1 (small power-grid network, tabular).

Level 1 of the VFA-4 "Energy Thief" project. The agent taps energy from the
transmission lines of a small power grid, building up an (unbanked) *surplus*
that it must *secure* before the grid's monitoring system raises an alarm and
wipes it.

Following the project brief, the grid is a small network of nodes (a plant, a
substation, consumers) joined by tappable transmission *edges*. The agent does
not move in space; at each step it decides *how* to operate:

* tap an edge at low or high intensity -- diverting energy into the surplus, at
  an alarm probability that grows with how aggressively it operates and with
  the current grid *load* (a busy grid is watched more closely);
* or *secure* part of the surplus, banking it safely as reward.

Level 1 is deliberately small so a **tabular** agent can solve it: the state is
just the discrete grid load and the discrete unbanked surplus, and the alarm
probability depends only on the current action and load -- there is no per-edge
suspicion history. That adaptive "heat" mechanic arrives in Level 2, where it
blows up the state space and motivates function approximation.

MDP summary
-----------
State   : (grid load L in {0..n_load-1}, unbanked surplus U in {0..surplus_max})
Actions : tap each edge at low / high intensity, or secure (bank) the surplus.
Reward  : +banked energy on a secure action;
          -alarm_penalty when an alarm fires (and U is reset to 0);
          0 otherwise.
Horizon : fixed; the episode truncates after ``max_steps`` steps.
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


# Default tappable edges as (name, base gain, base alarm risk). Edge A is a safe
# trickle; edge B is juicier but more closely watched.
DEFAULT_EDGES = (("A", 1, 0.03), ("B", 2, 0.08))


def _build_action_names(edges: tuple) -> list[str]:
    names: list[str] = []
    for name, _, _ in edges:
        names += [f"tap-{name}-low", f"tap-{name}-high"]
    names.append("secure")
    return names


# Module-level names for the default edge layout (index == action id).
ACTION_NAMES = _build_action_names(DEFAULT_EDGES)


class GridThiefEnv(_Base):
    """Level-1 Energy Thief power-grid network.

    Parameters
    ----------
    n_load : int
        Number of discrete grid-load phases. Load follows an exogenous random
        walk and scales the alarm probability (busier grid -> closer watch).
    surplus_max : int
        Cap on the (integer) unbanked surplus carried by the thief.
    bank_rate : int
        Maximum surplus that a single ``secure`` action banks as reward.
    max_steps : int
        Episode length before truncation (the task has no terminal state).
    edges : sequence of (name, gain, risk)
        Tappable transmission lines. ``gain`` is the energy diverted per low
        tap; ``risk`` is the base alarm probability per low tap.
    high_gain_mult, high_risk_mult : float
        A high-intensity tap multiplies the gain and the risk by these factors.
    load_risk : sequence of float, optional
        Alarm-probability multiplier per load phase (length ``n_load``).
    alarm_penalty : float
        Reward subtracted when an alarm fires.
    seed : int, optional
        Seed for the dynamics RNG (load walk and alarm draws).
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        n_load: int = 4,
        surplus_max: int = 8,
        bank_rate: int = 4,
        max_steps: int = 50,
        edges: tuple = DEFAULT_EDGES,
        high_gain_mult: int = 2,
        high_risk_mult: float = 2.5,
        load_risk: Optional[tuple] = None,
        alarm_penalty: float = 2.0,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        if n_load < 1:
            raise ValueError(f"n_load must be >= 1, got {n_load}.")
        if surplus_max < 1:
            raise ValueError(f"surplus_max must be >= 1, got {surplus_max}.")

        self.n_load = int(n_load)
        self.surplus_max = int(surplus_max)
        self.bank_rate = int(bank_rate)
        self.max_steps = int(max_steps)
        self.edges = tuple(edges)
        self.high_gain_mult = int(high_gain_mult)
        self.high_risk_mult = float(high_risk_mult)
        self.alarm_penalty = float(alarm_penalty)

        if load_risk is None:
            # Rises from calm to busy: stealing on a busy grid is far riskier.
            load_risk = tuple(0.5 + l for l in range(self.n_load))
        if len(load_risk) != self.n_load:
            raise ValueError("load_risk must have length n_load.")
        self.load_risk = tuple(float(x) for x in load_risk)

        self.action_names = _build_action_names(self.edges)
        self.n_taps = 2 * len(self.edges)
        self.SECURE = self.n_taps
        self.n_actions = self.n_taps + 1

        self.n_states = self.n_load * (self.surplus_max + 1)
        if spaces is not None:
            self.observation_space = spaces.Discrete(self.n_states)
            self.action_space = spaces.Discrete(self.n_actions)

        self._rng = np.random.default_rng(seed)

        # Episode state (initialised in reset()).
        self.load: int = 0
        self.surplus: int = 0
        self.t: int = 0

    # ------------------------------------------------------------------
    # Action helpers
    # ------------------------------------------------------------------
    def tap_action(self, edge_idx: int, high: bool = False) -> int:
        """Action id for tapping ``edge_idx`` at low/high intensity."""
        return edge_idx * 2 + (1 if high else 0)

    # ------------------------------------------------------------------
    # Observation encoding
    # ------------------------------------------------------------------
    def _encode(self, load: int, surplus: int) -> int:
        return load * (self.surplus_max + 1) + int(surplus)

    def decode(self, obs: int) -> tuple[int, int]:
        """Inverse of :meth:`_encode` -- returns (load, surplus)."""
        return obs // (self.surplus_max + 1), obs % (self.surplus_max + 1)

    def _get_obs(self) -> int:
        return self._encode(self.load, self.surplus)

    def _info(self, alarm: bool = False, stolen: int = 0, banked: int = 0) -> dict[str, Any]:
        return {
            "load": self.load,
            "surplus": self.surplus,
            "alarm": alarm,
            "stolen": stolen,
            "banked": banked,
            "t": self.t,
        }

    def _next_load(self) -> int:
        step = int(self._rng.choice((-1, 0, 1), p=(0.25, 0.5, 0.25)))
        return min(self.n_load - 1, max(0, self.load + step))

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
        self.load = 0
        self.surplus = 0
        self.t = 0
        return self._get_obs(), self._info()

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        if self.action_space is not None and not self.action_space.contains(action):
            raise ValueError(f"invalid action {action!r}")

        reward = 0.0
        alarm = False
        stolen = 0
        banked = 0

        if action == self.SECURE:
            banked = min(self.surplus, self.bank_rate)
            self.surplus -= banked
            reward = float(banked)
        else:  # a tap action
            edge_idx, high = divmod(action, 2)
            _, gain, base_risk = self.edges[edge_idx]
            # Alarm probability grows with intensity (aggressiveness) and load.
            p = base_risk * (self.high_risk_mult if high else 1.0) * self.load_risk[self.load]
            if self._rng.random() < min(1.0, p):
                alarm = True
                self.surplus = 0
                reward = -self.alarm_penalty
            else:
                stolen = gain * (self.high_gain_mult if high else 1)
                stolen = min(stolen, self.surplus_max - self.surplus)
                self.surplus += stolen

        # Grid load drifts on its own, independent of the agent.
        self.load = self._next_load()
        self.t += 1
        # Fixed-horizon task: no terminal state, only truncation on the clock.
        truncated = self.t >= self.max_steps

        return self._get_obs(), reward, False, truncated, self._info(alarm, stolen, banked)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render(self) -> str:
        bar = "#" * (self.load + 1) + "." * (self.n_load - self.load - 1)
        return f"t={self.t:<3} load=[{bar}] surplus={self.surplus}/{self.surplus_max}"
