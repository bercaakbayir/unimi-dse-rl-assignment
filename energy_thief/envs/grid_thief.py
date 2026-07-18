"""Energy Thief environment -- Level 1 (small power-grid network with flows).

Level 1 of the VFA-4 "Energy Thief" project. The grid is a real network of nodes
-- a **plant** (source), a **substation**, and several **consumers** (sinks) --
joined by transmission edges that carry **energy flows** from the plant out to the
consumers. Because generation and demand never match exactly, some edges carry
more than their consumer needs: that excess is **slack** (the grid's inefficiency).

The thief steals by **redirecting flow** off an edge, accumulating an **unbanked
surplus** of diverted energy:

* **skim** an edge -- divert only its slack (the waste). No consumer goes short,
  so detection risk is low;
* **overdraw** an edge -- divert the slack *and* dip into the delivered demand,
  starving the consumer. Bigger haul, but the shortfall is conspicuous, so the
  monitoring system is far likelier to raise an **alarm**;
* **lie low** -- operate nothing this step, drawing no attention (protects the
  accumulated surplus).

The thief is rewarded for the surplus it holds each step (it benefits from the
diverted energy while it keeps it); a **triggered alarm resets the accumulated
surplus** to zero, ending that benefit. The grid cycles through
discrete **demand phases** (an exogenous random walk); each phase fixes the flow
and slack on every edge and the monitoring sensitivity, so which edge is worth
skimming -- and how tight the grid is -- changes over time.

Level 1 is small enough for a **tabular** agent: the slack on every edge is a
deterministic function of the current demand phase, so the state is just
(demand phase, surplus).

MDP summary
-----------
State   : (demand phase P in {0..n_phase-1}, unbanked surplus U in {0..surplus_max})
Actions : skim / overdraw each consumer's edge, or lie low.
Reward  : the surplus held that step; an alarm resets the surplus to 0 (so the
          step after an alarm earns nothing until it is rebuilt).
Horizon : fixed; the episode truncates after ``max_steps`` steps.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    _Base = gym.Env
except ModuleNotFoundError:  # pragma: no cover
    gym = None
    spaces = None
    _Base = object


def _build_action_names(consumers: tuple) -> list[str]:
    names: list[str] = []
    for c in consumers:
        names += [f"skim-{c}", f"overdraw-{c}"]
    names.append("lie-low")
    return names


DEFAULT_CONSUMERS = ("C1", "C2", "C3")
ACTION_NAMES = _build_action_names(DEFAULT_CONSUMERS)


class GridThiefEnv(_Base):
    """Level-1 Energy Thief power-grid network with flows and slack.

    Parameters
    ----------
    n_consumers : int
        Number of consumer nodes (and tappable substation->consumer edges).
    n_phase : int
        Number of discrete demand phases (exogenous random walk). Higher phase =
        tighter grid (less slack, closer monitoring).
    surplus_max, max_steps : int
        Surplus cap and episode length.
    base_waste : int
        Total slack (overproduction) injected in phase 0; it falls by one per phase.
    base_sens : float
        Monitoring sensitivity in phase 0; it scales up with the phase.
    base_divert : float
        Detectability of any diversion (the constant term of the alarm model).
    shortfall_weight : float
        How strongly a delivered-demand shortfall raises the alarm probability.
    overdraw_extra : int
        Extra energy an ``overdraw`` takes beyond the slack (into delivered demand).
    seed : int, optional
        Seed for the dynamics RNG (phase walk and alarm draws).
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        n_consumers: int = 3,
        n_phase: int = 4,
        surplus_max: int = 8,
        max_steps: int = 50,
        base_waste: int = 3,
        base_sens: float = 0.10,
        base_divert: float = 1.0,
        shortfall_weight: float = 1.5,
        overdraw_extra: int = 2,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        if n_consumers < 1 or n_phase < 1 or surplus_max < 1:
            raise ValueError("n_consumers, n_phase, surplus_max must be >= 1.")

        self.n_consumers = int(n_consumers)
        self.n_phase = int(n_phase)
        self.surplus_max = int(surplus_max)
        self.max_steps = int(max_steps)
        self.base_waste = int(base_waste)
        self.base_sens = float(base_sens)
        self.base_divert = float(base_divert)
        self.shortfall_weight = float(shortfall_weight)
        self.overdraw_extra = int(overdraw_extra)

        self.consumers = tuple(f"C{i+1}" for i in range(self.n_consumers))
        self.action_names = _build_action_names(self.consumers)
        self.n_taps = 2 * self.n_consumers
        self.LIE_LOW = self.n_taps
        self.n_actions = self.n_taps + 1

        self._build_grid()

        self.n_states = self.n_phase * (self.surplus_max + 1)
        if spaces is not None:
            self.observation_space = spaces.Discrete(self.n_states)
            self.action_space = spaces.Discrete(self.n_actions)

        self._rng = np.random.default_rng(seed)

        self.phase: int = 0
        self.surplus: int = 0
        self.t: int = 0

    # ------------------------------------------------------------------
    # Grid layout: flows and slack per phase (deterministic)
    # ------------------------------------------------------------------
    def _build_grid(self) -> None:
        n = self.n_consumers
        self.demand = np.zeros((self.n_phase, n), dtype=np.int64)
        self.slack = np.zeros((self.n_phase, n), dtype=np.int64)
        self.flow = np.zeros((self.n_phase, n), dtype=np.int64)
        self.sens = np.zeros(self.n_phase)
        for ph in range(self.n_phase):
            demands = np.array([1 + ((i + ph) % 3) for i in range(n)])
            waste = max(0, self.base_waste - ph)          # tighter grid at high phase
            w = (demands.max() - demands + 1).astype(float)
            alloc = np.zeros(n, dtype=np.int64)
            for _ in range(waste):
                j = int(np.argmax(w)); alloc[j] += 1; w[j] -= 1.0
            self.demand[ph] = demands
            self.slack[ph] = alloc
            self.flow[ph] = demands + alloc
            self.sens[ph] = self.base_sens * (ph + 1)

    # ------------------------------------------------------------------
    # Action / observation helpers
    # ------------------------------------------------------------------
    def tap_action(self, consumer_idx: int, overdraw: bool = False) -> int:
        return consumer_idx * 2 + (1 if overdraw else 0)

    def _encode(self) -> int:
        return self.phase * (self.surplus_max + 1) + self.surplus

    def decode(self, obs: int) -> tuple[int, int]:
        """Inverse of :meth:`_encode` -- returns (demand phase, surplus)."""
        return obs // (self.surplus_max + 1), obs % (self.surplus_max + 1)

    def _get_obs(self) -> int:
        return self._encode()

    def _info(self, alarm=False, stolen=0, tapped=-1, shortfall=0) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "surplus": self.surplus,
            "alarm": alarm,
            "stolen": stolen,
            "tapped": tapped,        # consumer edge tapped this step, or -1
            "shortfall": shortfall,  # delivered-demand shortfall caused this step
            "t": self.t,
        }

    def _next_phase(self) -> int:
        step = int(self._rng.choice((-1, 0, 1), p=(0.25, 0.5, 0.25)))
        return min(self.n_phase - 1, max(0, self.phase + step))

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options=None) -> tuple[int, dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.phase = 0
        self.surplus = 0
        self.t = 0
        return self._get_obs(), self._info()

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        if self.action_space is not None and not self.action_space.contains(action):
            raise ValueError(f"invalid action {action!r}")

        reward = 0.0
        alarm = False
        stolen = 0
        tapped = -1
        shortfall = 0

        if action != self.LIE_LOW:
            c, overdraw = divmod(action, 2)
            tapped = c
            slack = int(self.slack[self.phase, c])
            flow = int(self.flow[self.phase, c])
            requested = min(slack + self.overdraw_extra, flow) if overdraw else slack
            take = min(requested, self.surplus_max - self.surplus)  # only what fits
            if take > 0:
                shortfall = max(0, take - slack)     # dips into delivered demand
                # Alarm probability: a base for any diversion plus the shortfall it
                # causes, scaled by the phase's monitoring sensitivity.
                p = self.sens[self.phase] * (self.base_divert + self.shortfall_weight * shortfall)
                if self._rng.random() < min(1.0, p):
                    alarm = True
                    self.surplus = 0                # the whole run is forfeited
                else:
                    stolen = take
                    self.surplus += stolen

        # Reward = surplus held this step: the thief benefits from the diverted
        # energy every step it keeps it, so accumulating a large surplus and not
        # losing it to an alarm is what pays off.
        reward = float(self.surplus)
        self.phase = self._next_phase()
        self.t += 1
        truncated = self.t >= self.max_steps
        return self._get_obs(), reward, False, truncated, self._info(alarm, stolen, tapped, shortfall)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render(self) -> str:
        flows = " ".join(f"{self.consumers[i]}:{self.flow[self.phase, i]}"
                         f"(+{self.slack[self.phase, i]})" for i in range(self.n_consumers))
        return (f"t={self.t:<3} phase={self.phase} (sens={self.sens[self.phase]:.2f})"
                f"  surplus={self.surplus}/{self.surplus_max}  flows[{flows}]")
