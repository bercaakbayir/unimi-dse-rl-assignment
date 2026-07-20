"""Energy Thief environment -- Level 1 (small power-grid network with flows).

Level 1 of the VFA-4 "Energy Thief" project. The grid is a real network of nodes
-- a **plant** (source), a **substation**, and several **consumers** (sinks) --
joined by transmission edges that carry **energy flows** from the plant out to the
consumers. Because generation and demand never match exactly, some edges carry
more than their consumer needs: that excess is **slack** (the grid's inefficiency).

The thief steals by **redirecting flow** off an edge, adding the diverted energy to
a running **haul**:

* **skim** an edge -- divert only its slack (the waste). No consumer goes short,
  so detection risk is low;
* **overdraw** an edge -- divert the slack *and* dip into the delivered demand,
  starving the consumer. Bigger haul, but the shortfall is conspicuous, so the
  monitoring system is far likelier to raise an **alarm**;
* **lie low** -- operate nothing this step, drawing no attention.

The thief is rewarded for **the energy it steals each step**; the return over an
episode is therefore the **total energy stolen**. A **triggered alarm** raises no
haul that step and **locks the thief out** for a few steps (taps become no-ops
while the grid is on alert) -- getting caught costs *future* stealing opportunities,
not the haul already banked. The grid cycles through discrete **demand phases** (an
exogenous random walk); each phase fixes the monitoring sensitivity and the
distribution of slack on every edge, and the **slack fluctuates each step**, so
which edge is worth skimming -- and how tight the grid is -- changes over time.

Level 1 is small enough for a **tabular** agent: the state is the grid condition the
thief reads to act -- (demand phase, the currently revealed per-edge slack levels,
and how many lock-out steps remain).

MDP summary
-----------
State   : (demand phase P in {0..n_phase-1},
           revealed slack level of each edge in {0..slack_levels-1},
           lock-out steps remaining in {0..k_lock})
Actions : skim / overdraw each consumer's edge, or lie low.
Reward  : the energy stolen that step (0 on lie-low, a no-op, or while locked out).
          The return is the total energy stolen over the episode.
Alarm   : raises no haul that step and locks stealing out for ``k_lock`` steps.
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
    """Level-1 Energy Thief power-grid network with flows and (stochastic) slack.

    Parameters
    ----------
    n_consumers : int
        Number of consumer nodes (and tappable substation->consumer edges).
    n_phase : int
        Number of discrete demand phases (exogenous random walk). Higher phase =
        tighter grid (less slack, closer monitoring).
    slack_levels : int
        Number of discrete slack levels an edge can reveal each step (0..levels-1).
    max_steps : int
        Episode length.
    base_sens : float
        Monitoring sensitivity in phase 0; it scales up with the phase.
    base_divert : float
        Detectability of any diversion (the constant term of the alarm model).
    shortfall_weight : float
        How strongly a delivered-demand shortfall raises the alarm probability.
    overdraw_extra : int
        Extra energy an ``overdraw`` takes beyond the slack (into delivered demand).
    k_lock : int
        Number of steps the thief is locked out of stealing after an alarm.
    surplus_max : int
        Display-only normaliser for the cumulative-haul gauge (the haul is uncapped).
    seed : int, optional
        Seed for the dynamics RNG (phase walk, slack draws and alarm draws).
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        n_consumers: int = 3,
        n_phase: int = 4,
        slack_levels: int = 3,
        max_steps: int = 50,
        base_sens: float = 0.10,
        base_divert: float = 0.5,
        shortfall_weight: float = 1.5,
        overdraw_extra: int = 2,
        k_lock: int = 2,
        surplus_max: int = 8,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        if n_consumers < 1 or n_phase < 1 or slack_levels < 2 or k_lock < 1:
            raise ValueError("n_consumers, n_phase >= 1; slack_levels >= 2; k_lock >= 1.")

        self.n_consumers = int(n_consumers)
        self.n_phase = int(n_phase)
        self.slack_levels = int(slack_levels)
        self.max_steps = int(max_steps)
        self.base_sens = float(base_sens)
        self.base_divert = float(base_divert)
        self.shortfall_weight = float(shortfall_weight)
        self.overdraw_extra = int(overdraw_extra)
        self.k_lock = int(k_lock)
        self.surplus_max = int(surplus_max)  # display-only gauge normaliser

        self.consumers = tuple(f"C{i+1}" for i in range(self.n_consumers))
        self.action_names = _build_action_names(self.consumers)
        self.n_taps = 2 * self.n_consumers
        self.LIE_LOW = self.n_taps
        self.n_actions = self.n_taps + 1

        self._build_grid()

        # State = (phase, per-edge slack level, lock-out steps remaining).
        self.n_states = (
            self.n_phase
            * (self.slack_levels ** self.n_consumers)
            * (self.k_lock + 1)
        )
        if spaces is not None:
            self.observation_space = spaces.Discrete(self.n_states)
            self.action_space = spaces.Discrete(self.n_actions)

        self._rng = np.random.default_rng(seed)

        self.phase: int = 0
        self.cur_slack: np.ndarray = np.zeros(self.n_consumers, dtype=np.int64)
        self.lock_remaining: int = 0
        self.surplus: int = 0   # cumulative haul (scoreboard only, uncapped)
        self.t: int = 0

    # ------------------------------------------------------------------
    # Grid layout: demand and monitoring sensitivity per phase (deterministic);
    # slack is drawn stochastically each step (see ``_draw_slack``).
    # ------------------------------------------------------------------
    def _build_grid(self) -> None:
        n = self.n_consumers
        self.demand = np.zeros((self.n_phase, n), dtype=np.int64)
        self.sens = np.zeros(self.n_phase)
        for ph in range(self.n_phase):
            self.demand[ph] = np.array([1 + ((i + ph) % 3) for i in range(n)])
            self.sens[ph] = self.base_sens * (ph + 1)  # monitoring tightens with phase

    def _slack_probs(self, phase: int) -> np.ndarray:
        """Categorical over slack levels {0,1,2} for a phase (loose->tight as phase rises)."""
        tau = phase / max(1, self.n_phase - 1)          # 0 (loose) .. 1 (tight)
        p_hi = 0.5 * (1.0 - tau)                         # lots of slack when loose
        p_lo = 0.2 + 0.5 * tau                           # little slack when tight
        p_mid = 1.0 - p_lo - p_hi
        return np.array([p_lo, p_mid, p_hi])

    def _draw_slack(self, phase: int) -> np.ndarray:
        """Reveal each edge's slack level for the given phase (i.i.d. per edge)."""
        p = self._slack_probs(phase)
        levels = np.arange(self.slack_levels)
        return self._rng.choice(levels, size=self.n_consumers, p=p).astype(np.int64)

    # ------------------------------------------------------------------
    # Action / observation helpers
    # ------------------------------------------------------------------
    def tap_action(self, consumer_idx: int, overdraw: bool = False) -> int:
        return consumer_idx * 2 + (1 if overdraw else 0)

    def _encode(self) -> int:
        idx = self.phase
        for c in range(self.n_consumers):
            idx = idx * self.slack_levels + int(self.cur_slack[c])
        idx = idx * (self.k_lock + 1) + self.lock_remaining
        return idx

    def decode(self, obs: int) -> tuple[int, np.ndarray, int]:
        """Inverse of :meth:`_encode` -- returns (demand phase, slack levels, lock)."""
        lock = obs % (self.k_lock + 1)
        obs //= (self.k_lock + 1)
        slack = np.zeros(self.n_consumers, dtype=np.int64)
        for c in range(self.n_consumers - 1, -1, -1):
            slack[c] = obs % self.slack_levels
            obs //= self.slack_levels
        phase = obs
        return phase, slack, lock

    def _get_obs(self) -> int:
        return self._encode()

    def _info(self, alarm=False, stolen=0, tapped=-1, shortfall=0) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "surplus": self.surplus,       # cumulative haul (scoreboard)
            "slack": self.cur_slack.copy(),  # currently revealed per-edge slack
            "alarm": alarm,
            "stolen": stolen,              # energy stolen this step (the reward)
            "tapped": tapped,              # consumer edge tapped this step, or -1
            "shortfall": shortfall,        # delivered-demand shortfall caused this step
            "locked": self.lock_remaining > 0,
            "lock_remaining": self.lock_remaining,
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
        self.cur_slack = self._draw_slack(self.phase)
        self.lock_remaining = 0
        self.surplus = 0
        self.t = 0
        return self._get_obs(), self._info()

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        if self.action_space is not None and not self.action_space.contains(action):
            raise ValueError(f"invalid action {action!r}")

        alarm = False
        stolen = 0
        tapped = -1
        shortfall = 0
        locked = self.lock_remaining > 0

        if (not locked) and action != self.LIE_LOW:
            c, overdraw = divmod(action, 2)
            tapped = c
            slack = int(self.cur_slack[c])
            flow = int(self.demand[self.phase, c]) + slack
            requested = min(slack + self.overdraw_extra, flow) if overdraw else slack
            take = requested
            if take > 0:
                shortfall = max(0, take - slack)     # dips into delivered demand
                # Alarm probability: a base for any diversion plus the shortfall it
                # causes, scaled by the phase's monitoring sensitivity.
                p = self.sens[self.phase] * (self.base_divert + self.shortfall_weight * shortfall)
                if self._rng.random() < min(1.0, p):
                    alarm = True                 # caught: no haul this step, lock-out follows
                else:
                    stolen = take
                    self.surplus += stolen       # haul is kept -- never confiscated

        # Reward = energy stolen this step. Summed over the episode this is the
        # total energy stolen -- the quantity the thief maximises.
        reward = float(stolen)

        # Lock-out bookkeeping: an alarm arms the lock-out; otherwise it counts down.
        if alarm:
            self.lock_remaining = self.k_lock
        elif self.lock_remaining > 0:
            self.lock_remaining -= 1

        self.phase = self._next_phase()
        self.cur_slack = self._draw_slack(self.phase)
        self.t += 1
        truncated = self.t >= self.max_steps
        return self._get_obs(), reward, False, truncated, self._info(alarm, stolen, tapped, shortfall)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render(self) -> str:
        flows = " ".join(f"{self.consumers[i]}:+{self.cur_slack[i]}"
                         for i in range(self.n_consumers))
        lock = f" LOCKED({self.lock_remaining})" if self.lock_remaining > 0 else ""
        return (f"t={self.t:<3} phase={self.phase} (sens={self.sens[self.phase]:.2f})"
                f"  haul={self.surplus}  slack[{flows}]{lock}")
