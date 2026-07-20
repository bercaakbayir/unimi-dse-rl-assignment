"""Energy Thief environment -- Level 2 (medium network, flows + adaptive suspicion).

Level 2 keeps Level 1's real power-grid model -- a plant feeding **consumers**
through transmission edges that carry **energy flows**, with the overproduction on
each edge available as divertible **slack** -- but on a bigger network: the plant
feeds **two substations**, each serving its own group of consumers, and one
substation is watched more closely than the other (a per-substation monitoring
factor). It also adds the brief's **adaptive monitoring**: each consumer edge now
carries a **suspicion** ("heat") that **rises each time it is tapped** (skim or
overdraw) and **cools when left alone**. Suspicion inflates that edge's alarm
probability, so repeatedly milking one line becomes self-defeating -- the thief
must **diversify** across the grid.

Reward is as in Level 1: the thief is rewarded for **the energy it steals each
step**, so the return is the **total energy stolen**. A **triggered alarm** raises
no haul that step, **locks the thief out** for ``k_lock`` steps, and **spikes the
monitoring**: the tapped edge's suspicion jumps to its maximum and every edge on the
same substation heats up. A **lie-low** action operates nothing.

The suspicion of every edge enters the state, so the discrete state space explodes
as ``k ** n_consumers``. Tabular Q-learning can still be *run*, but most states are
visited too rarely to learn: that curse of dimensionality is what motivates linear
function approximation with hand-crafted features.

MDP summary
-----------
State   : (demand phase P, suspicion sigma_c for each consumer edge, lock-out steps
           remaining). The cumulative haul is a scoreboard, not part of the state.
Actions : skim / overdraw each consumer's edge, or lie low.
Reward  : the energy stolen that step (0 on lie-low, a no-op, or while locked out).
Alarm   : no haul that step; locks stealing out for ``k_lock`` steps; suspicion spike.
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


DEFAULT_CONSUMERS = ("C1", "C2", "C3", "C4", "C5", "C6")
ACTION_NAMES = _build_action_names(DEFAULT_CONSUMERS)


class GridThiefEnvL2(_Base):
    """Level-2 Energy Thief network: flows + slack + two substations + suspicion.

    Adds to Level 1's parameters:

    n_substations : int
        Number of substations; consumers are split into contiguous groups.
    substation_sens : sequence of float, optional
        Per-substation monitoring multiplier (length n_substations); default rises
        by 0.3 per substation, so later substations are watched more closely.
    susp_levels : int
        Number of discrete suspicion levels per edge (``k``).
    susp_factor : float
        Each suspicion level multiplies an edge's alarm risk by ``1 + susp_factor``.
    cool_prob : float
        Per-step probability that an untapped edge's suspicion cools by one level.
    k_lock : int
        Number of steps the thief is locked out of stealing after an alarm.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        n_consumers: int = 6,
        n_substations: int = 2,
        n_phase: int = 4,
        max_steps: int = 50,
        base_waste: int = 4,
        base_sens: float = 0.08,
        base_divert: float = 0.5,
        shortfall_weight: float = 1.5,
        overdraw_extra: int = 2,
        susp_levels: int = 3,
        susp_factor: float = 1.0,
        cool_prob: float = 0.3,
        k_lock: int = 3,
        substation_sens: Optional[tuple] = None,
        surplus_max: int = 8,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        if n_consumers < 1 or n_phase < 1 or susp_levels < 1 or k_lock < 1:
            raise ValueError("counts must be >= 1.")
        if not 1 <= n_substations <= n_consumers:
            raise ValueError("need 1 <= n_substations <= n_consumers.")

        self.n_consumers = int(n_consumers)
        self.n_substations = int(n_substations)
        self.n_phase = int(n_phase)
        self.max_steps = int(max_steps)
        self.base_waste = int(base_waste)
        self.base_sens = float(base_sens)
        self.base_divert = float(base_divert)
        self.shortfall_weight = float(shortfall_weight)
        self.overdraw_extra = int(overdraw_extra)
        self.k = int(susp_levels)
        self.susp_factor = float(susp_factor)
        self.cool_prob = float(cool_prob)
        self.k_lock = int(k_lock)
        self.surplus_max = int(surplus_max)  # display-only gauge normaliser

        self.consumers = tuple(f"C{i+1}" for i in range(self.n_consumers))
        self.action_names = _build_action_names(self.consumers)

        # Two substations feed the consumers as contiguous groups; each has its own
        # monitoring sensitivity, so which substation a consumer sits under matters.
        self.substations = tuple(f"S{s+1}" for s in range(self.n_substations))
        self.substation_of = np.zeros(self.n_consumers, dtype=np.int64)
        base, rem, idx = divmod(self.n_consumers, self.n_substations) + (0,)
        for s in range(self.n_substations):
            for _ in range(base + (1 if s < rem else 0)):
                self.substation_of[idx] = s; idx += 1
        if substation_sens is None:
            substation_sens = tuple(1.0 + 0.3 * s for s in range(self.n_substations))
        if len(substation_sens) != self.n_substations:
            raise ValueError("substation_sens must have length n_substations.")
        self.substation_sens = tuple(float(x) for x in substation_sens)

        self.n_taps = 2 * self.n_consumers
        self.LIE_LOW = self.n_taps
        self.n_actions = self.n_taps + 1

        self._build_grid()

        # State = (phase, per-edge suspicion, lock-out steps remaining). The haul is
        # a scoreboard, not part of the state -- suspicion is what explodes |S|.
        self.n_states = (
            self.n_phase
            * (self.k ** self.n_consumers)
            * (self.k_lock + 1)
        )
        if spaces is not None:
            self.observation_space = spaces.Discrete(self.n_states)
            self.action_space = spaces.Discrete(self.n_actions)

        self._rng = np.random.default_rng(seed)

        self.phase: int = 0
        self.susp = np.zeros(self.n_consumers, dtype=np.int64)
        self.lock_remaining: int = 0
        self.surplus: int = 0   # cumulative haul (scoreboard only, uncapped)
        self.t: int = 0

    def _build_grid(self) -> None:
        n = self.n_consumers
        self.demand = np.zeros((self.n_phase, n), dtype=np.int64)
        self.slack = np.zeros((self.n_phase, n), dtype=np.int64)
        self.flow = np.zeros((self.n_phase, n), dtype=np.int64)
        self.sens = np.zeros(self.n_phase)
        for ph in range(self.n_phase):
            demands = np.array([1 + ((i + ph) % 3) for i in range(n)])
            waste = max(0, self.base_waste - ph)
            w = (demands.max() - demands + 1).astype(float)
            alloc = np.zeros(n, dtype=np.int64)
            for _ in range(waste):
                j = int(np.argmax(w)); alloc[j] += 1; w[j] -= 1.0
            self.demand[ph] = demands
            self.slack[ph] = alloc
            self.flow[ph] = demands + alloc
            self.sens[ph] = self.base_sens * (ph + 1)

    def tap_action(self, consumer_idx: int, overdraw: bool = False) -> int:
        return consumer_idx * 2 + (1 if overdraw else 0)

    def _encode(self) -> int:
        idx = self.phase
        for s in self.susp:
            idx = idx * self.k + int(s)
        idx = idx * (self.k_lock + 1) + self.lock_remaining
        return idx

    def decode(self, obs: int) -> tuple[int, np.ndarray, int]:
        """Inverse of :meth:`_encode` -- returns (phase, suspicion vec, lock)."""
        lock = obs % (self.k_lock + 1)
        obs //= (self.k_lock + 1)
        susp = np.zeros(self.n_consumers, dtype=np.int64)
        for i in range(self.n_consumers - 1, -1, -1):
            obs, susp[i] = divmod(obs, self.k)
        phase = obs
        return phase, susp, lock

    def _get_obs(self) -> int:
        return self._encode()

    def _info(self, alarm=False, stolen=0, tapped=-1, shortfall=0) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "surplus": self.surplus,          # cumulative haul (scoreboard)
            "suspicion": self.susp.copy(),
            "slack": self.slack[self.phase].copy(),
            "alarm": alarm,
            "stolen": stolen,                 # energy stolen this step (the reward)
            "tapped": tapped,
            "shortfall": shortfall,
            "locked": self.lock_remaining > 0,
            "lock_remaining": self.lock_remaining,
            "t": self.t,
        }

    def _next_phase(self) -> int:
        step = int(self._rng.choice((-1, 0, 1), p=(0.25, 0.5, 0.25)))
        return min(self.n_phase - 1, max(0, self.phase + step))

    def _cool(self, tapped: int) -> None:
        for i in range(self.n_consumers):
            if i != tapped and self.susp[i] > 0 and self._rng.random() < self.cool_prob:
                self.susp[i] -= 1

    def reset(self, *, seed: Optional[int] = None, options=None) -> tuple[int, dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.phase = 0
        self.susp = np.zeros(self.n_consumers, dtype=np.int64)
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
            slack = int(self.slack[self.phase, c])
            flow = int(self.flow[self.phase, c])
            requested = min(slack + self.overdraw_extra, flow) if overdraw else slack
            take = requested
            if take > 0:
                shortfall = max(0, take - slack)
                # Level-1 alarm model, inflated by the substation factor and this
                # edge's suspicion.
                p = (self.sens[self.phase]
                     * self.substation_sens[self.substation_of[c]]
                     * (self.base_divert + self.shortfall_weight * shortfall)
                     * (1.0 + self.susp_factor * self.susp[c]))
                self.susp[c] = min(self.susp[c] + 1, self.k - 1)   # heat rises on a tap
                if self._rng.random() < min(1.0, p):
                    alarm = True                                   # caught: no haul, lock-out follows
                    # Monitoring spike: the caught edge maxes out and the whole
                    # substation is put on alert.
                    self.susp[c] = self.k - 1
                    sub = self.substation_of[c]
                    for j in range(self.n_consumers):
                        if self.substation_of[j] == sub:
                            self.susp[j] = min(self.susp[j] + 1, self.k - 1)
                else:
                    stolen = take
                    self.surplus += stolen

        # Reward = energy stolen this step; the return is the total energy stolen.
        reward = float(stolen)

        # Lock-out bookkeeping: an alarm arms the lock-out; otherwise it counts down.
        if alarm:
            self.lock_remaining = self.k_lock
        elif self.lock_remaining > 0:
            self.lock_remaining -= 1

        self._cool(tapped)
        self.phase = self._next_phase()
        self.t += 1
        truncated = self.t >= self.max_steps
        return self._get_obs(), reward, False, truncated, self._info(alarm, stolen, tapped, shortfall)

    def render(self) -> str:
        heat = " ".join(f"{self.consumers[i]}{self.susp[i]}" for i in range(self.n_consumers))
        groups = "  ".join(f"{self.substations[s]}:" +
                           "".join(self.consumers[i] for i in range(self.n_consumers)
                                   if self.substation_of[i] == s)
                           for s in range(self.n_substations))
        lock = f" LOCKED({self.lock_remaining})" if self.lock_remaining > 0 else ""
        return (f"t={self.t:<3} phase={self.phase} haul={self.surplus}"
                f"  heat[{heat}]  [{groups}]{lock}")
