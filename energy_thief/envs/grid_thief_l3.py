"""Energy Thief environment -- Level 3 (large network, continuous flows, POMDP).

Level 3 is the hardest of the three complexity levels. It keeps the flow model of
Levels 1-2 (a plant feeding consumers through substations; the thief skims the slack
off a line or overdraws into delivered demand) but turns up three dials that finally
defeat tabular and linear methods:

1. **Continuous flows.** Consumer demands vary as continuous stochastic processes
   over the shift, so each line's **slack is a real number**, not a small integer.
   The state is continuous -- a Q-table cannot enumerate it.

2. **Large network.** Many consumers across several substations -> a high-dimensional
   continuous observation.

3. **Partial observability (the brief's challenging variant).** Every line still
   carries an adaptive **suspicion** (heat that rises when tapped and decays), and it
   still drives the alarm -- but the agent **cannot observe it**. It sees only
   *aggregate* grid statistics (total load, a global suspicion index, the recent alarm
   rate) that carry indirect information. To know whether a specific line is hot, the
   agent must infer the hidden state from its own history: a POMDP.

Because the value is a nonlinear function of an unobservable state, hand-crafted linear
features saturate and only a **DQN** (a neural network over stacked observations, with
replay + target network) can handle it.

As in Levels 1-2, the thief is rewarded for **the energy it steals each step** (the
return is the total energy stolen), and a **triggered alarm** raises no haul, **locks
the thief out** for ``k_lock`` steps, and spikes the (hidden) suspicion on the tapped
line. The lock-out counter *is* observable; the per-line suspicion is not.

MDP summary
-----------
State   : continuous. Observed: per-line slack, time, lock-out counter, and AGGREGATE
          stats (total load, mean suspicion, recent alarm rate). Hidden: per-line
          suspicion. The cumulative haul is a scoreboard, not part of the state.
Actions : skim / overdraw each consumer's edge, or lie low.
Reward  : the energy stolen that step (0 on lie-low, a no-op, or while locked out).
Alarm   : no haul that step; locks stealing out for ``k_lock`` steps; suspicion spike.
Horizon : fixed; the episode truncates after ``max_steps`` steps.
"""

from __future__ import annotations

from collections import deque
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


class GridThiefEnvL3(_Base):
    """Level-3 Energy Thief: continuous flows, large network, hidden suspicion.

    The observation is a continuous vector; the per-line suspicion that drives the
    alarm is part of the true state but is NOT observed -- only aggregate statistics
    are. Actions are discrete (skim/overdraw each line, or lie-low), so the
    environment plugs into a DQN.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        n_consumers: int = 9,
        n_substations: int = 3,
        max_steps: int = 50,
        surplus_cap: float = 10.0,
        base_sens: float = 0.05,
        base_divert: float = 0.5,
        shortfall_weight: float = 1.5,
        overdraw_frac: float = 0.5,
        susp_up: float = 0.30,
        susp_decay: float = 0.85,
        susp_factor: float = 1.0,
        alarm_window: int = 8,
        k_lock: int = 3,
        substation_sens: Optional[tuple] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        if not 1 <= n_substations <= n_consumers:
            raise ValueError("need 1 <= n_substations <= n_consumers.")

        self.n_consumers = int(n_consumers)
        self.n_substations = int(n_substations)
        self.max_steps = int(max_steps)
        self.surplus_cap = float(surplus_cap)   # display-only gauge normaliser
        self.base_sens = float(base_sens)
        self.base_divert = float(base_divert)
        self.shortfall_weight = float(shortfall_weight)
        self.overdraw_frac = float(overdraw_frac)
        self.susp_up = float(susp_up)
        self.susp_decay = float(susp_decay)
        self.susp_factor = float(susp_factor)
        self.alarm_window = int(alarm_window)
        self.k_lock = int(k_lock)

        self.consumers = tuple(f"C{i+1}" for i in range(self.n_consumers))
        self.action_names = _build_action_names(self.consumers)
        self.n_taps = 2 * self.n_consumers
        self.LIE_LOW = self.n_taps
        self.n_actions = self.n_taps + 1

        # Consumers split into contiguous substation groups; later substations
        # are watched more closely.
        self.substations = tuple(f"S{s+1}" for s in range(self.n_substations))
        self.substation_of = np.zeros(self.n_consumers, dtype=np.int64)
        base, rem, idx = divmod(self.n_consumers, self.n_substations) + (0,)
        for s in range(self.n_substations):
            for _ in range(base + (1 if s < rem else 0)):
                self.substation_of[idx] = s; idx += 1
        if substation_sens is None:
            substation_sens = tuple(1.0 + 0.3 * s for s in range(self.n_substations))
        self.substation_sens = np.array([float(x) for x in substation_sens])

        # Fixed demand structure; per-episode phase offsets add variety.
        self.base_demand = np.array([1.0 + (i % 3) for i in range(self.n_consumers)])
        self.amp = 0.6 * self.base_demand
        self.period = 9.0 + (np.arange(self.n_consumers) % 5)  # 9..13
        self.margin_period = 17.0
        self.load_ref = float(self.base_demand.sum())

        self._rng = np.random.default_rng(seed)

        # Observation: per-line slack + time + lock-out + [load, mean susp, alarm rate].
        self.obs_dim = self.n_consumers + 1 + 1 + 3
        if spaces is not None:
            self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(self.obs_dim,), dtype=np.float32)
            self.action_space = spaces.Discrete(self.n_actions)

        self.surplus = 0.0            # cumulative haul (scoreboard only, uncapped)
        self.susp = np.zeros(self.n_consumers)
        self.lock_remaining = 0
        self.t = 0
        self._phase_off = np.zeros(self.n_consumers)
        self._alarm_hist: deque = deque(maxlen=self.alarm_window)

    # ------------------------------------------------------------------
    # Continuous grid signals (deterministic in t, per-episode phase offset)
    # ------------------------------------------------------------------
    def _demand(self, t: int) -> np.ndarray:
        d = self.base_demand + self.amp * np.sin(2 * np.pi * t / self.period + self._phase_off)
        return np.maximum(0.2, d)

    def _margin(self, t: int) -> float:
        return 0.25 + 0.2 * np.sin(2 * np.pi * t / self.margin_period)

    def _slack(self, t: int) -> np.ndarray:
        return self._demand(t) * self._margin(t)

    def tap_action(self, consumer_idx: int, overdraw: bool = False) -> int:
        return consumer_idx * 2 + (1 if overdraw else 0)

    # ------------------------------------------------------------------
    # Observation (aggregate suspicion only; per-line suspicion is hidden)
    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        slack = self._slack(self.t)
        load = float(self._demand(self.t).sum())
        recent_alarm = float(np.mean(self._alarm_hist)) if self._alarm_hist else 0.0
        obs = np.concatenate([
            slack / 2.0,                       # per-line slack (observable flow)
            [self.t / self.max_steps],         # time in shift
            [self.lock_remaining / max(1, self.k_lock)],  # lock-out counter (observable)
            [load / self.load_ref],            # aggregate load
            [self.susp.mean()],                # AGGREGATE suspicion index (indirect)
            [recent_alarm],                    # recent alarm rate
        ]).astype(np.float32)
        return obs

    def _info(self, alarm=False, stolen=0.0, tapped=-1, shortfall=0.0) -> dict[str, Any]:
        return {
            "surplus": self.surplus,         # cumulative haul (scoreboard)
            "suspicion": self.susp.copy(),   # true hidden state (for plotting only)
            "slack": self._slack(self.t),
            "load": float(self._demand(self.t).sum()),
            "alarm": alarm,
            "stolen": stolen,                # energy stolen this step (the reward)
            "tapped": tapped,
            "shortfall": shortfall,
            "locked": self.lock_remaining > 0,
            "lock_remaining": self.lock_remaining,
            "t": self.t,
        }

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options=None) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.surplus = 0.0
        self.susp = np.zeros(self.n_consumers)
        self.lock_remaining = 0
        self.t = 0
        self._phase_off = self._rng.uniform(0, 2 * np.pi, size=self.n_consumers)
        self._alarm_hist = deque(maxlen=self.alarm_window)
        return self._get_obs(), self._info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.action_space is not None and not self.action_space.contains(action):
            raise ValueError(f"invalid action {action!r}")

        demand = self._demand(self.t)
        slack = self._slack(self.t)
        load = float(demand.sum())
        sens = self.base_sens * (1.0 + load / self.load_ref)

        alarm = False
        stolen = 0.0
        tapped = -1
        shortfall = 0.0
        locked = self.lock_remaining > 0

        if (not locked) and action != self.LIE_LOW:
            c, overdraw = divmod(action, 2)
            tapped = c
            if overdraw:
                extra = self.overdraw_frac * demand[c]     # dips into delivered demand
                requested = slack[c] + extra
            else:
                extra = 0.0
                requested = slack[c]                        # skim the waste only
            take = max(0.0, requested)                       # haul is uncapped now
            if take > 1e-9:
                # shortfall = how much of what we took came from delivered demand
                shortfall = max(0.0, take - slack[c])
                p = (sens * self.substation_sens[self.substation_of[c]]
                     * (self.base_divert + self.shortfall_weight * shortfall)
                     * (1.0 + self.susp_factor * self.susp[c]))
                self.susp[c] += self.susp_up * take
                if self._rng.random() < min(1.0, p):
                    alarm = True                             # caught: no haul, lock-out follows
                    self.susp[c] += 1.0                      # hidden suspicion spike
                else:
                    stolen = take
                    self.surplus += take                     # haul is kept -- never confiscated

        # Reward = energy stolen this step; the return is the total energy stolen.
        reward = float(stolen)

        # Lock-out bookkeeping: an alarm arms the lock-out; otherwise it counts down.
        if alarm:
            self.lock_remaining = self.k_lock
        elif self.lock_remaining > 0:
            self.lock_remaining -= 1

        self.susp *= self.susp_decay          # all lines cool a little each step
        self._alarm_hist.append(1.0 if alarm else 0.0)

        self.t += 1
        truncated = self.t >= self.max_steps
        return self._get_obs(), reward, False, truncated, self._info(alarm, stolen, tapped, shortfall)

    def render(self) -> str:
        lock = f" LOCKED({self.lock_remaining})" if self.lock_remaining > 0 else ""
        return (f"t={self.t:<3} haul={self.surplus:.1f}"
                f"  load={self._demand(self.t).sum():.1f}  mean_susp={self.susp.mean():.2f}"
                f"  (per-line susp hidden){lock}")
