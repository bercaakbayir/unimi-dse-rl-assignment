# The Energy Thief (VFA-4)

Reinforcement Learning final project — UniMI Data Science & Economics (AA 2025-26).

## Scenario

A thief on a **power-grid network** (plant → substations → consumers) steals energy by
redirecting flow off the consumer lines. Each step it can **skim** a line's slack (safe),
**overdraw** into delivered demand (more energy, riskier), or **lie low**. A monitoring
system may raise an **alarm**: the thief steals nothing that step and is **locked out**
for a few steps getting caught costs future stealing time, not the haul already taken.
The reward is the energy stolen each step, so the return is the **total energy stolen
over the shift**.

## Environments

One parameterized environment at three complexity levels (`energy_thief/envs/`):

| Level | Environment | State | Size |
|---|---|---|---|
| **L1** | `GridThiefEnv` — 3 consumers, 1 substation | discrete: demand phase, per-line slack, lock-out | 324 states |
| **L2** | `GridThiefEnvL2` — 6 consumers, 2 substations | discrete: phase, **per-line suspicion** (rises when tapped, cools when idle), lock-out | 11,664 states |
| **L3** | `GridThiefEnvL3` — 9 consumers, 3 substations | **continuous** observation (per-line slack, time, lock-out, aggregate stats); per-line suspicion **hidden** → POMDP | ∞ |

## Agents

All agents act ε-greedily with decaying ε (`energy_thief/agents/`):

- **Tabular Q-learning** (L1, L2) — off-policy TD control on a Q-table.
- **Linear FA** (L2, L3) — semi-gradient Q-learning, $q(s,a)=\mathbf{w}_a^\top\mathbf{x}(s)$ on hand-crafted features (bias, lock-out, phase one-hot, per-line suspicion & slack).
- **DQN** (L3) — neural $Q(s,a;\theta)$ with replay buffer, target network, and a frame-stack of the last 3 observations to infer the hidden suspicion from history.
- **Random policy** — baseline at every level.

## Results (greedy return, MWh; 5 seeds, DQN 3 seeds)

| | random | tabular Q | linear FA | DQN |
|---|:---:|:---:|:---:|:---:|
| **L1** | +131 | **+264 ± 8** | — | — |
| **L2** | +76 | +117 | **+186 ± 14** | — |
| **L3** | +107 | — | +109 | **+151 ± 9** |

- **L1:** the space is small and fully observed — the table alone learns a near-optimal, readable policy (~2× random).
- **L2:** the state explodes (curse of dimensionality; the table visits only 58.8% of states) — linear FA generalises across similar states and beats the strained table with ~650× fewer parameters.
- **L3:** continuous and partially observed — the value depends on the hidden suspicion, so linear FA over aggregate observations sits at random; only the history-aware **DQN** recovers a working policy.

Details, full MDP definitions, and discussion: `notebooks/level-{1,2,3}.ipynb` (run from the repo root).
