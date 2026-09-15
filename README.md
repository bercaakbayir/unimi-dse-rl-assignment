# The Energy Thief

Reinforcement Learning final project — UniMI Data Science & Economics (AA 2025-26). Project Code : VFA-4

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
- **Linear FA** (L1, L2, L3) — semi-gradient Q-learning, $q(s,a)=\mathbf{w}_a^\top\mathbf{x}(s)$ on hand-crafted features (bias, lock-out, phase one-hot, per-line suspicion and slack; raw observation at L3).
- **DQN** (L2, L3) — neural $Q(s,a;\theta)$ with experience replay and a target network, on the same input as linear FA.
- **Skim-max-slack rule** — hand-written baseline at every level: always skim the line with the most visible slack.
- **Random policy** — floor at every level.

Agents follow the reference implementations of the course `rlc` package (lectures 1, 3, 5); the environments and experiments are original.

## Results 

| | random | heuristic | tabular Q | linear FA | DQN |
|---|:---:|:---:|:---:|:---:|
| **L1** | 131 | 279 | **264 ± 8** | **277 ± 5** | — |
| **L2** | 76 | 177 | **109 ± 22** | **135 ± 25** | **176 ± 16** |
| **L3** | +107 | 187 | — | **155 ± 13** | **+173 ± 10** |

- **L1:** small and fully observed; both the table and linear FA come within a few percent of the hand-written rule. The table is preferred for readability, not return.
- **L2:** 11,664 states; the table visits 57% of them and its policy degrades during training. Linear FA generalises with 234 weights but stops 40 MWh short of the rule at this budget; DQN on the same features matches the rule.
- **L3:** continuous observation, per-line suspicion hidden. Linear FA trains only with a reduced discount (γ = 0.9); DQN trains stably at γ = 0.99 and gives the best learned policy. Neither exceeds the rule, which ignores suspicion — the hidden state is not the binding constraint.

Details, full MDP definitions, and discussion: `notebooks/level-{1,2,3}.ipynb` (run from the repo root).
