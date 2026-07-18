# The Energy Thief (VFA-4)

Reinforcement Learning final project — UniMI Data Science & Economics (AA 2025-26).

A thief on a **power-grid network** (plant → substations → consumers) steals energy by
**redirecting flow** off the consumer lines, while a monitoring system may raise an
**alarm**. One parameterized environment at three complexity levels drives the method
progression **tabular Q-learning → linear FA → DQN**.

## The MDP

Finite MDP $\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, P, R, \gamma\rangle$, $\gamma = 0.99$.

**Actions** $\mathcal{A}$ — for each consumer edge $c$: **skim** (divert only the line's
slack $s_c$) or **overdraw** (take $s_c+\Delta$, a shortfall $h$ into delivered demand),
plus **lie-low** (do nothing). $|\mathcal{A}| = 2n+1$.

**Dynamics** $P$ — demand drifts, setting each line's slack $s_c$ and the monitoring
sensitivity $\varsigma$. A tap adds $\text{take}$ to the surplus $U$; it trips the alarm with
$$p = \min\!\big(1,\ \varsigma\,(\beta_0+\beta_h\,h)\ [\,\times\, m_{\text{sub}(c)}(1+\sigma_c)\,]\big),$$
which resets $U\to0$. Skimming ($h=0$) is cheap; overdrawing and a tighter grid are riskier;
the bracket (L2/L3) adds a per-substation factor $m$ and a per-line **suspicion** $\sigma_c$
that rises when tapped and cools when idle.

**Reward** $R = U'$ — the surplus held after the step (an alarm makes it $0$). Maximize
$\mathbb{E}\big[\sum_t \gamma^t R_t\big]$, i.e. learn $q_\star$ satisfying the Bellman optimality
equation $q_\star(s,a)=\mathbb{E}[\,R + \gamma \max_{a'} q_\star(s',a')\,]$.

**State** $\mathcal{S}$ — the levels differ only here:

| Level | state $s$ | size | method |
|-------|-----------|------|--------|
| **L1** | $(\phi,\,U)$ — demand phase, surplus | 36 | tabular Q |
| **L2** | $(\phi,\,U,\,\sigma_1..\sigma_n)$ — + per-line suspicion | 26,244 | tabular Q, **linear FA** |
| **L3** | continuous vector (per-line slack, surplus, aggregate stats); per-line $\sigma$ **hidden** | ∞ (POMDP) | linear FA, DQN |

## Q-learning (tabular, off-policy TD control)

Update: $Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha\big[\,R_{t+1} + \gamma\max_a Q(S_{t+1},a) - Q(S_t,A_t)\,\big]$.

```
Initialize Q(s, a) = 0
for each episode:
    observe S
    for each step:
        A ← ε-greedy(Q, S)
        take A, observe R, S'
        Q(S, A) ← Q(S, A) + α [ R + γ max_a Q(S', a) − Q(S, A) ]
        S ← S'
    ε ← max(ε_min, ε · ε_decay)
```

## Linear FA (semi-gradient Q-learning)

Value is linear in **hand-crafted features** $\mathbf{x}(s)$, one weight vector per action:
$q(s,a) = \mathbf{w}_a^\top \mathbf{x}(s)$. Same update on the weights:

```
Initialize w_a = 0 for each action a
for each episode:
    observe S
    for each step:
        A ← ε-greedy(q, S)            # q(S,a) = w_a · x(S)
        take A, observe R, S'
        δ ← R + γ max_a' ( w_a' · x(S') ) − w_A · x(S)
        w_A ← w_A + α δ x(S)
        S ← S'
    ε ← max(ε_min, ε · ε_decay)
```

Features: **L2** = [bias, surplus, phase one-hot, per-line $\sigma$, per-line slack];
**L3** = the continuous observation + bias.

## Results (5 seeds, greedy return, MWh)

| | random | tabular Q | linear FA |
|---|:---:|:---:|:---:|
| **L1** | ~0 | **+1540** | — |
| **L2** | +790 | +1270 | **+1434** (generalises, beats table) |
| **L3** | +1000 | +1100 (≈ random) | +980 (≈ random) |

L1 tabular works → L2 the state explodes, linear FA generalises and wins → L3 both fail
(continuous + partial-obs; the value depends on the hidden $\sigma$) → DQN.

## Layout

```
energy_thief/
  envs/grid_thief.py        # GridThiefEnv    — L1 (36 states)
  envs/grid_thief_l2.py     # GridThiefEnvL2  — L2 (+ 2 substations, suspicion; 26k states)
  envs/grid_thief_l3.py     # GridThiefEnvL3  — L3 (continuous, partial-obs)
  agents/q_learning.py      # QLearningAgent        — tabular
  agents/linear_q_learning.py  # LinearQLearningAgent — semi-gradient linear FA
notebooks/                  # level-{1,2,3}.ipynb — MDP, env, Q-learning, linear FA, animation
```

Run notebooks from the repo root. Energy is shown in **MW** (1 unit = 5 MW), return in **MWh**.
