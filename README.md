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
sensitivity $\varsigma$. A tap steals $\text{take}$ energy; it trips the alarm with
$$p = \min\!\big(1,\ \varsigma\,(\beta_0+\beta_h\,h)\ [\,\times\, m_{\text{sub}(c)}(1+\sigma_c)\,]\big).$$
Skimming ($h=0$) is cheap; overdrawing and a tighter grid are riskier; the bracket (L2/L3)
adds a per-substation factor $m$ and a per-line **suspicion** $\sigma_c$ that rises when
tapped and cools when idle. **On an alarm** the thief steals nothing that step and is
**locked out** for $k_{\text{lock}}$ steps (taps become no-ops), and the monitoring spikes —
getting caught costs *future* stealing time, not the haul already taken.

**Reward** $R = \text{energy stolen this step}$ (0 on lie-low, a no-op, or while locked out),
so the return is the **total energy stolen** over the shift. Maximize
$\mathbb{E}\big[\sum_t \gamma^t R_t\big]$, i.e. learn $q_\star$ satisfying the Bellman optimality
equation $q_\star(s,a)=\mathbb{E}[\,R + \gamma \max_{a'} q_\star(s',a')\,]$. The accumulated haul
is the *score*, not part of the state.

**State** $\mathcal{S}$ — the grid condition the thief reads to act; the levels differ only here:

| Level | state $s$ | size | method |
|-------|-----------|------|--------|
| **L1** | $(\phi,\,s_1..s_n,\,\ell)$ — phase, per-line slack, lock-out | 324 | tabular Q |
| **L2** | $(\phi,\,\sigma_1..\sigma_n,\,\ell)$ — + per-line suspicion | 11,664 | tabular Q, **linear FA** |
| **L3** | continuous vector (per-line slack, time, lock-out, aggregate stats); per-line $\sigma$ **hidden** | ∞ (POMDP) | linear FA, **DQN** |

This realizes the brief's three levels — **small discrete grid → medium grid with richer state →
large grid with continuous variables**. Each level foregrounds the one axis that motivates its
method: L1 puts *stochastic slack* in the state (a non-trivial small table); L2's **richer state**
adds the *adaptive per-line suspicion* $\sigma_c$ (the $k^n$ explosion), with slack kept a
deterministic function of $\phi$ so the space stays ~$10^4$ rather than ~$10^6$ and the tabular
baseline remains meaningful; L3 makes the flows *continuous* and the suspicion *hidden* (POMDP).
The levels are thus increasing in complexity but not strictly nested — a deliberate modeling choice.

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

Features: **L2** = [bias, lock-out, phase one-hot, per-line $\sigma$, per-line slack];
**L3** = the continuous observation + bias.

## DQN (Level 3)

A neural network $Q(s,a;\theta)$ trained on the same one-sample Bellman target, stabilised by a
**replay buffer** + a **target network**, and fed a **frame-stack of the last $k=3$ observations**
so it can infer the hidden per-line suspicion from history — what a linear map of one aggregate
observation cannot do.

## Results (5 seeds, greedy return, MWh)

| | random | tabular Q | linear FA | DQN |
|---|:---:|:---:|:---:|:---:|
| **L1** | +131 | **+264** | — | — |
| **L2** | +76 | +117 | **+191** (generalises, beats table) | — |
| **L3** | +107 | — | +109 (≈ random) | **+161** |

L1 tabular works → L2 the state explodes, linear FA generalises and beats the strained table → L3
continuous + partial-obs, the value depends on the hidden $\sigma$ so linear FA sits at random and
only the **DQN** copes.

**Tradeoffs.** The progression buys **policy quality** at the cost of **interpretability** and
**sample efficiency**: the L1 table gives a legible, inspectable policy from little experience; the
DQN reaches a working policy where the others can't, but needs far more experience and is a black
box. Linear FA sits in between — cheap and still readable through its weights, as long as the state
is fully observed.

## Layout

```
energy_thief/
  envs/grid_thief.py        # GridThiefEnv    — L1 (324 states)
  envs/grid_thief_l2.py     # GridThiefEnvL2  — L2 (+ 2 substations, suspicion; 11,664 states)
  envs/grid_thief_l3.py     # GridThiefEnvL3  — L3 (continuous, partial-obs)
  agents/q_learning.py      # QLearningAgent        — tabular
  agents/linear_q_learning.py  # LinearQLearningAgent — semi-gradient linear FA
  agents/dqn.py             # DQNAgent              — replay + target net (needs torch)
notebooks/                  # level-{1,2,3}.ipynb — MDP, env, training, results, animations
```

Run notebooks from the repo root. Energy is shown in **MW** (1 unit = 5 MW), return in **MWh**.
