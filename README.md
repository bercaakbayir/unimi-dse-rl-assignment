# The Energy Thief (VFA-4)

Reinforcement Learning final project — UniMI Data Science & Economics (AA 2025-26).

A thief taps energy from a **power-grid network** into an *unbanked surplus*, and
must **secure** (bank) it before a monitoring **alarm** wipes the surplus. The
same parameterized environment is instantiated at three complexity levels:

| Level | Environment |
|-------|-------------|
| **L1** | small discrete network |
| **L2** | medium network + adaptive per-node suspicion (heat) |
| **L3** | large network, continuous flows, partial observability |

We evaluate **three algorithms** — tabular Q-learning, linear function
approximation (semi-gradient), and DQN — and run the full **3×3 matrix**: every
algorithm on every level. The diagonal is the *hypothesized* best fit (tabular
for L1, linear for L2, DQN for L3), but the point of the experiment is the
**off-diagonal** cells: showing *why* the tabular table breaks down as the state
space grows, and where approximation becomes necessary, is the evidence that
justifies each method transition.

|            | L1 (small) | L2 (medium + heat) | L3 (large, partial-obs) |
|------------|:----------:|:------------------:|:-----------------------:|
| **Tabular Q-learning** | ✔ expected fit | curse of dimensionality | infeasible |
| **Linear FA** | works, redundant | ✔ expected fit | limited by features |
| **DQN** | works, over-engineered | works | ✔ expected fit |

This README documents **Level 1**: the MDP and the tabular Q-learning algorithm.
The other levels and algorithms are added as the project progresses.

---

## The Scenario

The agent is an energy thief on a **power-grid network**: a **plant** generates
energy that flows through a **substation** out to several **consumers**. Because
generation never matches demand exactly, some edges carry more than their consumer
needs — that excess is **slack** (the grid's inefficiency).

The thief steals by **redirecting flow** off an edge into an *unbanked surplus*:

- **skim** an edge — divert only its slack (the waste). No consumer goes short, so
  detection risk is low;
- **overdraw** an edge — take the slack *plus* some delivered demand, starving the
  consumer. Bigger haul, but the shortfall is conspicuous, so the **monitoring
  system** is far likelier to raise an **alarm**;
- **secure** — bank part of the surplus as reward.

An alarm wipes the unbanked surplus and costs a penalty. The grid cycles through
discrete **demand phases**; each phase fixes the flow and slack on every edge and
the monitoring sensitivity, so *where* it is worth redirecting flow — and how tight
the grid is — changes over time. No model of the alarm probabilities is given to
the agent; it learns from experience.

---

## The MDP

We model the heist as a finite Markov Decision Process
$$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, P, R, \gamma, \rho_0 \rangle.$$
The grid is a plant → substation → $n$ consumers. Each consumer edge $c$ carries a
flow $f_c$ with divertible slack $s_c$, and a **demand phase** $\phi$ fixes the flows,
slacks, and monitoring sensitivity $\varsigma(\phi)$ (higher phase = tighter grid,
less slack, closer watch).

### State space $\mathcal{S}$

Slack is a deterministic function of the demand phase, so the state is just the phase
and the unbanked surplus:
$$s = (\phi, U), \qquad \phi \in \{0,\dots,n_\phi-1\}, \quad U \in \{0,\dots,U_{\max}\},$$
$$\operatorname{id}(s) = \phi\,(U_{\max}+1) + U, \qquad |\mathcal{S}| = n_\phi(U_{\max}+1) = 4 \cdot 9 = 36.$$
No terminal state (fixed-horizon continuing task); $\rho_0$ deterministic at $(\phi,U)=(0,0)$.

### Action space $\mathcal{A}$

Skim or overdraw each consumer edge, or secure:
$$\mathcal{A} = \{\textsf{skim-}c,\ \textsf{overdraw-}c : c \in \text{consumers}\} \cup \{\textsf{secure}\}, \qquad |\mathcal{A}| = 2n + 1 = 7.$$

### Transition kernel $P$

The demand phase drifts by an exogenous random walk
$\phi' = \operatorname{clip}(\phi + \xi)$, $\xi \in \{-1,0,+1\}$ w.p. $(0.25,0.5,0.25)$.
A tap on edge $c$ diverts $\text{steal}$ energy (skim: $s_c$; overdraw: $s_c + \Delta$,
capped at the flow), causing a delivered-demand **shortfall** $h = \max(0, \text{steal}-s_c)$.
The alarm fires with
$$p = \min\!\big(1,\ \varsigma(\phi)\,(\beta_0 + \beta_h\,h)\big),$$
so skimming waste ($h=0$) is cheap and overdrawing (a visible shortfall) is
conspicuous, and a tighter grid (higher $\varsigma$) is riskier throughout. On an
alarm $U \to 0$; otherwise the haul is added, $U' = \min(U + \text{steal}, U_{\max})$.
A **secure** banks $b = \min(U, B)$ (bank rate $B=4$): $U' = U - b$.

### Reward $R$

Zero except when banking energy (secure) or tripping the alarm (penalty $\varrho = 2.0$):
$$
R(s, a, s') =
\begin{cases}
+b, & a = \textsf{secure} \quad (\text{banked energy}) \\[2pt]
-\varrho, & a \text{ is a tap and the alarm fires} \\[2pt]
0, & \text{otherwise.}
\end{cases}
$$
The tension: surplus is worthless until secured, and an alarm wipes whatever is
unbanked — so the thief **skims the highest-slack edge while the grid is loose** and
**secures / backs off** when it tightens. The episode **truncates** after $T = 50$
steps (no terminal state; bootstrap on truncation, $\gamma < 1$).

### Objective

The agent seeks a policy $\pi$ maximizing the expected $\gamma$-discounted return
($\gamma = 0.99$)
$$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1},$$
equivalently the optimal action-value function
$$q_\star(s,a) = \max_\pi \mathbb{E}_\pi\!\left[ G_t \mid S_t = s,\ A_t = a \right],$$
which satisfies the Bellman optimality equation
$$q_\star(s,a) = \sum_{s'} P(s' \mid s, a)\Big[\, R(s,a,s') + \gamma \max_{a'} q_\star(s', a') \,\Big].$$
Knowing $q_\star$ is enough to act optimally, since the greedy policy is optimal:
$\pi_\star(s) \in \arg\max_a q_\star(s,a)$.

---

## Tabular Q-Learning

Off-policy TD control: learn $q_\star$ directly from experience, without knowing
$P$. We keep a table $Q(s,a)$ and, after each observed transition
$(S_t, A_t, R_{t+1}, S_{t+1})$, replace the expectation in the Bellman optimality
equation by its one-sample estimate and take a step toward it:

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \Big[\, \underbrace{R_{t+1} + \gamma \max_{a} Q(S_{t+1}, a)}_{\text{TD target}} - Q(S_t, A_t) \,\Big].$$

It is *off-policy* because the target bootstraps on $\max_a Q(S_{t+1}, a)$ — the
greedy value — regardless of the (exploratory) action actually taken next. On a
terminal transition there is no future return, so the target is just $R_{t+1}$.
Behaviour is $\varepsilon$-greedy with $\varepsilon$ annealed between episodes,
so exploration gives way to exploitation as the estimates improve.

### Pseudocode

```
Q-learning (off-policy TD control) for estimating π ≈ π*

Parameters: step size α ∈ (0,1], discount γ, exploration ε
Initialize Q(s, a) = 0 for all s ∈ S, a ∈ A

Loop for each episode:
    Initialize S
    Loop for each step of the episode:
        Choose A from S using ε-greedy policy derived from Q
        Take action A, observe R, S', terminated
        target ← R                       if terminated
        target ← R + γ · max_a Q(S', a)  otherwise
        Q(S, A) ← Q(S, A) + α · [ target − Q(S, A) ]
        S ← S'
    until S is terminal or the episode truncates
    Anneal ε ← max(ε_min, ε · ε_decay)
```

### Hyperparameters (Level 1)

| symbol | name | value |
|--------|------|-------|
| $\alpha$ | learning rate | $0.1$ |
| $\gamma$ | discount | $0.99$ |
| $\varepsilon$ | exploration (start → min) | $1.0 \to 0.05$ |
| $\varepsilon_{\text{decay}}$ | per-episode decay | $0.999$ |

Trained over 5 seeds × 8 000 episodes, the greedy policy reaches a mean return
$\approx +54$, and the learned policy is legible: it **skims the consumer edge with
the most slack in each demand phase**, and secures / backs off when the grid
tightens (little slack, high sensitivity).

---

## Layout

```
energy_thief/
  envs/grid_thief.py      # GridThiefEnv   — L1 power-grid network MDP (36 states)
  envs/grid_thief_l2.py   # GridThiefEnvL2 — L2 medium network + adaptive suspicion (26k states)
  agents/q_learning.py    # QLearningAgent — tabular off-policy TD control
notebooks/
  level-1.ipynb           # L1: training, learning + eval curves, ε schedule, policy, timeline
  level-2.ipynb           # L2: same + state-coverage (curse of dimensionality), suspicion timeline
scripts/
  sanity_check_env.py     # random + heuristic plumbing check
```

Run the notebook from the repo root (it inserts `..` on the path so the
`energy_thief` package imports).
