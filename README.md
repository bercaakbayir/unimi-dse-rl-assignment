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

The agent is an energy thief operating on a **power-grid network** — nodes are
plants, substations, and consumers, joined by transmission **edges** that carry
energy. Where supply and demand don't quite match, an edge carries exploitable
slack. The thief **taps** an edge to divert energy into an *unbanked surplus*,
then **secures** (banks) that surplus as reward.

The catch is a **monitoring system**. Every tap risks tripping an **alarm**, and
the probability grows with **how aggressively** the thief operates (a high-intensity
tap on a juicy line) and with the current **grid load** (a busy grid is watched
more closely). When the alarm fires, the entire unbanked surplus is wiped and a
penalty is incurred.

So the thief faces a sequence of coupled decisions under uncertainty:

- **Which line to tap** — the safe trickle or the juicy, closely-watched one.
- **How aggressively** — a bigger haul now vs. a higher chance of losing it.
- **Tap or bank** — keep stealing while the grid is quiet, or secure the surplus
  before the load rises and an alarm wipes it.

There is no single "right" move; the value of an action depends on the load and
the surplus currently at risk. This is a **sequential decision problem under
uncertainty**, formalized as a Markov Decision Process and solved by learning
from experience — no model of the alarm probabilities is given to the agent.

---

## The MDP

We model the heist as a finite Markov Decision Process
$$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, P, R, \gamma, \rho_0 \rangle,$$
with state space $\mathcal{S}$, action space $\mathcal{A}$, transition kernel
$P(s' \mid s, a)$, reward function $R(s, a, s')$, discount $\gamma \in (0,1)$,
and initial-state distribution $\rho_0$.

The grid is a small fixed network with tappable edges $\mathcal{E}$; each edge
$e$ has a base haul $g_e$ and a base alarm risk $r_e$ (default: edge A a safe
trickle $(1, 0.03)$, edge B juicier but watched $(2, 0.08)$). A grid **load** $L$
scales the monitoring sensitivity via a per-load multiplier $\rho(L)$.

### State space $\mathcal{S}$

The agent does not move in space; the state is the current grid load and the
unbanked surplus, encoded as a single integer:
$$s = (L, U), \qquad L \in \{0,\dots,n_L-1\}, \quad U \in \{0,\dots,U_{\max}\},$$
$$\mathcal{S} \;\cong\; \{0, 1, \dots, n_L(U_{\max}+1) - 1\}, \qquad
\operatorname{id}(s) = L\,(U_{\max}+1) + U,$$
so $|\mathcal{S}| = n_L (U_{\max}+1) = 4 \cdot 9 = 36$. There is **no terminal
state** (fixed-horizon continuing task); $\rho_0$ is deterministic at
$(L,U) = (0,0)$.

### Action space $\mathcal{A}$

Tap each edge at low or high intensity, or secure:
$$\mathcal{A} = \{\textsf{tap-}e\textsf{-low},\ \textsf{tap-}e\textsf{-high} : e \in \mathcal{E}\} \cup \{\textsf{secure}\}, \qquad |\mathcal{A}| = 2|\mathcal{E}| + 1 = 5.$$
A **high** tap multiplies both haul and risk — the "aggressiveness" — by
$\kappa_g = 2$ and $\kappa_r = 2.5$ respectively.

### Transition kernel $P$

The load drifts on its own via an exogenous random walk, independent of the
action:
$$L' = \operatorname{clip}(L + \xi,\ 0,\ n_L-1), \qquad \xi \in \{-1,0,+1\}\ \text{w.p.}\ (0.25, 0.5, 0.25).$$
A **tap** on edge $e$ at intensity $k \in \{\text{low}, \text{high}\}$ trips the
alarm with probability increasing in aggressiveness and load; otherwise it adds
to the surplus:
$$
U' =
\begin{cases}
0, & \text{w.p. } p = \min\!\big(1,\ r_e\,\kappa_r^{[k=\text{high}]}\,\rho(L)\big) \quad (\text{alarm}) \\[2pt]
\min\!\big(U + g_e\,\kappa_g^{[k=\text{high}]},\ U_{\max}\big), & \text{otherwise.}
\end{cases}
$$
A **secure** action banks $b = \min(U, B)$ (bank rate $B = 4$): $U' = U - b$.

### Reward $R$

Zero except when banking energy (secure) or tripping the alarm (penalty
$\varrho = 2.0$):
$$
R(s, a, s') =
\begin{cases}
+b, & a = \textsf{secure} \quad (\text{banked energy}) \\[2pt]
-\varrho, & a \text{ is a tap and the alarm fires} \\[2pt]
0, & \text{otherwise.}
\end{cases}
$$
The tension: surplus is worthless until secured, and an alarm wipes whatever is
unbanked — so the thief must read the load, tap while the grid is quiet, and bank
before pushing its luck. The episode **truncates** after $T = 50$ steps (a fixed
horizon; there is no terminal state, so the agent bootstraps on truncation and
relies on $\gamma < 1$).

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

Trained over 5 seeds × 8 000 episodes, the greedy policy **beats** the hand-coded
heuristic: mean greedy return $\approx +56$ (Q-learning) vs $\approx +38$
(heuristic) vs $\approx +11$ (random).

---

## Layout

```
energy_thief/
  envs/grid_thief.py      # GridThiefEnv — the L1 power-grid network MDP
  agents/q_learning.py    # QLearningAgent — tabular off-policy TD control
notebooks/
  01_level1_tabular.ipynb # L1 experiments: training, curves, baselines, policy
scripts/
  sanity_check_env.py     # random + heuristic plumbing check
```

Run the notebook from the repo root (it inserts `..` on the path so the
`energy_thief` package imports).
