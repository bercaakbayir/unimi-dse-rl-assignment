# The Energy Thief (VFA-4)

Reinforcement Learning final project — UniMI Data Science & Economics (AA 2025-26).

A thief moves on a grid-shaped power grid, **siphons** energy into an *unbanked
surplus*, and must reach the **exit** to cash out before a monitoring **alarm**
wipes the surplus. The same parameterized environment is instantiated at three
complexity levels:

| Level | Environment |
|-------|-------------|
| **L1** | small discrete grid |
| **L2** | medium grid + per-cell heat (adaptive monitoring) |
| **L3** | large grid, continuous flows, partial observability |

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

A thief has broken into the control room of a city power grid. Each cell of the
grid holds a fixed amount of divertible energy; standing on a rich cell, the
thief can **siphon** power into a portable battery — the *unbanked surplus*.
Stolen energy is worthless while it sits in the battery: the only way to score
is to carry it to the **exit** (a black-market drop point) and **cash out**.

The catch is a **monitoring system**. Every siphon has a chance of tripping an
**alarm**; when it does, security locks the grid down and the thief loses the
entire battery — the accumulated surplus is wiped and a penalty is incurred.
Siphoning harder (`siphon-high`) steals twice as much per step but is four times
more likely to trip the alarm. And the clock is running: after a fixed number of
steps the heist ends whether or not the thief has cashed out.

So the thief faces a sequence of coupled decisions under uncertainty:

- **Where to steal** — which rich cells are worth the detour from the exit.
- **How aggressively** — the greedy `siphon-high` vs the safer `siphon-low`.
- **When to stop and run** — bank a modest, safe haul now, or gamble on a bigger
  one and risk losing everything to the alarm.

There is no single "right" move; the value of an action depends on the surplus
already at risk and the distance still to travel. This is exactly a **sequential
decision problem under uncertainty**, which we formalize as a Markov Decision
Process and solve by learning from experience — no model of the alarm
probabilities is given to the agent.

---

## The MDP

We model the heist as a finite Markov Decision Process
$$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, P, R, \gamma, \rho_0 \rangle,$$
with state space $\mathcal{S}$, action space $\mathcal{A}$, transition kernel
$P(s' \mid s, a)$, reward function $R(s, a, s')$, discount $\gamma \in (0,1)$,
and initial-state distribution $\rho_0$.

Fix the grid side $N = 5$, the surplus cap $u_{\max} = 8$, and a **static** value
map $V : \{0,\dots,N-1\}^2 \to \{0,1,2\}$ giving the divertible energy of each
cell. The start cell $s_{\text{start}} = (0,0)$, the exit cell
$e = (N-1, N-1)$, and $V$ are fixed for the whole run.

### State space $\mathcal{S}$

A state is a position–surplus pair, encoded as a single integer (as in the
lecture notebooks):
$$s = (r, c, u), \qquad (r,c) \in \{0,\dots,N-1\}^2, \quad u \in \{0,\dots,u_{\max}\},$$
$$\mathcal{S} \;\cong\; \{0, 1, \dots, N^2(u_{\max}+1) - 1\}, \qquad
\operatorname{id}(s) = (r N + c)(u_{\max}+1) + u,$$
so $|\mathcal{S}| = N^2 (u_{\max}+1) = 25 \cdot 9 = 225$. The **terminal** states
are those on the exit cell, $\mathcal{S}_{\text{term}} = \{(r,c,u) : (r,c) = e\}$.
The initial distribution is deterministic:
$\rho_0(s) = \mathbb{1}[\,s = (0,0,0)\,]$.

### Action space $\mathcal{A}$

$$\mathcal{A} = \{0,1,2,3,4,5\} = \{\uparrow,\downarrow,\leftarrow,\rightarrow,\ \textsf{siphon-low},\ \textsf{siphon-high}\}.$$
Moves have displacement $\delta_a \in \{(-1,0),(1,0),(0,-1),(0,1)\}$; the two
siphons carry a haul multiplier $k_a$ and an alarm probability $p_a$:
$$k_{\textsf{low}} = 1,\ p_{\textsf{low}} = 0.05, \qquad
k_{\textsf{high}} = 2,\ p_{\textsf{high}} = 0.20.$$

### Transition kernel $P$

The dynamics are deterministic except for the alarm on a siphon. Let
$v = V(r,c)$ be the current cell value.

**Move** $a \in \{\uparrow,\downarrow,\leftarrow,\rightarrow\}$ — deterministic;
position clamps at the walls, surplus is unchanged:
$$(r',c') = \operatorname{clip}\big((r,c) + \delta_a,\ 0,\ N-1\big), \qquad
P\big((r',c',u) \mid (r,c,u), a\big) = 1.$$

**Siphon** $a \in \{\textsf{low}, \textsf{high}\}$ — position unchanged; on an
empty cell ($v = 0$) it is a no-op, otherwise the alarm fires with probability
$p_a$:
$$
P\big(s' \mid (r,c,u), a\big) =
\begin{cases}
1, & v = 0,\ s' = (r,c,u) \\[2pt]
p_a, & v > 0,\ s' = (r,c,\,0) \quad (\text{alarm}) \\[2pt]
1 - p_a, & v > 0,\ s' = \big(r,c,\ \min(u + k_a v,\ u_{\max})\big).
\end{cases}
$$

### Reward $R$

Reward is zero everywhere except at the two decisive events — cashing out at the
exit, and tripping the alarm (penalty $\rho = 2.0$):
$$
R(s, a, s') =
\begin{cases}
+u, & a \text{ is a move and } (r',c') = e \quad (\text{cash out}) \\[2pt]
-\rho, & a \text{ is a siphon and the alarm fires} \\[2pt]
0, & \text{otherwise.}
\end{cases}
$$
The tension is entirely here: surplus is worthless until banked at the exit, and
`siphon-high` trades a doubled haul for a $4\times$ higher chance of losing it
all. The episode ends when a terminal state is reached, or **truncates** after
$T = 50$ steps (a time limit, not part of $P$).

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

Trained over 5 seeds × 10 000 episodes, the greedy policy matches the hand-coded
heuristic baseline (return $\approx +7.4$ vs a random baseline near $0$).

---

## Layout

```
energy_thief/
  envs/grid_thief.py      # GridThiefEnv — the parameterized MDP
  agents/q_learning.py    # QLearningAgent — tabular off-policy TD control
notebooks/
  01_level1_tabular.ipynb # L1 experiments: training, curves, baselines, rollout
scripts/
  sanity_check_env.py     # random + heuristic plumbing check
```

Run the notebook from the repo root (it inserts `..` on the path so the
`energy_thief` package imports).
