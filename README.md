# The Energy Thief (VFA-4)

Reinforcement Learning final project — UniMI Data Science & Economics (AA 2025-26).

A thief moves on a grid-shaped power grid, **siphons** energy into an *unbanked
surplus*, and must reach the **exit** to cash out before a monitoring **alarm**
wipes the surplus. The same parameterized environment is instantiated at three
complexity levels to force a methodological progression:

| Level | Environment | Method |
|-------|-------------|--------|
| **L1** | small discrete grid | tabular Q-learning |
| **L2** | medium grid + per-cell heat | linear function approximation |
| **L3** | large grid, continuous flows, partial observability | DQN |

This README documents **Level 1**: the MDP and the tabular Q-learning algorithm.

---

## The MDP

We model the heist as a finite Markov Decision Process
$\langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle$.

### States $\mathcal{S}$

A state is the pair
$$s = (\text{pos}, u), \qquad \text{pos} \in \{0,\dots,N-1\}^2, \quad u \in \{0,\dots,u_{\max}\},$$
where $\text{pos}$ is the thief's grid cell and $u$ is the current **unbanked
surplus**. With grid size $N=5$ and $u_{\max}=8$, the state is encoded as a
single integer
$$s = (\text{row} \cdot N + \text{col}) \cdot (u_{\max}+1) + u,$$
giving $|\mathcal{S}| = N^2 \cdot (u_{\max}+1) = 25 \cdot 9 = 225$ states. The
start cell $(0,0)$, exit cell $(N-1, N-1)$, and the value map are **fixed for
the whole run**, so every state can be visited enough for a table to converge.

### Actions $\mathcal{A}$

Six discrete actions, $|\mathcal{A}| = 6$:

| id | action | effect |
|----|--------|--------|
| 0–3 | `up, down, left, right` | move one cell (clamped at the walls) |
| 4 | `siphon-low` | steal $v$, alarm probability $p_{\text{low}} = 0.05$ |
| 5 | `siphon-high` | steal $2v$, alarm probability $p_{\text{high}} = 0.20$ |

where $v$ is the value of the current cell (0 if the cell is empty).

### Transitions $P$ and Reward $R$

The dynamics are stochastic only through the alarm. Writing $u'$ for the next
surplus:

- **Move:** deterministic; position updates, $u' = u$, reward $R = 0$.
- **Siphon** on a cell with value $v > 0$:
  - with probability $p$ → **alarm**: $u' = 0$ and $R = -\rho$
    (penalty $\rho = 2.0$);
  - otherwise → $u' = \min(u + kv,\; u_{\max})$ and $R = 0$
    ($k=1$ for low, $k=2$ for high).
- **Reach the exit** (cash out, *terminal*): $R = +u$, then $u' = 0$.
- The episode **truncates** after $T = 50$ steps.

The tension is entirely in the reward: surplus is only worth points once banked
at the exit, and `siphon-high` trades a larger haul for a $4\times$ higher risk
of losing everything.

### Objective

The agent maximizes the expected discounted return
$$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}, \qquad \gamma = 0.99,$$
i.e. it learns the optimal action-value function
$Q^*(s,a) = \max_\pi \mathbb{E}_\pi[G_t \mid S_t = s, A_t = a]$.

---

## Tabular Q-Learning

Off-policy TD control. We keep a table $Q(s,a)$ and, after every transition
$(S, A, R, S')$, move it toward the bootstrapped greedy target:

$$Q(S, A) \leftarrow Q(S, A) + \alpha \Big[\, R + \gamma \max_{a} Q(S', a) - Q(S, A) \,\Big].$$

On a terminal transition there is no future return, so the target is just $R$.
Exploration uses an $\varepsilon$-greedy policy with $\varepsilon$ annealed
between episodes.

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
