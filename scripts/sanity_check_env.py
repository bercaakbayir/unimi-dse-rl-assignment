"""Quick sanity check for the Level-1 GridThiefEnv (power-grid network).

Runs a random policy and a tiny hand-coded heuristic to confirm the environment
is well-formed: spaces, reset/step contract, reward signs, alarms, and secure/bank
all behave as intended. No learning here -- just plumbing.

Usage:  PYTHONPATH=. python scripts/sanity_check_env.py
"""

from __future__ import annotations

import numpy as np

from energy_thief.envs import GridThiefEnv, ACTION_NAMES


def run_random(env: GridThiefEnv, n_episodes: int = 2000, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    returns, alarms = [], 0
    for _ in range(n_episodes):
        obs, info = env.reset()
        assert env.observation_space.contains(obs)
        ep_ret, terminated, truncated = 0.0, False, False
        while not (terminated or truncated):
            action = int(rng.integers(env.n_actions))
            obs, reward, terminated, truncated, info = env.step(action)
            assert env.observation_space.contains(obs)
            ep_ret += reward
            alarms += int(info["alarm"])
        returns.append(ep_ret)
    print(
        f"[random ]  mean return {np.mean(returns):+.2f} +/- {np.std(returns):.2f}  "
        f"| alarms/ep {alarms / n_episodes:.2f}"
    )


def run_heuristic(env: GridThiefEnv, n_episodes: int = 2000) -> None:
    """Tap aggressively while the grid is quiet, trickle under some load, and
    secure once the surplus is worth banking or the grid gets busy. Not optimal,
    but should beat random -- a plumbing check that reward rewards sense."""
    returns = []
    for _ in range(n_episodes):
        obs, info = env.reset()
        terminated = truncated = False
        ep_ret = 0.0
        while not (terminated or truncated):
            load, surplus = env.decode(obs)
            if surplus >= env.bank_rate or load >= 2:
                action = env.SECURE
            elif load == 0:
                action = env.tap_action(1, high=True)
            else:
                action = env.tap_action(0, high=False)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_ret += reward
        returns.append(ep_ret)
    print(f"[heurist]  mean return {np.mean(returns):+.2f} +/- {np.std(returns):.2f}")


def show_one_episode(env: GridThiefEnv, seed: int = 3) -> None:
    rng = np.random.default_rng(seed)
    obs, info = env.reset()
    print("\nOne random shift (load bar grows with grid load):")
    print(env.render())
    for _ in range(8):
        action = int(rng.integers(env.n_actions))
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"-> {ACTION_NAMES[action]:<12} reward={reward:+.1f} "
              f"alarm={info['alarm']} stolen={info['stolen']} banked={info['banked']}")
        print("   " + env.render())
        if terminated or truncated:
            break


def main() -> None:
    env = GridThiefEnv(seed=1)
    print(f"n_states = {env.n_states}  |  n_actions = {env.n_actions}")
    print(f"edges = {env.edges}  |  load_risk = {env.load_risk}")
    run_random(env)
    run_heuristic(env)
    show_one_episode(env)


if __name__ == "__main__":
    main()
