"""Quick sanity check for the Level-1 GridThiefEnv.

Runs a random policy and a tiny hand-coded "greedy siphoner" to confirm the
environment is well-formed: spaces, reset/step contract, reward signs, alarms,
and escape/cash-out all behave as intended. No learning here -- just plumbing.

Usage:  python scripts/sanity_check_env.py
"""

from __future__ import annotations

import numpy as np

from energy_thief.envs import GridThiefEnv, ACTION_NAMES


def run_random(env: GridThiefEnv, n_episodes: int = 2000, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    returns, escapes, alarms = [], 0, 0
    for _ in range(n_episodes):
        obs, info = env.reset()
        assert env.observation_space.contains(obs)
        ep_ret, terminated, truncated = 0.0, False, False
        while not (terminated or truncated):
            action = int(rng.integers(env.action_space.n))
            obs, reward, terminated, truncated, info = env.step(action)
            assert env.observation_space.contains(obs)
            ep_ret += reward
            alarms += int(info["alarm"])
        escapes += int(terminated)
        returns.append(ep_ret)
    print(
        f"[random ]  mean return {np.mean(returns):+.2f} +/- {np.std(returns):.2f}  "
        f"| escape rate {escapes / n_episodes:.1%}  | alarms/ep {alarms / n_episodes:.2f}"
    )


def run_greedy(env: GridThiefEnv, n_episodes: int = 2000) -> None:
    """A dumb heuristic: siphon-low if the current cell has value, else head
    toward the exit. Not optimal, but should beat random -- a plumbing check
    that reward actually rewards sensible behaviour."""
    returns, escapes = [], 0
    for _ in range(n_episodes):
        obs, info = env.reset()
        terminated = truncated = False
        ep_ret = 0.0
        while not (terminated or truncated):
            (r, c), surplus = env.decode(obs)
            if env.value_map[r, c] > 0 and surplus < env.surplus_max:
                action = 4  # siphon-low
            else:  # move toward the exit corner (down/right)
                er, ec = env.exit_pos
                action = 1 if r < er else 3  # DOWN then RIGHT
            obs, reward, terminated, truncated, info = env.step(action)
            ep_ret += reward
        escapes += int(terminated)
        returns.append(ep_ret)
    print(
        f"[greedy ]  mean return {np.mean(returns):+.2f} +/- {np.std(returns):.2f}  "
        f"| escape rate {escapes / n_episodes:.1%}"
    )


def show_one_episode(env: GridThiefEnv, seed: int = 3) -> None:
    rng = np.random.default_rng(seed)
    obs, info = env.reset()
    print("\nValue map (numbers = tappable cells, E = exit, A = thief):")
    print(env.render())
    for _ in range(6):
        action = int(rng.integers(env.action_space.n))
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"\n-> {ACTION_NAMES[action]:<12} reward={reward:+.1f} "
              f"alarm={info['alarm']} siphoned={info['siphoned']}")
        print(env.render())
        if terminated or truncated:
            break


def main() -> None:
    env = GridThiefEnv(seed=42)
    print(f"n_states = {env.n_states}  |  n_actions = {env.action_space.n}")
    print(f"start = {env.start_pos}  exit = {env.exit_pos}")
    run_random(env)
    run_greedy(env)
    show_one_episode(env)


if __name__ == "__main__":
    main()
