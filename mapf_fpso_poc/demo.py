from __future__ import annotations

import argparse
import random

from mapf_fpso_poc.env import GridWorld
from mapf_fpso_poc.baseline import run_episode_baseline
from mapf_fpso_poc.sim import run_episode


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def _max(values: list[float]) -> float:
    return float(max(values) if values else 0.0)


def _print_summary(label: str, result: dict[str, object]) -> None:
    final_dists = result["final_goal_dists"]
    steps_ran = result["steps_ran"]
    print(f"{label} steps ran: {steps_ran}")
    print(f"{label} mean final goal distance: {_mean(final_dists):.4f}")
    print(f"{label} max final goal distance:  {_max(final_dists):.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agents", type=int, default=6)
    p.add_argument("--width", type=int, default=28)
    p.add_argument("--height", type=int, default=28)
    p.add_argument("--obstacle-density", type=float, default=0.18)
    p.add_argument("--steps", type=int, default=220)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-baseline", action="store_true", help="Skip baseline run.")
    args = p.parse_args()

    rng = random.Random(args.seed)
    obstacles = GridWorld.random_obstacles(args.width, args.height, density=args.obstacle_density, rng=rng)
    env = GridWorld(args.width, args.height, obstacles=obstacles)

    # Sample distinct free starts/goals.
    pts = env.sample_free_positions(args.agents * 2, rng=rng)
    starts = pts[: args.agents]
    goals = pts[args.agents :]

    if not args.no_baseline:
        baseline = run_episode_baseline(env, starts, goals, steps=args.steps, seed=args.seed)
        _print_summary("BASELINE", baseline)
        print("")

    result = run_episode(env, starts, goals, steps=args.steps, seed=args.seed)

    _print_summary("FIREFLY", result)


if __name__ == "__main__":
    main()

