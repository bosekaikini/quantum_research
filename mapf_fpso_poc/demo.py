from __future__ import annotations

import argparse
import random

from mapf_fpso_poc.baseline import run_episode_baseline
from mapf_fpso_poc.baseline_astar import run_episode_baseline_astar
from mapf_fpso_poc.baseline_astar_reserved_path import run_episode_baseline_astar_reserved_path
from mapf_fpso_poc.baseline_reserved_path import run_episode_baseline_reserved_path
from mapf_fpso_poc.env import GridWorld
from mapf_fpso_poc.sim import run_episode
from mapf_fpso_poc.sim_valley import run_episode_valley
from mapf_fpso_poc.valley_env import ValleyWorld


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
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-baseline", action="store_true", help="Skip standard discrete baseline.")
    p.add_argument(
        "--no-reserved-baseline",
        action="store_true",
        help="Skip reserved-path baseline (trails block other agents).",
    )
    p.add_argument(
        "--no-astar-baseline",
        action="store_true",
        help="Skip A* prioritized-planning baseline.",
    )
    p.add_argument(
        "--no-astar-reserved-baseline",
        action="store_true",
        help="Skip A* baseline with reserved-path trails.",
    )
    args = p.parse_args()

    rng = random.Random(args.seed)
    obstacles = GridWorld.random_obstacles(
        args.width, args.height, density=args.obstacle_density, rng=rng
    )
    env = GridWorld(args.width, args.height, obstacles=obstacles)

    # Sample distinct free starts/goals.
    pts = env.sample_free_positions(args.agents * 2, rng=rng)
    starts = pts[: args.agents]
    goals = pts[args.agents :]

    if not args.no_baseline:
        baseline = run_episode_baseline(
            env, starts, goals, steps=args.steps, seed=args.seed
        )
        _print_summary("BASELINE (discrete grid)", baseline)
        print("")

    if not args.no_astar_baseline:
        astar_result = run_episode_baseline_astar(
            env, starts, goals, steps=args.steps, seed=args.seed
        )
        _print_summary("A* BASELINE (prioritized)", astar_result)
        print("")

    if not args.no_astar_reserved_baseline:
        astar_reserved = run_episode_baseline_astar_reserved_path(
            env, starts, goals, steps=args.steps, seed=args.seed
        )
        _print_summary("A* RESERVED-PATH BASELINE", astar_reserved)
        print(f"A* RESERVED-PATH BASELINE reserved cells: {astar_reserved['reserved_cell_count']}")
        print("")

    if not args.no_reserved_baseline:
        reserved = run_episode_baseline_reserved_path(
            env, starts, goals, steps=args.steps, seed=args.seed
        )
        _print_summary("RESERVED-PATH BASELINE", reserved)
        print(f"RESERVED-PATH BASELINE reserved cells: {reserved['reserved_cell_count']}")
        print("")

    result = run_episode(env, starts, goals, steps=args.steps, seed=args.seed)
    _print_summary("FIREFLY (grid projection)", result)
    print("")

    valley_env = ValleyWorld.from_grid(env)
    valley_result = run_episode_valley(valley_env, starts, goals, steps=args.steps, seed=args.seed)
    _print_summary("FIREFLY (continuous valleys)", valley_result)


if __name__ == "__main__":
    main()
