from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mapf_fpso_poc.env import GridWorld
from mapf_fpso_poc.baseline import run_episode_baseline
from mapf_fpso_poc.sim import run_episode


ROOT_DIR = Path(__file__).resolve().parents[2]


def _norm(p: Sequence[float]) -> float:
    return math.sqrt(float(p[0]) * float(p[0]) + float(p[1]) * float(p[1]))


def _pairwise_collision_counts(
    traj: List[List[Tuple[float, float]]],
    r_min: float,
) -> List[int]:
    """Count near-collisions per timestep."""
    counts: List[int] = []
    for positions in traj:
        c = 0
        n = len(positions)
        for i in range(n):
            for j in range(i + 1, n):
                dx = float(positions[i][0] - positions[j][0])
                dy = float(positions[i][1] - positions[j][1])
                if dx * dx + dy * dy < r_min * r_min:
                    c += 1
        counts.append(c)
    return counts


def _goal_dist_series(
    traj: List[List[Tuple[float, float]]],
    goals: List[Tuple[float, float]],
) -> Tuple[List[float], List[float]]:
    mean_d: List[float] = []
    max_d: List[float] = []
    for positions in traj:
        ds = [_norm((positions[i][0] - goals[i][0], positions[i][1] - goals[i][1])) for i in range(len(goals))]
        if not ds:
            mean_d.append(0.0)
            max_d.append(0.0)
        else:
            mean_d.append(float(sum(ds) / len(ds)))
            max_d.append(float(max(ds)))
    return mean_d, max_d


def _run_single_scenario(
    seed: int,
    num_agents: int = 6,
    width: int = 28,
    height: int = 28,
    obstacle_density: float = 0.18,
    steps: int = 220,
    r_min: float = 0.06,
) -> dict:
    import random

    rng = random.Random(seed)
    obstacles = GridWorld.random_obstacles(width, height, density=obstacle_density, rng=rng)
    env = GridWorld(width, height, obstacles=obstacles)
    pts = env.sample_free_positions(num_agents * 2, rng=rng)
    starts = pts[:num_agents]
    goals = pts[num_agents:]

    baseline = run_episode_baseline(env, starts, goals, steps=steps, seed=seed)
    firefly = run_episode(env, starts, goals, steps=steps, seed=seed)

    base_traj = baseline["traj"]
    ff_traj = firefly["traj"]

    base_mean_d, base_max_d = _goal_dist_series(base_traj, goals)
    ff_mean_d, ff_max_d = _goal_dist_series(ff_traj, goals)

    base_col = _pairwise_collision_counts(base_traj, r_min=r_min)
    ff_col = _pairwise_collision_counts(ff_traj, r_min=r_min)

    return {
        "seed": seed,
        "env": env,
        "starts": starts,
        "goals": goals,
        "baseline": baseline,
        "firefly": firefly,
        "series": {
            "baseline_mean_dist": base_mean_d,
            "baseline_max_dist": base_max_d,
            "baseline_collisions": base_col,
            "firefly_mean_dist": ff_mean_d,
            "firefly_max_dist": ff_max_d,
            "firefly_collisions": ff_col,
        },
    }


def plot_trajectories(env: GridWorld, starts, goals, baseline_traj, firefly_traj, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    titles = ["Baseline (discrete MAPF, greedy)", "Firefly/FPSO (continuous swarm)"]

    for ax, traj, title in zip(axes, [baseline_traj, firefly_traj], titles):
        ax.set_title(title)
        ax.set_xlim(env.bounds.xmin, env.bounds.xmax)
        ax.set_ylim(env.bounds.ymin, env.bounds.ymax)
        ax.set_aspect("equal", adjustable="box")

        dx, dy = env.cell_size()
        shrink = 0.85  # shrink obstacles slightly so agent markers sit clearly in free space
        for center in env.iter_obstacle_centers():
            ax.add_patch(
                plt.Rectangle(
                    (center[0] - (dx * shrink) / 2.0, center[1] - (dy * shrink) / 2.0),
                    dx * shrink,
                    dy * shrink,
                    facecolor="black",
                    edgecolor="black",
                    alpha=0.25,
                )
            )

        colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(starts))))
        for i, (s, g) in enumerate(zip(starts, goals)):
            c = colors[i % len(colors)]
            xs = [p[i][0] for p in traj]
            ys = [p[i][1] for p in traj]
            # Draw only markers at sampled positions so it never visually looks like
            # an agent is "inside" an obstacle, even when hugging walls.
            ax.scatter(xs, ys, color=c, s=10, alpha=0.9)
            ax.scatter([s[0]], [s[1]], color=c, marker="o", s=40, edgecolor="white", linewidth=0.5, zorder=5)
            ax.scatter([g[0]], [g[1]], color=c, marker="*", s=80, edgecolor="white", linewidth=0.5, zorder=5)

        ax.grid(True, linestyle=":", alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_time_series(series: dict, out_path: Path) -> None:
    n_steps = max(len(series["baseline_mean_dist"]), len(series["firefly_mean_dist"]))
    t = np.arange(n_steps)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1, ax2 = axes
    ax1.plot(t[: len(series["baseline_max_dist"])], series["baseline_max_dist"], label="Baseline max dist", color="C0")
    ax1.plot(t[: len(series["firefly_max_dist"])], series["firefly_max_dist"], label="Firefly max dist", color="C1")
    ax1.set_ylabel("Max distance to goal")
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle=":", alpha=0.3)

    ax2.plot(t[: len(series["baseline_collisions"])], series["baseline_collisions"], label="Baseline near-collisions", color="C0")
    ax2.plot(t[: len(series["firefly_collisions"])], series["firefly_collisions"], label="Firefly near-collisions", color="C1")
    ax2.set_ylabel(f"Pairs with d < r_min")
    ax2.set_xlabel("Step")
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle=":", alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def aggregate_table(results: Iterable[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        base = r["baseline"]
        ff = r["firefly"]
        base_d = base["final_goal_dists"]
        ff_d = ff["final_goal_dists"]
        rows.append(
            {
                "seed": r["seed"],
                "baseline_steps": base["steps_ran"],
                "baseline_mean_dist": float(sum(base_d) / max(1, len(base_d))),
                "baseline_max_dist": float(max(base_d) if base_d else 0.0),
                "firefly_steps": ff["steps_ran"],
                "firefly_mean_dist": float(sum(ff_d) / max(1, len(ff_d))),
                "firefly_max_dist": float(max(ff_d) if ff_d else 0.0),
            }
        )
    return pd.DataFrame(rows)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run FPSO-MAPF analysis and generate plots/tables.")
    parser.add_argument("--agents", type=int, default=6)
    parser.add_argument("--width", type=int, default=28)
    parser.add_argument("--height", type=int, default=28)
    parser.add_argument("--obstacle-density", type=float, default=0.18)
    parser.add_argument("--steps", type=int, default=220)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--r-min", type=float, default=0.06)
    args = parser.parse_args()

    results = []
    for seed in range(args.seeds):
        res = _run_single_scenario(
            seed=seed,
            num_agents=args.agents,
            width=args.width,
            height=args.height,
            obstacle_density=args.obstacle_density,
            steps=args.steps,
            r_min=args.r_min,
        )
        results.append(res)

    # Use the first scenario for trajectory/time-series plots (for a visually clean example).
    first = results[0]
    plots_dir = ROOT_DIR / "plots" / "mapf_fpso_poc"

    plot_trajectories(
        first["env"],
        first["starts"],
        first["goals"],
        first["baseline"]["traj"],
        first["firefly"]["traj"],
        plots_dir / "trajectories_baseline_vs_firefly.png",
    )
    plot_time_series(first["series"], plots_dir / "distances_and_collisions.png")

    # Aggregate table across seeds.
    table = aggregate_table(results)
    table_path = plots_dir / "summary_table.csv"
    plots_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(table_path, index=False)
    print(f"Saved plots and table under {plots_dir}")


if __name__ == "__main__":
    main()

