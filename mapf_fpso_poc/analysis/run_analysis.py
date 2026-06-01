from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Callable, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mapf_fpso_poc.env import GridWorld
from mapf_fpso_poc.baseline_astar import run_episode_baseline_astar
from mapf_fpso_poc.baseline_astar_reserved_path import run_episode_baseline_astar_reserved_path
from mapf_fpso_poc.sim import run_episode
from mapf_fpso_poc.sim_valley import run_episode_valley
from mapf_fpso_poc.valley_env import ValleyWorld


ROOT_DIR = Path(__file__).resolve().parents[2]


def next_numbered_path(directory: Path, stem: str, suffix: str) -> Path:
    """Return ``directory/stem_N.suffix`` where N is one higher than any existing file."""
    directory.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(rf"^{re.escape(stem)}_(\d+){re.escape(suffix)}$")
    max_n = 0
    if (directory / f"{stem}{suffix}").exists():
        max_n = 1
    for path in directory.iterdir():
        match = pattern.match(path.name)
        if match:
            max_n = max(max_n, int(match.group(1)) + 1)
    if max_n == 0:
        max_n = 1
    return directory / f"{stem}_{max_n}{suffix}"


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


def _build_series(
    left_traj: List[List[Tuple[float, float]]],
    firefly_traj: List[List[Tuple[float, float]]],
    goals: List[Tuple[float, float]],
    r_min: float,
    prefix: str,
) -> dict[str, list[float]]:
    left_mean_d, left_max_d = _goal_dist_series(left_traj, goals)
    ff_mean_d, ff_max_d = _goal_dist_series(firefly_traj, goals)
    left_col = _pairwise_collision_counts(left_traj, r_min=r_min)
    ff_col = _pairwise_collision_counts(firefly_traj, r_min=r_min)
    return {
        f"{prefix}_mean_dist": left_mean_d,
        f"{prefix}_max_dist": left_max_d,
        f"{prefix}_collisions": left_col,
        "firefly_mean_dist": ff_mean_d,
        "firefly_max_dist": ff_max_d,
        "firefly_collisions": ff_col,
    }


def _run_single_scenario(
    seed: int,
    run_left: Callable,
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

    left = run_left(env, starts, goals, steps=steps, seed=seed)
    firefly = run_episode(env, starts, goals, steps=steps, seed=seed)
    return {
        "seed": seed,
        "env": env,
        "starts": starts,
        "goals": goals,
        "left": left,
        "firefly": firefly,
        "valley_env": None,
    }


def _run_astar_vs_valley_firefly_scenario(
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
    valley_env = ValleyWorld.from_grid(env)
    pts = env.sample_free_positions(num_agents * 2, rng=rng)
    starts = pts[:num_agents]
    goals = pts[num_agents:]

    astar = run_episode_baseline_astar(env, starts, goals, steps=steps, seed=seed)
    firefly = run_episode_valley(valley_env, starts, goals, steps=steps, seed=seed)
    return {
        "seed": seed,
        "env": env,
        "valley_env": valley_env,
        "starts": starts,
        "goals": goals,
        "left": astar,
        "firefly": firefly,
    }


def plot_trajectories(
    env: GridWorld,
    starts,
    goals,
    left_traj,
    firefly_traj,
    out_path: Path,
    *,
    left_title: str,
    valley_env: ValleyWorld | None = None,
    right_title: str = "Firefly/FPSO (continuous swarm)",
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    panels = [
        (axes[0], left_traj, left_title, False),
        (axes[1], firefly_traj, right_title, True),
    ]

    for ax, traj, title, is_valley_panel in panels:
        ax.set_title(title)
        ax.set_xlim(env.bounds.xmin, env.bounds.xmax)
        ax.set_ylim(env.bounds.ymin, env.bounds.ymax)
        ax.set_aspect("equal", adjustable="box")

        if is_valley_panel and valley_env is not None:
            xs, ys, depth = valley_env.sample_depth_grid(resolution=50)
            ax.contourf(xs, ys, depth, levels=14, cmap="YlOrBr", alpha=0.5)
        else:
            dx, dy = env.cell_size()
            shrink = 0.85
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
            ax.scatter(xs, ys, color=c, s=10, alpha=0.9)
            ax.scatter([s[0]], [s[1]], color=c, marker="o", s=40, edgecolor="white", linewidth=0.5, zorder=5)
            ax.scatter([g[0]], [g[1]], color=c, marker="*", s=80, edgecolor="white", linewidth=0.5, zorder=5)

        ax.grid(True, linestyle=":", alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_time_series(
    series: dict,
    out_path: Path,
    *,
    prefix: str,
    left_label: str,
    firefly_label: str = "Firefly",
) -> None:
    n_steps = max(len(series[f"{prefix}_mean_dist"]), len(series["firefly_mean_dist"]))
    t = np.arange(n_steps)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1, ax2 = axes
    ax1.plot(
        t[: len(series[f"{prefix}_max_dist"])],
        series[f"{prefix}_max_dist"],
        label=f"{left_label} max dist",
        color="C0",
    )
    ax1.plot(
        t[: len(series["firefly_max_dist"])],
        series["firefly_max_dist"],
        label=f"{firefly_label} max dist",
        color="C1",
    )
    ax1.set_ylabel("Max distance to goal")
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle=":", alpha=0.3)

    ax2.plot(
        t[: len(series[f"{prefix}_collisions"])],
        series[f"{prefix}_collisions"],
        label=f"{left_label} near-collisions",
        color="C0",
    )
    ax2.plot(
        t[: len(series["firefly_collisions"])],
        series["firefly_collisions"],
        label=f"{firefly_label} near-collisions",
        color="C1",
    )
    ax2.set_ylabel("Pairs with d < r_min")
    ax2.set_xlabel("Step")
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle=":", alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def aggregate_table(results: Iterable[dict], prefix: str) -> pd.DataFrame:
    rows = []
    for r in results:
        left = r["left"]
        ff = r["firefly"]
        left_d = left["final_goal_dists"]
        ff_d = ff["final_goal_dists"]
        row = {
            "seed": r["seed"],
            f"{prefix}_steps": left["steps_ran"],
            f"{prefix}_mean_dist": float(sum(left_d) / max(1, len(left_d))),
            f"{prefix}_max_dist": float(max(left_d) if left_d else 0.0),
            "firefly_steps": ff["steps_ran"],
            "firefly_mean_dist": float(sum(ff_d) / max(1, len(ff_d))),
            "firefly_max_dist": float(max(ff_d) if ff_d else 0.0),
        }
        if "reserved_cell_count" in left:
            row[f"{prefix}_reserved_cells"] = left["reserved_cell_count"]
        rows.append(row)
    return pd.DataFrame(rows)


def _generate_comparison(
    results: list[dict],
    first: dict,
    goals: list[Tuple[float, float]],
    plots_dir: Path,
    *,
    file_stem: str,
    prefix: str,
    left_title: str,
    left_label: str,
    r_min: float,
    valley_env: ValleyWorld | None = None,
    right_title: str = "Firefly/FPSO (continuous swarm)",
    firefly_label: str = "Firefly",
) -> tuple[Path, Path, Path]:
    series = _build_series(first["left"]["traj"], first["firefly"]["traj"], goals, r_min, prefix)

    traj_path = next_numbered_path(plots_dir, f"trajectories_{file_stem}", ".png")
    series_path = next_numbered_path(plots_dir, f"distances_and_collisions_{file_stem}", ".png")
    table_path = next_numbered_path(plots_dir, f"summary_table_{file_stem}", ".csv")

    plot_trajectories(
        first["env"],
        first["starts"],
        first["goals"],
        first["left"]["traj"],
        first["firefly"]["traj"],
        traj_path,
        left_title=left_title,
        valley_env=valley_env or first.get("valley_env"),
        right_title=right_title,
    )
    plot_time_series(
        series,
        series_path,
        prefix=prefix,
        left_label=left_label,
        firefly_label=firefly_label,
    )
    aggregate_table(results, prefix).to_csv(table_path, index=False)
    return traj_path, series_path, table_path


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
    parser.add_argument(
        "--only",
        choices=("astar", "astar_reserved", "astar_vs_valley", "both", "all"),
        default="astar_vs_valley",
        help="Which comparisons to generate (astar_vs_valley = A* grid vs Firefly valley obstacles).",
    )
    args = parser.parse_args()

    scenario_kwargs = dict(
        num_agents=args.agents,
        width=args.width,
        height=args.height,
        obstacle_density=args.obstacle_density,
        steps=args.steps,
        r_min=args.r_min,
    )

    plots_dir = ROOT_DIR / "plots" / "mapf_fpso_poc"
    saved: list[str] = []

    if args.only in ("astar", "both", "all"):
        astar_results = [
            _run_single_scenario(seed, run_episode_baseline_astar, **scenario_kwargs)
            for seed in range(args.seeds)
        ]
        paths = _generate_comparison(
            astar_results,
            astar_results[0],
            astar_results[0]["goals"],
            plots_dir,
            file_stem="astar_vs_firefly",
            prefix="astar",
            left_title="A* baseline (prioritized, discrete)",
            left_label="A*",
            r_min=args.r_min,
        )
        saved.extend(p.name for p in paths)

    if args.only in ("astar_reserved", "both", "all"):
        reserved_results = [
            _run_single_scenario(seed, run_episode_baseline_astar_reserved_path, **scenario_kwargs)
            for seed in range(args.seeds)
        ]
        paths = _generate_comparison(
            reserved_results,
            reserved_results[0],
            reserved_results[0]["goals"],
            plots_dir,
            file_stem="astar_reserved_vs_firefly",
            prefix="astar_reserved",
            left_title="A* + reserved paths (discrete)",
            left_label="A* reserved",
            r_min=args.r_min,
        )
        saved.extend(p.name for p in paths)

    if args.only in ("astar_vs_valley", "all"):
        valley_results = [
            _run_astar_vs_valley_firefly_scenario(seed, **scenario_kwargs)
            for seed in range(args.seeds)
        ]
        paths = _generate_comparison(
            valley_results,
            valley_results[0],
            valley_results[0]["goals"],
            plots_dir,
            file_stem="astar_vs_firefly_valleys",
            prefix="astar",
            left_title="A* (discrete grid obstacles)",
            left_label="A*",
            r_min=args.r_min,
            valley_env=valley_results[0]["valley_env"],
            right_title="Firefly/FPSO (continuous valley obstacles)",
            firefly_label="Firefly valleys",
        )
        saved.extend(p.name for p in paths)

    print(f"Saved: {', '.join(saved)}")
    print(f"Directory: {plots_dir}")


if __name__ == "__main__":
    main()
