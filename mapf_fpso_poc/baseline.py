from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Sequence

from mapf_fpso_poc.env import GridWorld
from mapf_fpso_poc import discrete


@dataclass
class BaselineAgent:
    pos: tuple[float, float]
    start: tuple[float, float]
    goal: tuple[float, float]


def _norm(v: Sequence[float]) -> float:
    return math.sqrt(float(v[0]) * float(v[0]) + float(v[1]) * float(v[1]))


def _sub(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    return (float(a[0] - b[0]), float(a[1] - b[1]))


def step_baseline(
    env: GridWorld,
    agents: list[BaselineAgent],
    rng: random.Random,
) -> dict[str, float]:
    """Discrete MAPF-style baseline: one grid move per agent per timestep.

    Each agent greedily picks a 4-connected neighbor (or wait) that reduces
    Manhattan distance to its goal, with vertex/swap conflict resolution.
    """
    del rng  # reserved for future tie-breaking

    current_cells = [discrete.pos_to_cell(env, a.pos) for a in agents]
    proposed: list[tuple[int, int]] = []

    for i, a in enumerate(agents):
        occ = {current_cells[j] for j in range(len(agents)) if j != i}
        proposed.append(discrete.greedy_next_cell(env, a.pos, a.goal, blocked=occ))

    final_cells = discrete.resolve_simultaneous_moves(env, current_cells, proposed)

    for a, cell in zip(agents, final_cells):
        a.pos = discrete.cell_to_pos(env, cell)

    mean_goal_dist = float(sum(_norm(_sub(a.goal, a.pos)) for a in agents) / max(1, len(agents)))
    return {"mean_goal_dist": mean_goal_dist}


def run_episode_baseline(
    env: GridWorld,
    starts: list[tuple[float, float]],
    goals: list[tuple[float, float]],
    steps: int,
    seed: int = 0,
    **step_kwargs,
) -> dict[str, object]:
    rng = random.Random(seed)
    agents = [
        BaselineAgent(pos=tuple(s), start=tuple(s), goal=tuple(g))
        for s, g in zip(starts, goals)
    ]

    traj: list[list[tuple[float, float]]] = []
    traj.append([a.pos for a in agents])

    stats = []
    for t in range(int(steps)):
        s = step_baseline(env, agents, rng, **step_kwargs)
        traj.append([a.pos for a in agents])
        stats.append(s)

        if all(discrete.at_goal_cell(env, a.pos, a.goal) for a in agents):
            traj = traj[: t + 2]
            break

    final_dists = [_norm(_sub(a.goal, a.pos)) for a in agents]
    return {
        "agents": agents,
        "traj": traj,
        "stats": stats,
        "final_goal_dists": final_dists,
        "steps_ran": int(len(traj) - 1),
    }
