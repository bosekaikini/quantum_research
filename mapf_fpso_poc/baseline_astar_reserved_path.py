from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Sequence

from mapf_fpso_poc.env import GridWorld
from mapf_fpso_poc import astar, discrete


@dataclass
class AStarReservedAgent:
    pos: tuple[float, float]
    start: tuple[float, float]
    goal: tuple[float, float]
    trail: set[tuple[int, int]] = field(default_factory=set)


def _norm(v: Sequence[float]) -> float:
    return math.sqrt(float(v[0]) * float(v[0]) + float(v[1]) * float(v[1]))


def _sub(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    return (float(a[0] - b[0]), float(a[1] - b[1]))


def _others_trails(agents: list[AStarReservedAgent], agent_index: int) -> set[tuple[int, int]]:
    blocked: set[tuple[int, int]] = set()
    for j, other in enumerate(agents):
        if j != agent_index:
            blocked |= other.trail
    return blocked


def _agent_priority(env: GridWorld, agent: AStarReservedAgent) -> tuple[int, int]:
    dist = discrete.cell_manhattan(
        discrete.pos_to_cell(env, agent.pos),
        discrete.pos_to_cell(env, agent.goal),
    )
    return (-dist, discrete.pos_to_cell(env, agent.pos))


def step_baseline_astar_reserved_path(
    env: GridWorld,
    agents: list[AStarReservedAgent],
    rng: random.Random,
) -> dict[str, float]:
    """Prioritized A* with foreign trails treated as blocked cells.

    Combines ``baseline_astar`` replanning with ``baseline_reserved_path`` trail
    reservations: other agents' visited cells cannot be entered (except goal).
    """
    del rng

    current_cells = [discrete.pos_to_cell(env, a.pos) for a in agents]
    order = sorted(range(len(agents)), key=lambda i: _agent_priority(env, agents[i]))

    proposed: list[tuple[int, int] | None] = [None] * len(agents)
    reserved_this_step: set[tuple[int, int]] = set()

    for i in order:
        a = agents[i]
        current = current_cells[i]
        goal = discrete.pos_to_cell(env, a.goal)

        if current == goal:
            proposed[i] = current
            reserved_this_step.add(current)
            continue

        others_now = {current_cells[j] for j in range(len(agents)) if j != i}
        foreign_trails = _others_trails(agents, i)
        blocked = others_now | reserved_this_step | foreign_trails
        blocked.discard(current)
        blocked.discard(goal)

        path = astar.astar_path(env, current, goal, blocked=blocked)
        if path is None or len(path) < 2:
            proposed[i] = current
        else:
            proposed[i] = path[1]

        reserved_this_step.add(proposed[i])

    proposed_cells = [p if p is not None else current_cells[i] for i, p in enumerate(proposed)]
    final_cells = discrete.resolve_simultaneous_moves(env, current_cells, proposed_cells)

    for a, cell in zip(agents, final_cells):
        a.pos = discrete.cell_to_pos(env, cell)
        a.trail.add(cell)

    mean_goal_dist = float(sum(_norm(_sub(a.goal, a.pos)) for a in agents) / max(1, len(agents)))
    return {"mean_goal_dist": mean_goal_dist}


def run_episode_baseline_astar_reserved_path(
    env: GridWorld,
    starts: list[tuple[float, float]],
    goals: list[tuple[float, float]],
    steps: int,
    seed: int = 0,
    **step_kwargs,
) -> dict[str, object]:
    rng = random.Random(seed)
    agents = [
        AStarReservedAgent(
            pos=tuple(s),
            start=tuple(s),
            goal=tuple(g),
            trail={discrete.pos_to_cell(env, s)},
        )
        for s, g in zip(starts, goals)
    ]

    traj: list[list[tuple[float, float]]] = []
    traj.append([a.pos for a in agents])

    stats = []
    for t in range(int(steps)):
        s = step_baseline_astar_reserved_path(env, agents, rng, **step_kwargs)
        traj.append([a.pos for a in agents])
        stats.append(s)

        if all(discrete.at_goal_cell(env, a.pos, a.goal) for a in agents):
            traj = traj[: t + 2]
            break

    final_dists = [_norm(_sub(a.goal, a.pos)) for a in agents]
    reserved_cells = set()
    for a in agents:
        reserved_cells |= a.trail

    return {
        "agents": agents,
        "traj": traj,
        "stats": stats,
        "final_goal_dists": final_dists,
        "steps_ran": int(len(traj) - 1),
        "reserved_cell_count": len(reserved_cells),
    }
