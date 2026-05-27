from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Sequence

from mapf_fpso_poc.env import GridWorld


@dataclass
class BaselineAgent:
    pos: tuple[float, float]
    vel: tuple[float, float]
    start: tuple[float, float]
    goal: tuple[float, float]


def _add(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    return (float(a[0] + b[0]), float(a[1] + b[1]))


def _sub(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    return (float(a[0] - b[0]), float(a[1] - b[1]))


def _mul(k: float, v: Sequence[float]) -> tuple[float, float]:
    return (float(k * v[0]), float(k * v[1]))


def _norm(v: Sequence[float]) -> float:
    return math.sqrt(float(v[0]) * float(v[0]) + float(v[1]) * float(v[1]))


def step_baseline(
    env: GridWorld,
    agents: list[BaselineAgent],
    rng: random.Random,
    *,
    v_max: float = 0.05,
    r_min: float = 0.06,
    repulse_gain: float = 0.03,
    noise: float = 0.0,
) -> dict[str, float]:
    """Baseline controller for comparison.

    Behavior:
    - move directly toward goal (clipped by v_max)
    - project into free space if landing in obstacle
    - simple local repulsion within r_min

    This is intentionally simpler than the Firefly/FPSO update (no peer attraction,
    no brightness, no personal-best memory).
    """

    positions = [a.pos for a in agents]
    for i, a in enumerate(agents):
        to_goal = _sub(a.goal, a.pos)
        d = _norm(to_goal)
        if d > 1e-9:
            v_goal = _mul(min(v_max, d) / d, to_goal)
        else:
            v_goal = (0.0, 0.0)

        repulse = (0.0, 0.0)
        for j, op in enumerate(positions):
            if j == i:
                continue
            dvec = _sub(a.pos, op)
            dist = _norm(dvec)
            if 1e-9 < dist < r_min:
                repulse = _add(
                    repulse,
                    _mul(repulse_gain * (r_min - dist) / (r_min * dist), dvec),
                )

        noise_vec = (noise * rng.gauss(0.0, 1.0), noise * rng.gauss(0.0, 1.0)) if noise > 0 else (0.0, 0.0)
        v = _add(_add(v_goal, repulse), noise_vec)

        speed = _norm(v)
        if speed > v_max:
            v = _mul(v_max / speed, v)

        a.vel = v

    for a in agents:
        a.pos = env.project_to_free(_add(a.pos, a.vel))

    mean_goal_dist = float(sum(_norm(_sub(a.goal, a.pos)) for a in agents) / max(1, len(agents)))
    return {"mean_goal_dist": mean_goal_dist}


def run_episode_baseline(
    env: GridWorld,
    starts: list[tuple[float, float]],
    goals: list[tuple[float, float]],
    steps: int,
    seed: int = 0,
    stop_at_goal_radius: float = 0.03,
    **step_kwargs,
) -> dict[str, object]:
    rng = random.Random(seed)
    agents = [
        BaselineAgent(pos=tuple(s), vel=(0.0, 0.0), start=tuple(s), goal=tuple(g))
        for s, g in zip(starts, goals)
    ]

    traj: list[list[tuple[float, float]]] = []
    traj.append([a.pos for a in agents])

    stats = []
    for t in range(int(steps)):
        s = step_baseline(env, agents, rng, **step_kwargs)
        traj.append([a.pos for a in agents])
        stats.append(s)

        if stop_at_goal_radius > 0:
            if all(_norm(_sub(a.goal, a.pos)) <= stop_at_goal_radius for a in agents):
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

