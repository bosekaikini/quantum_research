from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Sequence

from mapf_fpso_poc.env import GridWorld


@dataclass
class Agent:
    pos: tuple[float, float]
    vel: tuple[float, float]
    start: tuple[float, float]
    goal: tuple[float, float]
    pbest_pos: tuple[float, float]
    pbest_brightness: float = 0.0
    brightness: float = 0.0


def _add(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    return (float(a[0] + b[0]), float(a[1] + b[1]))


def _sub(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    return (float(a[0] - b[0]), float(a[1] - b[1]))


def _mul(k: float, v: Sequence[float]) -> tuple[float, float]:
    return (float(k * v[0]), float(k * v[1]))


def _norm(v: Sequence[float]) -> float:
    return math.sqrt(float(v[0]) * float(v[0]) + float(v[1]) * float(v[1]))


def _norm2(v: Sequence[float]) -> float:
    return float(v[0]) * float(v[0]) + float(v[1]) * float(v[1])


def compute_brightness(
    pos: Sequence[float],
    goal: Sequence[float],
    is_free: bool,
    alpha: float,
) -> float:
    """Brightness from one-pager: exp(-alpha * ||x-g||^2) with obstacle mask."""
    if not is_free:
        return 0.0
    dx = float(pos[0] - goal[0])
    dy = float(pos[1] - goal[1])
    d2 = dx * dx + dy * dy
    return float(math.exp(-alpha * d2))


def apply_collision_penalty(
    brightness: float,
    pos: Sequence[float],
    other_positions: list[tuple[float, float]],
    r_min: float,
    lambda_c: float,
) -> float:
    if brightness <= 0.0 or not other_positions:
        return brightness
    penalty = 0.0
    for op in other_positions:
        d = _norm(_sub(pos, op))
        if d < r_min:
            penalty += (r_min - d) ** 2
    if penalty <= 0.0:
        return brightness
    return float(brightness * math.exp(-lambda_c * penalty))


def firefly_attraction(
    pos: Sequence[float],
    other_pos: Sequence[float],
    beta: float,
    gamma: float,
) -> tuple[float, float]:
    """Same kernel style as `firefly/movement.py`: beta*exp(-gamma*r^2)*(other-pos)."""
    delta = _sub(other_pos, pos)
    r2 = _norm2(delta)
    return _mul(float(beta * math.exp(-gamma * r2)), delta)


def step_swarm(
    env: GridWorld,
    agents: list[Agent],
    rng: random.Random,
    omega: float = 0.6,
    c_pbest: float = 0.9,
    c_goal: float = 0.8,
    beta: float = 0.45,
    gamma: float = 8.0,
    alpha_brightness: float = 12.0,
    noise: float = 0.02,
    v_max: float = 0.05,
    r_min: float = 0.06,
    lambda_c: float = 12.0,
    repulse_gain: float = 0.02,
) -> dict[str, float]:
    """One online timestep of a Firefly-style FPSO-MAPF hybrid.

    Not a full MAPF solver; this is a POC "online continuous swarm navigation"
    consistent with the one-pager's update pattern and your repo's firefly kernel.
    """

    positions = [a.pos for a in agents]

    # 1) Compute brightness with obstacle mask + collision penalty.
    raw = []
    for i, a in enumerate(agents):
        b = compute_brightness(a.pos, a.goal, env.is_free_pos(a.pos), alpha=alpha_brightness)
        others = [positions[j] for j in range(len(agents)) if j != i]
        b = apply_collision_penalty(b, a.pos, others, r_min=r_min, lambda_c=lambda_c)
        a.brightness = b
        raw.append(b)
        if b > a.pbest_brightness:
            a.pbest_brightness = b
            a.pbest_pos = a.pos

    # 2) Velocity update: inertia + goal pull + attraction to brighter peers + noise + repulsion.
    for i, a in enumerate(agents):
        brighter = [agents[j] for j in range(len(agents)) if agents[j].brightness > a.brightness]

        attract = (0.0, 0.0)
        if brighter:
            # Use a few best peers to reduce O(n^2) blowup in bigger demos.
            brighter_sorted = sorted(brighter, key=lambda x: x.brightness, reverse=True)
            for peer in brighter_sorted[: min(4, len(brighter_sorted))]:
                attract = _add(attract, firefly_attraction(a.pos, peer.pos, beta=beta, gamma=gamma))

        # FPSO-style "personal best" pull from the one-pager: c1 * r1 * (p_i - x_i).
        r1 = rng.random()
        pbest_pull = _mul(c_pbest * r1, _sub(a.pbest_pos, a.pos))

        goal_pull = _mul(c_goal, _sub(a.goal, a.pos))

        repulse = (0.0, 0.0)
        for j, op in enumerate(positions):
            if j == i:
                continue
            dvec = _sub(a.pos, op)
            d = _norm(dvec)
            if 1e-9 < d < r_min:
                repulse = _add(
                    repulse,
                    _mul(repulse_gain * (r_min - d) / (r_min * d), dvec),
                )

        noise_vec = (noise * rng.gauss(0.0, 1.0), noise * rng.gauss(0.0, 1.0))
        v = _add(
            _add(_add(_add(_mul(omega, a.vel), goal_pull), pbest_pull), attract),
            _add(repulse, noise_vec),
        )

        speed = _norm(v)
        if speed > v_max:
            v = _mul(v_max / speed, v)

        a.vel = v

    # 3) Position update with obstacle projection.
    for a in agents:
        a.pos = env.project_to_free(_add(a.pos, a.vel))

    goal_dists = [_norm(_sub(a.goal, a.pos)) for a in agents]
    mean_goal_dist = float(sum(goal_dists) / max(1, len(goal_dists)))
    mean_brightness = float(sum(a.brightness for a in agents) / max(1, len(agents)))
    return {"mean_goal_dist": mean_goal_dist, "mean_brightness": mean_brightness, "min_brightness": float(min(raw) if raw else 0.0)}


def run_episode(
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
        Agent(
            pos=tuple(s),
            vel=(0.0, 0.0),
            start=tuple(s),
            goal=tuple(g),
            pbest_pos=tuple(s),
        )
        for s, g in zip(starts, goals)
    ]

    traj: list[list[tuple[float, float]]] = []
    traj.append([a.pos for a in agents])

    stats = []
    for t in range(steps):
        s = step_swarm(env, agents, rng, **step_kwargs)
        traj.append([a.pos for a in agents])
        stats.append(s)

        if stop_at_goal_radius > 0:
            if all(_norm(_sub(a.goal, a.pos)) <= stop_at_goal_radius for a in agents):
                traj = traj[: t + 2]  # already includes initial state
                break

    final_dists = [_norm(_sub(a.goal, a.pos)) for a in agents]
    return {
        "agents": agents,
        "traj": traj,
        "stats": stats,
        "final_goal_dists": final_dists,
        "steps_ran": int(len(traj) - 1),
    }

