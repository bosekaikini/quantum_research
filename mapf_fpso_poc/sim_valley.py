from __future__ import annotations

import random

from mapf_fpso_poc.sim import Agent, _add, _mul, _norm, _sub, apply_collision_penalty, compute_brightness, firefly_attraction
from mapf_fpso_poc.valley_env import ValleyWorld


def _near_goal_scale(dist_to_goal: float, goal_radius: float) -> float:
    """Taper swarm interactions to zero as the agent enters the goal neighborhood."""
    if goal_radius <= 0:
        return 1.0
    return float(min(1.0, max(0.0, (dist_to_goal - goal_radius) / goal_radius)))


def _dist_to_goal(agent: Agent) -> float:
    return _norm(_sub(agent.goal, agent.pos))


def _capture_at_goal(agent: Agent, goal_radius: float) -> bool:
    """If within capture radius, snap to goal exactly (distance 0) and stop."""
    if _dist_to_goal(agent) <= goal_radius:
        agent.pos = (float(agent.goal[0]), float(agent.goal[1]))
        agent.vel = (0.0, 0.0)
        return True
    return False


def default_goal_radius(env: ValleyWorld) -> float:
    """Capture radius ~ one grid cell (agents count as 'at goal' for MAPF POC)."""
    dx, dy = env.cell_size()
    return float(max(dx, dy) * 0.55)


def step_swarm_valley(
    env: ValleyWorld,
    agents: list[Agent],
    rng: random.Random,
    arrived: set[int],
    goal_radius: float = 0.05,
    omega: float = 0.6,
    c_pbest: float = 0.9,
    c_goal: float = 1.0,
    beta: float = 0.45,
    gamma: float = 8.0,
    alpha_brightness: float = 12.0,
    noise: float = 0.02,
    v_max: float = 0.05,
    r_min: float = 0.06,
    lambda_c: float = 12.0,
    repulse_gain: float = 0.02,
    valley_gain: float = 1.0,
) -> dict[str, float]:
    """Firefly/FPSO step with continuous valley obstacles (no grid projection).

    Agents within ``goal_radius`` of their goal stop moving (no lap/orbit).
    Peer repulsion/attraction taper near the goal to avoid clustering rings.
    """
    positions = [a.pos for a in agents]

    raw = []
    for i, a in enumerate(agents):
        if i not in arrived and _capture_at_goal(a, goal_radius):
            arrived.add(i)

        b = compute_brightness(a.pos, a.goal, env.is_free_pos(a.pos), alpha=alpha_brightness)
        if i not in arrived:
            others = [positions[j] for j in range(len(agents)) if j != i and j not in arrived]
            b = apply_collision_penalty(b, a.pos, others, r_min=r_min, lambda_c=lambda_c)
        a.brightness = b
        raw.append(b)
        if b > a.pbest_brightness:
            a.pbest_brightness = b
            a.pbest_pos = a.pos

    for i, a in enumerate(agents):
        if i in arrived:
            continue

        dist_goal = _dist_to_goal(a)
        near_scale = _near_goal_scale(dist_goal, goal_radius)

        brighter = [
            agents[j]
            for j in range(len(agents))
            if j not in arrived and agents[j].brightness > a.brightness
        ]

        attract = (0.0, 0.0)
        if brighter and near_scale > 0:
            brighter_sorted = sorted(brighter, key=lambda x: x.brightness, reverse=True)
            for peer in brighter_sorted[: min(4, len(brighter_sorted))]:
                attract = _add(
                    attract,
                    _mul(near_scale, firefly_attraction(a.pos, peer.pos, beta=beta, gamma=gamma)),
                )

        r1 = rng.random()
        pbest_pull = _mul(c_pbest * r1 * near_scale, _sub(a.pbest_pos, a.pos))
        goal_pull = _mul(c_goal, _sub(a.goal, a.pos))
        valley_pull = _mul(valley_gain * near_scale, env.valley_pull(a.pos))

        repulse = (0.0, 0.0)
        for j, op in enumerate(positions):
            if j == i or j in arrived:
                continue
            dvec = _sub(a.pos, op)
            d = _norm(dvec)
            if 1e-9 < d < r_min:
                repulse = _add(
                    repulse,
                    _mul(
                        near_scale * repulse_gain * (r_min - d) / (r_min * d),
                        dvec,
                    ),
                )

        noise_vec = (
            near_scale * noise * rng.gauss(0.0, 1.0),
            near_scale * noise * rng.gauss(0.0, 1.0),
        )
        v = _add(
            _add(_add(_add(_mul(omega * near_scale, a.vel), goal_pull), pbest_pull), attract),
            _add(_add(repulse, valley_pull), noise_vec),
        )

        speed = _norm(v)
        if speed > v_max:
            v = _mul(v_max / speed, v)
        a.vel = v

    for i, a in enumerate(agents):
        if i in arrived:
            continue
        a.pos = env.clamp_pos(_add(a.pos, a.vel))
        if _capture_at_goal(a, goal_radius):
            arrived.add(i)

    goal_dists = [_dist_to_goal(a) for a in agents]
    mean_goal_dist = float(sum(goal_dists) / max(1, len(goal_dists)))
    mean_brightness = float(sum(a.brightness for a in agents) / max(1, len(agents)))
    return {
        "mean_goal_dist": mean_goal_dist,
        "mean_brightness": mean_brightness,
        "min_brightness": float(min(raw) if raw else 0.0),
    }


def run_episode_valley(
    env: ValleyWorld,
    starts: list[tuple[float, float]],
    goals: list[tuple[float, float]],
    steps: int,
    seed: int = 0,
    stop_at_goal_radius: float | None = None,
    snap_all_goals_at_end: bool = True,
    **step_kwargs,
) -> dict[str, object]:
    rng = random.Random(seed)
    arrived: set[int] = set()
    goal_radius = default_goal_radius(env) if stop_at_goal_radius is None else float(stop_at_goal_radius)
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
    for _t in range(steps):
        s = step_swarm_valley(
            env,
            agents,
            rng,
            arrived,
            goal_radius=goal_radius,
            **step_kwargs,
        )
        traj.append([a.pos for a in agents])
        stats.append(s)

        if goal_radius > 0 and len(arrived) == len(agents):
            break

    if snap_all_goals_at_end:
        for i, a in enumerate(agents):
            a.pos = (float(a.goal[0]), float(a.goal[1]))
            a.vel = (0.0, 0.0)
            arrived.add(i)
        if traj:
            traj[-1] = [a.pos for a in agents]

    final_dists = [0.0] * len(agents) if snap_all_goals_at_end else [_dist_to_goal(a) for a in agents]
    return {
        "agents": agents,
        "traj": traj,
        "stats": stats,
        "final_goal_dists": final_dists,
        "steps_ran": int(len(traj) - 1),
    }
