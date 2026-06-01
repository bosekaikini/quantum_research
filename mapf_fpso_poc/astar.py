from __future__ import annotations

import heapq
from typing import Sequence

from mapf_fpso_poc.env import GridWorld
from mapf_fpso_poc.discrete import cell_manhattan, free_neighbors, pos_to_cell


def astar_path(
    env: GridWorld,
    start: tuple[int, int],
    goal: tuple[int, int],
    blocked: set[tuple[int, int]] | None = None,
) -> list[tuple[int, int]] | None:
    """Shortest 4-connected path on the grid (unit cost), or None if unreachable.

    ``blocked`` cells cannot be entered except ``start`` and ``goal`` (always allowed).
    """
    blocked = blocked or set()
    if start == goal:
        return [start]

    if goal in blocked and goal != start:
        return None

    def heuristic(cell: tuple[int, int]) -> int:
        return cell_manhattan(cell, goal)

    open_heap: list[tuple[int, int, tuple[int, int]]] = []
    heapq.heappush(open_heap, (heuristic(start), 0, start))
    g_score: dict[tuple[int, int], int] = {start: 0}
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    closed: set[tuple[int, int]] = set()

    while open_heap:
        _, g_val, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        closed.add(current)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        for nbr in free_neighbors(env, current):
            if nbr in blocked and nbr not in (start, goal):
                continue
            if nbr in closed:
                continue
            tentative = g_val + 1
            if tentative < g_score.get(nbr, 10**9):
                g_score[nbr] = tentative
                came_from[nbr] = current
                heapq.heappush(open_heap, (tentative + heuristic(nbr), tentative, nbr))

    return None


def astar_path_from_pos(
    env: GridWorld,
    start_pos: Sequence[float],
    goal_pos: Sequence[float],
    blocked: set[tuple[int, int]] | None = None,
) -> list[tuple[int, int]] | None:
    return astar_path(
        env,
        pos_to_cell(env, start_pos),
        pos_to_cell(env, goal_pos),
        blocked=blocked,
    )
