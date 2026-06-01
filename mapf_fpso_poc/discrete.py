from __future__ import annotations

from typing import Iterable, Sequence

from mapf_fpso_poc.env import GridWorld

# 4-connected moves on (row=cy, col=cx): stay, N, S, W, E
DELTAS: tuple[tuple[int, int], ...] = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))


def cell_manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def pos_to_cell(env: GridWorld, pos: Sequence[float]) -> tuple[int, int]:
    return env.pos_to_cell(pos)


def cell_to_pos(env: GridWorld, cell: tuple[int, int]) -> tuple[float, float]:
    return env.cell_center(cell[0], cell[1])


def free_neighbors(env: GridWorld, cell: tuple[int, int]) -> list[tuple[int, int]]:
    cy, cx = cell
    out: list[tuple[int, int]] = []
    for dcy, dcx in DELTAS:
        ny, nx = cy + dcy, cx + dcx
        if 0 <= ny < env.height and 0 <= nx < env.width and not env.obstacles[ny][nx]:
            out.append((ny, nx))
    return out


def at_goal_cell(env: GridWorld, pos: Sequence[float], goal: Sequence[float]) -> bool:
    return pos_to_cell(env, pos) == pos_to_cell(env, goal)


def occupied_cells(env: GridWorld, positions: Iterable[Sequence[float]]) -> set[tuple[int, int]]:
    return {pos_to_cell(env, p) for p in positions}


def greedy_next_cell(
    env: GridWorld,
    pos: Sequence[float],
    goal: Sequence[float],
    blocked: set[tuple[int, int]],
) -> tuple[int, int]:
    """Pick free neighbor (or wait) that most reduces Manhattan distance to goal."""
    current = pos_to_cell(env, pos)
    target = pos_to_cell(env, goal)
    if current == target:
        return current

    candidates = free_neighbors(env, current)
    # Vertex constraint: cannot move into a cell occupied by another agent.
    candidates = [c for c in candidates if c not in blocked or c == current]
    if not candidates:
        return current
    return min(candidates, key=lambda c: cell_manhattan(c, target))


def resolve_simultaneous_moves(
    env: GridWorld,
    current_cells: list[tuple[int, int]],
    proposed_cells: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Resolve vertex conflicts and swap conflicts; conflicting agents wait."""
    n = len(current_cells)
    final = list(proposed_cells)

    # Vertex: two agents cannot end in the same cell.
    for i in range(n):
        if sum(1 for j in range(n) if final[j] == final[i]) > 1:
            final[i] = current_cells[i]

    # Edge swap: i: A->B and j: B->A at the same time.
    for i in range(n):
        for j in range(i + 1, n):
            if final[i] == current_cells[j] and final[j] == current_cells[i]:
                if current_cells[i] != current_cells[j]:
                    final[i] = current_cells[i]
                    final[j] = current_cells[j]

    return final
