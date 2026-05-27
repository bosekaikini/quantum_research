from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Bounds:
    xmin: float = 0.0
    xmax: float = 1.0
    ymin: float = 0.0
    ymax: float = 1.0

    def clamp(self, pos: Sequence[float]) -> tuple[float, float]:
        x = float(min(self.xmax, max(self.xmin, float(pos[0]))))
        y = float(min(self.ymax, max(self.ymin, float(pos[1]))))
        return (x, y)


class GridWorld:
    """Grid obstacle map embedded into continuous [0,1]x[0,1] space.

    Obstacles live on a boolean grid. Continuous positions are mapped to cells.
    Projection to free space is implemented by searching a growing ring of cells
    around the current cell and snapping to the nearest free cell center.
    """

    def __init__(self, width: int, height: int, obstacles: list[list[bool]] | None = None, bounds: Bounds | None = None):
        if width <= 1 or height <= 1:
            raise ValueError("width and height must be >= 2")
        self.width = int(width)
        self.height = int(height)
        self.bounds = bounds or Bounds()

        if obstacles is None:
            self.obstacles = [[False for _ in range(self.width)] for __ in range(self.height)]
        else:
            if len(obstacles) != self.height or any(len(row) != self.width for row in obstacles):
                raise ValueError(f"obstacles must have shape {(self.height, self.width)}")
            self.obstacles = [[bool(v) for v in row] for row in obstacles]

    def cell_size(self) -> tuple[float, float]:
        return ((self.bounds.xmax - self.bounds.xmin) / self.width, (self.bounds.ymax - self.bounds.ymin) / self.height)

    def pos_to_cell(self, pos: Sequence[float]) -> tuple[int, int]:
        x, y = self.bounds.clamp(pos)
        dx, dy = self.cell_size()
        cx = int(math.floor((x - self.bounds.xmin) / dx))
        cy = int(math.floor((y - self.bounds.ymin) / dy))
        cx = max(0, min(self.width - 1, cx))
        cy = max(0, min(self.height - 1, cy))
        return (cy, cx)

    def cell_center(self, cy: int, cx: int) -> tuple[float, float]:
        dx, dy = self.cell_size()
        x = self.bounds.xmin + (cx + 0.5) * dx
        y = self.bounds.ymin + (cy + 0.5) * dy
        return (float(x), float(y))

    def is_free_pos(self, pos: Sequence[float]) -> bool:
        cy, cx = self.pos_to_cell(pos)
        return not bool(self.obstacles[cy][cx])

    def project_to_free(self, pos: Sequence[float], max_ring: int = 12) -> tuple[float, float]:
        p = self.bounds.clamp(pos)
        cy, cx = self.pos_to_cell(p)
        if not self.obstacles[cy][cx]:
            return p

        best: tuple[float, float] | None = None
        best_d2 = float("inf")
        for r in range(1, max_ring + 1):
            for ny in range(max(0, cy - r), min(self.height, cy + r + 1)):
                for nx in range(max(0, cx - r), min(self.width, cx + r + 1)):
                    if abs(ny - cy) != r and abs(nx - cx) != r:
                        continue
                    if self.obstacles[ny][nx]:
                        continue
                    candidate = self.cell_center(ny, nx)
                    dxp = float(candidate[0] - p[0])
                    dyp = float(candidate[1] - p[1])
                    d2 = dxp * dxp + dyp * dyp
                    if d2 < best_d2:
                        best = candidate
                        best_d2 = d2
            if best is not None:
                break

        return best if best is not None else p

    @staticmethod
    def random_obstacles(width: int, height: int, density: float, rng: random.Random) -> list[list[bool]]:
        density = float(min(0.8, max(0.0, density)))
        return [[(rng.random() < density) for _ in range(int(width))] for __ in range(int(height))]

    def sample_free_positions(self, k: int, rng: random.Random) -> list[tuple[float, float]]:
        free_cells: list[tuple[int, int]] = []
        for cy in range(self.height):
            for cx in range(self.width):
                if not self.obstacles[cy][cx]:
                    free_cells.append((cy, cx))
        if not free_cells:
            raise ValueError("No free cells available to sample from")
        pts: list[tuple[float, float]] = []
        for _ in range(int(k)):
            cy, cx = free_cells[rng.randrange(0, len(free_cells))]
            pts.append(self.cell_center(cy, cx))
        return pts

    def iter_obstacle_centers(self) -> Iterable[tuple[float, float]]:
        for cy in range(self.height):
            for cx in range(self.width):
                if self.obstacles[cy][cx]:
                    yield self.cell_center(cy, cx)

