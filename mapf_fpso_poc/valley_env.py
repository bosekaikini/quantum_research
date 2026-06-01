from __future__ import annotations

import math
from typing import Iterable, Sequence

from mapf_fpso_poc.env import Bounds, GridWorld


class ValleyWorld:
    """Continuous obstacle field: Gaussian valleys agents fall into.

    Valley centers are placed at grid obstacle cell centers (same layout as A*).
    There are no hard grid projections — agents feel a smooth pull toward valley
    bottoms and lose brightness when deep inside a valley.
    """

    def __init__(
        self,
        grid: GridWorld,
        valley_centers: list[tuple[float, float]],
        sigma: float,
        pull_strength: float,
        depth_threshold: float = 0.72,
    ):
        self.grid = grid
        self.bounds = grid.bounds
        self.valley_centers = list(valley_centers)
        self.sigma = float(max(sigma, 1e-6))
        self.pull_strength = float(pull_strength)
        self.depth_threshold = float(depth_threshold)
        self._inv_2sigma2 = 1.0 / (2.0 * self.sigma * self.sigma)

    @classmethod
    def from_grid(
        cls,
        grid: GridWorld,
        *,
        sigma_scale: float = 1.35,
        pull_strength: float = 0.14,
        depth_threshold: float = 0.72,
    ) -> ValleyWorld:
        dx, dy = grid.cell_size()
        sigma = max(dx, dy) * sigma_scale
        centers = list(grid.iter_obstacle_centers())
        return cls(
            grid,
            centers,
            sigma=sigma,
            pull_strength=pull_strength,
            depth_threshold=depth_threshold,
        )

    def cell_size(self) -> tuple[float, float]:
        return self.grid.cell_size()

    def iter_obstacle_centers(self) -> Iterable[tuple[float, float]]:
        yield from self.grid.iter_obstacle_centers()

    def valley_depth(self, pos: Sequence[float]) -> float:
        """0 on rims, approaches 1 at valley centers (deepest)."""
        x, y = float(pos[0]), float(pos[1])
        depth = 0.0
        for cx, cy in self.valley_centers:
            dx = x - cx
            dy = y - cy
            depth = max(depth, math.exp(-(dx * dx + dy * dy) * self._inv_2sigma2))
        return float(depth)

    def valley_pull(self, pos: Sequence[float]) -> tuple[float, float]:
        """Attractive force pulling agents into nearby valleys."""
        x, y = float(pos[0]), float(pos[1])
        fx = 0.0
        fy = 0.0
        for cx, cy in self.valley_centers:
            dx = cx - x
            dy = cy - y
            w = math.exp(-(dx * dx + dy * dy) * self._inv_2sigma2)
            fx += self.pull_strength * dx * w
            fy += self.pull_strength * dy * w
        return (float(fx), float(fy))

    def is_free_pos(self, pos: Sequence[float]) -> bool:
        return self.valley_depth(pos) < self.depth_threshold

    def clamp_pos(self, pos: Sequence[float]) -> tuple[float, float]:
        return self.bounds.clamp(pos)

    def sample_depth_grid(self, resolution: int = 48) -> tuple[list[float], list[float], list[list[float]]]:
        """Grid of depth values for contour plotting."""
        xs = [
            self.bounds.xmin + (self.bounds.xmax - self.bounds.xmin) * i / (resolution - 1)
            for i in range(resolution)
        ]
        ys = [
            self.bounds.ymin + (self.bounds.ymax - self.bounds.ymin) * j / (resolution - 1)
            for j in range(resolution)
        ]
        z = [[self.valley_depth((x, y)) for x in xs] for y in ys]
        return xs, ys, z
