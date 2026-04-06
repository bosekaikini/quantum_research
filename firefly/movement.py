"""Movement logic for the Firefly Algorithm."""

from __future__ import annotations

import math
import random
from typing import Any


def euclidean_distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
	"""Return the Euclidean distance between two 2D points."""
	dx = p2[0] - p1[0]
	dy = p2[1] - p1[1]
	return math.sqrt(dx * dx + dy * dy)


def move_firefly(
	firefly: Any,
	other_firefly: Any,
	alpha: float,
	beta: float,
	gamma: float,
) -> tuple[float, float]:
	"""Move a dimmer firefly toward a brighter firefly and return new coordinates.

	The function is pure with respect to the firefly objects: it returns a new
	position and does not modify either input object.
	"""
	current_pos = (float(firefly.position[0]), float(firefly.position[1]))

	# Only move if the other firefly is brighter.
	if other_firefly.brightness <= firefly.brightness:
		return current_pos

	other_pos = (float(other_firefly.position[0]), float(other_firefly.position[1]))

	# Compute pairwise distance and attractiveness decay.
	distance = euclidean_distance(current_pos, other_pos)
	beta_t = beta * math.exp(-gamma * (distance ** 2))

	# Independent uniform random perturbation in each dimension.
	noise_x = random.uniform(-0.5, 0.5)
	noise_y = random.uniform(-0.5, 0.5)

	# Move toward brighter firefly plus randomized exploration.
	new_x = current_pos[0] + beta_t * (other_pos[0] - current_pos[0]) + alpha * noise_x
	new_y = current_pos[1] + beta_t * (other_pos[1] - current_pos[1]) + alpha * noise_y

	# Keep coordinates within valid bounds.
	new_x = min(1.0, max(0.0, new_x))
	new_y = min(1.0, max(0.0, new_y))

	return (new_x, new_y)
