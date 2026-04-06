from __future__ import annotations

import math
import statistics
from typing import Sequence


def euclidean_distance(point_a: Sequence[float], point_b: Sequence[float]) -> float:
    dx = float(point_b[0]) - float(point_a[0])
    dy = float(point_b[1]) - float(point_a[1])
    return math.sqrt(dx * dx + dy * dy)


def cluster_indices(fireflies, center_index: int, radius: float) -> list[int]:
    center = fireflies[center_index]
    return [
        index
        for index, candidate in enumerate(fireflies)
        if euclidean_distance(center.position, candidate.position) <= radius
    ]


def cluster_brightness_values(fireflies, brightness_values: Sequence[float], radius: float) -> list[float]:
    cluster_scores: list[float] = []
    for index in range(len(fireflies)):
        member_indexes = cluster_indices(fireflies, index, radius)
        if not member_indexes:
            cluster_scores.append(float(brightness_values[index]))
            continue

        values = [float(brightness_values[member_index]) for member_index in member_indexes]
        cluster_scores.append(sum(values) / len(values))

    return cluster_scores


def cluster_cumulative_values(fireflies, brightness_values: Sequence[float], radius: float) -> list[float]:
    cluster_totals: list[float] = []
    for index in range(len(fireflies)):
        member_indexes = cluster_indices(fireflies, index, radius)
        if not member_indexes:
            cluster_totals.append(float(brightness_values[index]))
            continue

        values = [float(brightness_values[member_index]) for member_index in member_indexes]
        cluster_totals.append(sum(values))

    return cluster_totals


def _adaptive_noise_floor(cluster_scores: Sequence[float]) -> float:
    if not cluster_scores:
        return 0.0

    if len(cluster_scores) == 1:
        return float(cluster_scores[0])

    median_score = statistics.median(cluster_scores)
    upper_half = [score for score in cluster_scores if score >= median_score]
    lower_half = [score for score in cluster_scores if score <= median_score]
    q3 = statistics.median(upper_half) if upper_half else median_score
    q1 = statistics.median(lower_half) if lower_half else median_score
    iqr = q3 - q1
    return float(median_score + 0.5 * iqr)


def select_cluster_indexes(cluster_scores: Sequence[float], noise_floor: float | None, cluster_chosen: int) -> list[int]:
    threshold = _adaptive_noise_floor(cluster_scores) if noise_floor is None else float(noise_floor)
    ranked_indexes = [
        index
        for index, score in sorted(enumerate(cluster_scores), key=lambda item: item[1], reverse=True)
        if score > threshold
    ]

    if not ranked_indexes:
        ranked_indexes = [max(range(len(cluster_scores)), key=lambda index: cluster_scores[index])]

    return ranked_indexes[: max(1, cluster_chosen)]