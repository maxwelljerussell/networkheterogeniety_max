from __future__ import annotations

from typing import Optional
import numpy as np


def generate_clustered_points(
    num_points: int,
    extent: float,
    n_clusters: int = 4,
    cluster_std: float = 0.08,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate clustered 2D points in a square [-extent/2, extent/2]^2.

    - n_clusters: number of Gaussian clusters
    - cluster_std: relative std as fraction of extent (e.g. 0.08 means 8% of extent)
    """
    rng = np.random.default_rng(seed)

    # cluster centers uniformly in the box
    centers = rng.uniform(-extent / 2, extent / 2, size=(n_clusters, 2))

    # allocate points to clusters (roughly equal, but random)
    counts = rng.multinomial(num_points, [1.0 / n_clusters] * n_clusters)

    pts = []
    sigma = cluster_std * extent
    for k, c in enumerate(centers):
        if counts[k] == 0:
            continue
        cloud = rng.normal(loc=c, scale=sigma, size=(counts[k], 2))
        pts.append(cloud)

    points = np.vstack(pts) if pts else np.zeros((0, 2), dtype=float)

    # clip into the box so masking doesn’t get weird
    # points = np.clip(points, -extent / 2, extent / 2)
    half = extent / 2

    # reflect into [-half, half] instead of clipping
    points = np.where(points >  half,  2*half - points, points)
    points = np.where(points < -half, -2*half - points, points)

    # in rare cases a point can still be outside after one reflection if it's far out;
    # repeat once more to be safe
    points = np.where(points >  half,  2*half - points, points)
    points = np.where(points < -half, -2*half - points, points)


    # shuffle so indices don't correlate with cluster id
    rng.shuffle(points, axis=0)
    return points