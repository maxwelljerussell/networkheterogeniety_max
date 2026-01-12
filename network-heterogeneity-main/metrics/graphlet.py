import math
import numpy as np
import networkx as nx

from graphlet.graphlet_counter import count_3_4_graphlets

def randomize_graph_degree_preserving(G: nx.Graph,
                                      nswap_factor: float = 10.0,
                                      max_tries_factor: float = 100.0):
    """
    Randomize G by double-edge swaps, preserving degrees.
    Returns a new graph.
    """
    H = G.copy()
    m = H.number_of_edges()
    nswap = int(nswap_factor * m)
    max_tries = int(max_tries_factor * m)
    if nswap > 0 and m > 1:
        nx.double_edge_swap(H, nswap=nswap, max_tries=max_tries)
    return H

def graphlet_z_scores(G: nx.Graph, n_shuffles: int = 100):
    """
    Compute z-scores for 3+4 node graphlets using degree-preserving shuffles.
    Returns:
        z : np.array of length 8 (same order as count_3_4_graphlets)
        real_counts, mean_null, std_null
    """
    # Make sure real_counts is a NumPy array
    real_counts = np.asarray(count_3_4_graphlets(G), dtype=float)

    # collect motif counts over shuffled graphs
    samples = []
    for _ in range(n_shuffles):
        Gr = randomize_graph_degree_preserving(G)
        samples.append(count_3_4_graphlets(Gr))

    # shape: (n_shuffles, n_motifs)
    samples = np.asarray(samples, dtype=float)

    mean_null = samples.mean(axis=0)
    std_null = samples.std(axis=0, ddof=0)

    # z_i = (N_i - <N_i^s>) / std_i
    z = np.zeros_like(real_counts, dtype=float)
    mask = std_null > 0
    z[mask] = (real_counts[mask] - mean_null[mask]) / std_null[mask]
    # if std_null == 0, leave z = 0

    return z, real_counts, mean_null, std_null


def heterogeneity_from_z(z: np.ndarray) -> float:
    """
    Motif heterogeneity from z-scores (Motif Significance Profile entropy).
    Returns value in [0, 1].
    """
    zabs = np.abs(z.astype(float))
    Zsum = zabs.sum()
    if Zsum == 0:
        return 0.0

    p = zabs / Zsum
    p_nonzero = p[p > 0]
    H = -(p_nonzero * np.log(p_nonzero)).sum()
    M = len(p_nonzero)
    if M <= 1:
        return 0.0
    return float(H / math.log(M))


def graphlet_heterogeneity(G: nx.Graph,
                           n_shuffles: int = 100):
    """
    Full pipeline:
      1) Counts 3+4-node motifs
      2) Computes z-scores via degree-preserving shuffles
      3) Returns heterogeneity H_norm in [0,1] and the z-vector
    """
    z, real_counts, mean_null, std_null = graphlet_z_scores(G, n_shuffles=n_shuffles)
    H = heterogeneity_from_z(z)
    return H, z, real_counts, mean_null, std_null