import networkx as nx
import numpy as np
from collections import Counter
import math

def node_degree_var(G, weight=None):
    degs = np.array([d for _, d in G.degree(weight=weight)])
    return np.mean((degs - degs.mean())**2)


def degree_distribution_var(G):
    """
    Compute the normalized heterogeneity measure H_m
    Returns (H_m, h, h2).
    """
    N = G.number_of_nodes()
    if N == 0:
        return 0.0, 0.0, 0.0

    degrees = [d for _, d in G.degree()]

    counts = Counter(degrees)
    Pk_vals = [c / N for c in counts.values()]

    h2 = (1.0 / N) * sum(p * (1.0 - p)**2 for p in Pk_vals)
    h = math.sqrt(h2)

    h_het2 = 1.0 - 3.0 / N + (N + 2.0) / (N**3)
    h_het = math.sqrt(h_het2) if h_het2 > 0 else 0.0

    # H_m = h / h_het
    Hm = h / h_het if h_het > 0 else 0.0
    return Hm, h, h2
