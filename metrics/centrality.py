from collections import deque

import networkx as nx
import numpy as np


def _variance(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return 0.0
    mean = values.mean()
    return float(np.mean((values - mean) ** 2))


def _betweenness_centrality_dict(adj_dict):
    """
    Brandes algorithm on an unweighted, undirected graph
    given as {node: [neighbors]}.
    """
    G = adj_dict
    BC = dict.fromkeys(G, 0.0)

    for s in G:
        stack = []
        pred = {w: [] for w in G}
        sigma = dict.fromkeys(G, 0.0)
        sigma[s] = 1.0
        dist = dict.fromkeys(G, -1)
        dist[s] = 0

        queue = deque([s])
        while queue:
            v = queue.popleft()
            stack.append(v)
            for w in G[v]:
                if dist[w] < 0:
                    queue.append(w)
                    dist[w] = dist[v] + 1
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)

        delta = dict.fromkeys(G, 0.0)
        while stack:
            w = stack.pop()
            for v in pred[w]:
                delta_v = (sigma[v] / sigma[w]) * (1 + delta[w])
                delta[v] += delta_v
            if w != s:
                BC[w] += delta[w]

    # normalize for undirected
    scale = 1 / 2
    for v in BC:
        BC[v] *= scale
    return BC


def betweenness_centrality_heterogeneity(G: nx.Graph):
    """
    Compute betweenness centrality for each node of G and return
    (heterogeneity, centrality_dict).
    """
    adj = nx.to_dict_of_lists(G)
    centrality_dict = _betweenness_centrality_dict(adj)
    values = list(centrality_dict.values())
    H = _variance(values)
    return H, centrality_dict
