# idea from https://ieeexplore.ieee.org/abstract/document/7373304

import networkx as nx
import numpy as np
from itertools import combinations

def enumerate_3node_graphlets(G: nx.Graph):
    """
    Enumerate 3-node graphlets in an undirected NetworkX graph G.
    Returns:
        triangles: list of 3-tuples (u, v, w) for g_{3,1} (triangles)
        paths:     list of 3-tuples (u, v, w) for g_{3,2} (2-edge paths)
    """
    adj = {u: set(G.neighbors(u)) for u in G.nodes()}
    nodes = list(G.nodes())

    X = {u: 0 for u in nodes}

    triangles = set() 
    paths = set()

    for u in nodes:
        for v in adj[u]:
            if u < v:
                Star_u = set()
                Star_v = set()
                Tri_e = set()

                for w in adj[u]:
                    if w == v:
                        continue
                    Star_u.add(w)
                    X[w] = 1

                for w in adj[v]:
                    if w == u:
                        continue
                    if X.get(w, 0) == 1:
                        Tri_e.add(w)
                        Star_u.discard(w)
                    else:
                        Star_v.add(w)

                # triangles g_{3,1}
                for w in Tri_e:
                    tri = tuple(sorted((u, v, w)))
                    triangles.add(tri)

                # 2-edge paths g_{3,2}
                for w in Star_u:
                    s1, s2 = sorted((v, w))
                    path = (u, s1, s2)
                    paths.add(path)
                for w in Star_v:
                    s1, s2 = sorted((u, w))
                    path = (v, s1, s2)
                    paths.add(path)

                for w in adj[u]:
                    if w == v:
                        continue
                    X[w] = 0

    triangles = sorted(triangles)
    paths = sorted(paths)

    return triangles, paths

def count_3node_graphlets(G: nx.Graph):
    """
    Returns counts of 3-node graphlets:
      g3_1: triangles
      g3_2: 2-edge paths
    as a dict and as a fixed-order vector.
    """
    triangles, paths = enumerate_3node_graphlets(G)
    c3_1 = len(triangles)  # triangles
    c3_2 = len(paths)      # 2-edge paths

    counts_dict = {
        "g3_1": c3_1,
        "g3_2": c3_2,
    }
    # fixed order if you want a vector form
    counts_vec = [c3_1, c3_2]

    return counts_dict, counts_vec

def enumerate_4node_graphlets(G: nx.Graph):
    """
    Enumerate 4-node graphlets g4_1..g4_6 in an undirected NetworkX graph G.

    Uses triangles and 3-node paths from enumerate_3node_graphlets plus
    adjacency structure to detect 4-node motifs in a triangle-/path-centric way.

    Returns:
        {
          'g4_1': [ (a,b,c,d), ... ],   # K4 (4-clique)
          'g4_2': [ ... ],              # diamond (K4 - 1 edge)
          'g4_3': [ ... ],              # tailed triangle
          'g4_4': [ ... ],              # 4-cycle (C4)
          'g4_5': [ ... ],              # 3-star (K1,3)
          'g4_6': [ ... ],              # 4-path (P4)
        }
    """
    adj = {u: set(G.neighbors(u)) for u in G.nodes()}
    nodes = list(G.nodes())

    triangles, paths3 = enumerate_3node_graphlets(G)

    g4_1 = set()  # K4
    g4_2 = set()  # diamond
    g4_3 = set()  # tailed triangle
    g4_4 = set()  # 4-cycle
    g4_5 = set()  # 3-star
    g4_6 = set()  # 4-path

    # 1) Triangle-based expansion: g4_1, g4_2, g4_3

    for tri in triangles:
        a, b, c = tri
        tri_set = {a, b, c}

        candidate_neighbors = (adj[a] | adj[b] | adj[c]) - tri_set

        for x in candidate_neighbors:
            k = 0
            if x in adj[a]:
                k += 1
            if x in adj[b]:
                k += 1
            if x in adj[c]:
                k += 1

            quad = tuple(sorted((a, b, c, x)))

            if k == 3:
                g4_1.add(quad)
            elif k == 2:
                g4_2.add(quad)
            elif k == 1:
                g4_3.add(quad)

    # 2) Path-based expansion: g4_4 (C4), g4_5 (star), g4_6 (P4)

    # paths3 are (center, side1, side2) with side1 < side2
    for (c, s1, s2) in paths3:
        triple_set = {c, s1, s2}

        candidates = (adj[c] | adj[s1] | adj[s2]) - triple_set

        for x in candidates:
            x_s1 = x in adj[s1]
            x_c  = x in adj[c]
            x_s2 = x in adj[s2]

            score = 0
            if x_s1:
                score += 2
            if x_c:
                score += 1
            if x_s2:
                score += 2

            if score == 0:
                continue

            quad = tuple(sorted((c, s1, s2, x)))

            if score == 1:
                g4_5.add(quad)
            elif score == 2:
                g4_6.add(quad)
            elif score == 4:
                g4_4.add(quad)

    result = {
        "g4_1": sorted(g4_1),
        "g4_2": sorted(g4_2),
        "g4_3": sorted(g4_3),
        "g4_4": sorted(g4_4),
        "g4_5": sorted(g4_5),
        "g4_6": sorted(g4_6),
    }
    return result

def count_4node_graphlets(G: nx.Graph):
    """
    Returns counts of 4-node graphlets:
      g4_1..g4_6 as in your docstring
    as a dict and as a fixed-order vector.
    """
    g4 = enumerate_4node_graphlets(G)

    counts_dict = {name: len(g4[name]) for name in ["g4_1","g4_2","g4_3","g4_4","g4_5","g4_6"]}
    counts_vec = [counts_dict[f"g4_{i}"] for i in range(1, 7)]

    return counts_dict, counts_vec

def count_3_4_graphlets(G: nx.Graph):
    """
    Combined counts in a fixed order:
    [g3_1, g3_2, g4_1, g4_2, g4_3, g4_4, g4_5, g4_6]
    """
    _, c3 = count_3node_graphlets(G)
    _, c4 = count_4node_graphlets(G)
    return c3 + c4
