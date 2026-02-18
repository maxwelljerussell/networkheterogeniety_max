import networkx as nx
import numpy as np

def clustering_heterogeneity(G):
    """
    Compute clustering heterogeneity for graph G:
    H = (1/N) * sum_i (C_i - mean(C))^2
    """
    C = nx.clustering(G)
    
    values = np.array(list(C.values()))
    mean = values.mean()
    H = np.mean((values - mean)**2)    # variance
    
    return H, C
