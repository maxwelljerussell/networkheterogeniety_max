"""
analyzer.py

High-level interface for computing heterogeneity metrics on NetworkX graphs.
This module wraps together low-level metric functions defined in the
`metrics` package and exposes them through a single unified class.
"""

from typing import Dict
import networkx as nx

from metrics.node_degree import (
    node_degree_var,
    degree_distribution_var,
)
from metrics.centrality import (
    betweenness_centrality_heterogeneity,
)
from metrics.clustering import (
    clustering_heterogeneity,
)


class HeterogeneityAnalyzer:
    """
    Analyze structural heterogeneity of a NetworkX graph.

    Parameters
    ----------
    G : nx.Graph
        The graph to analyze.
    """

    def __init__(self, G: nx.Graph):
        self.G = G

    # Metric groups

    def degree_metrics(self) -> Dict[str, float]:
        """Compute degree-based heterogeneity measures."""
        deg_var = node_degree_var(self.G)
        deg_dist_var, h, h2 = degree_distribution_var(self.G)
        return {
            "Degree variance": deg_var,
            "Degree distribution variance": deg_dist_var,
        }

    def centrality_metrics(self) -> Dict[str, float]:
        """Compute centrality-based heterogeneity measures."""
        bc_het, _ = betweenness_centrality_heterogeneity(self.G)
        return {
            "Betweenness centrality heterogeneity": bc_het,
        }

    def clustering_metrics(self) -> Dict[str, float]:
        """Compute clustering-based heterogeneity measures."""
        clust_het, _ = clustering_heterogeneity(self.G)
        return {
            "Clustering heterogeneity": clust_het,
        }

    def all_metrics(self) -> Dict[str, float]:
        """
        Compute all available heterogeneity metrics for the graph.
        Returns a flat dictionary of metric name → value.
        """
        metrics = {}
        metrics.update(self.degree_metrics())
        metrics.update(self.centrality_metrics())
        metrics.update(self.clustering_metrics())
        return metrics