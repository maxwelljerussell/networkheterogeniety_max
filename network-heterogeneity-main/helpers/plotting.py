# helpers/plotting.py

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from typing import Union
from classes.pattern import Pattern
from helpers.pixelation import pixelate_network
from netsalt_pipeline.simulation import compute_pump_profile


def plot_networks_and_metrics(graphs, metrics_by_name, normalized=True):
    """
    One figure:
      - top row: toy network plots
      - bottom row: per-network bar chart of all metrics.

    Parameters
    ----------
    graphs : dict
        {graph_name: nx.Graph}
    metrics_by_name : dict
        {graph_name: {metric_name: value}}
    normalized : bool
        Whether the metrics have been normalized (only used for axis label).
    """
    n = len(graphs)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    names = list(graphs.keys())

    for j, name in enumerate(names):
        G = graphs[name]
        metrics = metrics_by_name[name]

        # --- top: network ---
        ax_net = axes[0, j]
        pos = nx.spring_layout(G, seed=0)
        nx.draw_networkx(G, pos=pos, ax=ax_net, node_size=40, with_labels=False)
        ax_net.set_title(name)
        ax_net.axis("off")

        # --- bottom: metrics ---
        ax_bar = axes[1, j]
        metric_names = list(metrics.keys())
        metric_values = [metrics[m] for m in metric_names]

        x = np.arange(len(metric_names))
        ax_bar.bar(x, metric_values)
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(metric_names, rotation=60, ha="right")
        ax_bar.set_ylabel("Normalized value" if normalized else "Value")
        ax_bar.set_title(
            f"{'Normalized ' if normalized else ''}metrics for {name}"
        )
        ax_bar.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle("Networks and their heterogeneity metrics", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_network(
    graph: nx.Graph,
    path: Union[str, None] = None,
    plot_nodes: bool = False,
    ax: Union[plt.Axes, None] = None,
    edge_width: float = 0.3,
    pattern: Union[Pattern, None] = None,
    max_extent: Union[int, None] = None,
    color="red",
    node_size: float = 1.0,
    node_color="black",
    cmap=None,
    edge_vmin: Union[float, None] = None,
    edge_vmax: Union[float, None] = None,
    node_vmin: Union[float, None] = None,
    node_vmax: Union[float, None] = None,
):
    """
    Plot the graph.
    """

    external_axis = ax is not None
    save_fig = path is not None
    with_pattern = pattern is not None

    if with_pattern:
        width = pattern.width
        height = pattern.height
        graph = pixelate_network(graph, width, height)
        pump = compute_pump_profile(graph, pattern)
        edge_color = [color if pump[i] > 0 else "grey" for i in range(len(pump))]
    else:
        edge_color = color

    assert not (external_axis and save_fig), "Cannot save figure with external axis."

    if ax is None:
        _, ax = plt.subplots()
    if max_extent is not None:
        ax.set_xlim(-max_extent / 2 * 1.1, max_extent / 2 * 1.1)
        ax.set_ylim(-max_extent / 2 * 1.1, max_extent / 2 * 1.1)

    positions = {u: graph.nodes[u]["position"] for u in graph.nodes}

    nx.draw(
        graph,
        positions,
        node_size=node_size if plot_nodes else 0,
        width=edge_width,
        node_color=node_color,
        cmap=cmap,
        with_labels=False,
        edge_color=edge_color,
        alpha=1,
        ax=ax,
        edge_vmin=edge_vmin,
        edge_vmax=edge_vmax,
        vmin=node_vmin,
        vmax=node_vmax,
    )

    ax.set_aspect("equal", "box")
    ax.set_axis_on()
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

    if not external_axis:
        if save_fig:
            plt.savefig(path + ".pdf")
            plt.savefig(path + ".png")
        else:
            plt.show()