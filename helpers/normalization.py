from typing import Dict

def normalize_metrics(metrics_by_graph: Dict[str, Dict[str, float]]):
    """
    Normalize each metric across all graphs by dividing by the max
    value of that metric.

    Parameters
    ----------
    metrics_by_graph : dict
        {graph_name : {metric_name : value}}

    Returns
    -------
    normalized : dict
        Same structure but each metric is scaled to [0, 1].
    """
    # Get all metric names (assumes all graphs share same metrics)
    metric_names = list(next(iter(metrics_by_graph.values())).keys())

    # Find max for each metric
    max_values = {m: 0.0 for m in metric_names}
    for measures in metrics_by_graph.values():
        for m, v in measures.items():
            max_values[m] = max(max_values[m], v)

    # Normalize
    normalized = {}
    for graph_name, measures in metrics_by_graph.items():
        normalized[graph_name] = {
            m: (measures[m] / max_values[m] if max_values[m] > 0 else 0.0)
            for m in metric_names
        }

    return normalized