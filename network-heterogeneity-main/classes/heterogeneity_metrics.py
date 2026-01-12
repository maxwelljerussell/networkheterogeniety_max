from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class HeterogeneityMetrics:
    node_degree_het: Any = None
    centrality_het: Any = None
    clustering_het: Any = None
    graphlet_het: Any = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_degree_het": self.node_degree_het,
            "centrality_het": self.centrality_het,
            "clustering_het": self.clustering_het,
            "graphlet_het": self.graphlet_het,
        }
    
    
    