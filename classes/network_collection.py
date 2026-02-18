from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Sequence
import uuid

import pickle
import networkx as nx

from classes.heterogeneity_metrics import HeterogeneityMetrics
from classes.network_metadata import NetworkMetadata
from manager.network_io import generate_network_id

@dataclass
class NetworkRecord:
    network_id: str
    graph: nx.Graph
    metadata: NetworkMetadata
    heterogeneity: Optional[HeterogeneityMetrics] = None


@dataclass
class NetworkCollection:
    batch_id: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    records: List[NetworkRecord] = field(default_factory=list)

    @classmethod
    def from_networks(
        cls,
        networks: Sequence[nx.Graph],
        batch_id: str,
        metadata_list: Optional[Sequence[NetworkMetadata]] = None,
    ) -> "NetworkCollection":
        if metadata_list is not None and len(metadata_list) != len(networks):
            raise ValueError("metadata_list must have same length as networks")

        coll = cls(batch_id=batch_id)
        for i, G in enumerate(networks):
            nid = f"{batch_id}_{i:04d}_{uuid.uuid4().hex[:8]}"
            meta = metadata_list[i] if metadata_list is not None else None
            coll.records.append(NetworkRecord(network_id=nid, graph=G, metadata=meta))
        return coll

    def __len__(self) -> int:
        return len(self.records)
    
    def network_ids(self) -> List[str]:
        return [r.network_id for r in self.records]
    
    def set_metadatga(self, network_id: str, metadata: NetworkMetadata) -> None:
        rec = self._get_record(network_id)
        rec.metadata = metadata

    def set_heterogeneity_metrics(
        self,
        network_id: str,
        node_degree_het: Any = None,
        centrality_het: Any = None,
        clustering_het: Any = None,
        graphlet_het: Any = None,
    ) -> None:
        rec = self._get_record(network_id)
        if node_degree_het is not None:
            rec.heterogeneity.node_degree_het = node_degree_het
        if centrality_het is not None:
            rec.heterogeneity.centrality_het = centrality_het
        if clustering_het is not None:
            rec.heterogeneity.clustering_het = clustering_het
        if graphlet_het is not None:
            rec.heterogeneity.graphlet_het = graphlet_het

    def to_pickle(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    @staticmethod
    def from_pickle(path: Path) -> "NetworkCollection":
        path = Path(path)
        with path.open("rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, NetworkCollection):
            raise TypeError(f"Pickle did not contain NetworkCollection: got {type(obj)}")
        return obj

    def _get_record(self, network_id: str) -> NetworkRecord:
        for rec in self.records:
            if rec.network_id == network_id:
                return rec
        raise KeyError(f"Unknown network_id: {network_id}")