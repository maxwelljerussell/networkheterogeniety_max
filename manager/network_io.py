import pickle
from datetime import datetime
from pathlib import Path
from typing import Mapping

import networkx as nx

from helpers.plotting import plot_network
from manager.database.db import DB_ROOT
from classes.netsalt_job import NetsaltJob
from manager.database.jobs_io import insert_job
from manager.log import info, warn, err, dbg, get_network_logger

def _network_dir(network_id: str) -> Path:
    d = DB_ROOT / network_id
    d.mkdir(parents=True, exist_ok=True)
    return d

def generate_network_id(batch_id: str) -> str:
    """
    Create a unique network identifier that includes a timestamp + batch_id.

    Example: "2025-12-10_14-23-55_123456_batchA"
    """
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    return f"{ts}_{batch_id}"


def init_network_in_db(
    G: nx.Graph,
    batch_id: str,
    db_root: Path = DB_ROOT,
) -> str:
    """
    Create a new network entry in the 'database' folder:

        database/<network_id>/
            network.gpickle
            network.png
            network.pdf

    Returns the generated network_id.
    """
    network_id = generate_network_id(batch_id)

    network_dir = db_root / network_id
    info(f"[network-init] Creating network {network_id} in {network_dir}")
    network_dir.mkdir(parents=True, exist_ok=False)

    network_plot_base = network_dir / "network"
    plot_network(
        graph=G,
        path=str(network_plot_base),
        plot_nodes=False,
        edge_width=0.3,
        color="black",
        node_size=1.0,
    )

    network_graph_path = network_dir / "network.gpickle"
    with network_graph_path.open("wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    info(f"[network-init] Finished saving network {network_id}")
    return network_id


def import_networks_and_seed_jobs(
    pickle_path: Path,
    batch_id: str,
):
    """
    1) Load networks from a pickle (dict or list).
    2) Save each network into database/<network_id>/...
    3) Insert a passive job for each network into the jobs DB.

    The pickle is allowed to be:
      - dict[str, nx.Graph]
      - dict[str, {"graph": nx.Graph, "metadata": ...}]
      - list[nx.Graph]
    """
    info(f"[seed] Starting import from pickle: {pickle_path} (batch_id={batch_id})")

    if not pickle_path.exists():
        msg = f"[seed] Pickle file not found: {pickle_path}"
        err(msg)
        raise FileNotFoundError(msg)
    
    with pickle_path.open("rb") as f:
        data = pickle.load(f)

    graphs: dict[str, nx.Graph] = {}

    if isinstance(data, dict):
        for name, val in data.items():
            if isinstance(val, nx.Graph):
                G = val
            elif isinstance(val, dict) and "graph" in val:
                G = val["graph"]
            else:
                msg = (
                    f"[seed] Unsupported value type for key '{name}': {type(val)}. "
                    "Expected nx.Graph or {'graph': G, ...}."
                )
                err(msg)
                raise TypeError(msg)
            graphs[name] = G

    elif isinstance(data, list):
        for i, val in enumerate(data):
            if not isinstance(val, nx.Graph):
                msg = (
                    f"[seed] List element {i} has type {type(val)}, expected nx.Graph."
                )
                err(msg)
                raise TypeError(msg)
            graphs[f"net_{i}"] = val
    else:
        msg = f"[seed] Unsupported pickle format: {type(data)}"
        err(msg)
        raise TypeError(msg)

    info(f"[seed] Found {len(graphs)} networks to import")
    network_ids: list[str] = []

    for name, G in graphs.items():
        info(f"[seed] Importing network '{name}'")

        try:
            # 1) Save graph + plot to database/<network_id>
            network_id = init_network_in_db(G, batch_id=batch_id)
            network_ids.append(network_id)

            net_log = get_network_logger(network_id)
            net_log.info(f"Network imported: original name={name}")

            # 2) Create PASSIVE job
            passive_job = NetsaltJob(
                network_id=network_id,
                pattern_id=None,
                pattern_idx=None,
                pattern_size=None,
                job_type="passive",
            )
            insert_job(passive_job)
            dbg(f"[seed] Inserted passive job: {passive_job.job_id}")

        except Exception as e:
            err(f"[seed] ERROR importing network '{name}': {e}")
            continue

    if not network_ids:
        msg = "[seed] No networks were successfully imported."
        err(msg)
        raise RuntimeError(msg)
    info(f"[seed] Finished importing networks. Total imported={len(network_ids)}")
    return network_ids