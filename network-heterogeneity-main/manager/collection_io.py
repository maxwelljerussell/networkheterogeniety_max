import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional

import networkx as nx

from manager.log import info, warn, err, dbg, get_network_logger
from manager.database.db import DB_ROOT, DB_PATH, connect_sqlite, execute_with_retry
from classes.network_collection import NetworkCollection
from helpers.plotting import plot_network
from classes.netsalt_job import NetsaltJob
from manager.database.jobs_io import insert_job, init_jobs_db

def init_collections_db(db_path: Path = DB_PATH):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = connect_sqlite(db_path)
    cur = conn.cursor()

    execute_with_retry(
        cur,
        """
        CREATE TABLE IF NOT EXISTS network_collections (
            collection_id TEXT PRIMARY KEY,
            batch_id TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """,
        None
    )

    execute_with_retry(
        cur,
        """
        CREATE TABLE IF NOT EXISTS network_collection_records (
            collection_id TEXT NOT NULL,
            network_id TEXT NOT NULL,
            metadata_json TEXT,
            heterogeneity_json TEXT,
            PRIMARY KEY (collection_id, network_id),
            FOREIGN KEY (collection_id) REFERENCES network_collections(collection_id)
        )
        """,
        None
    )

    conn.commit()
    conn.close()

def save_network_collection_index(
    collection,
    db_path: Path = DB_PATH,
    collection_id: Optional[str] = None,
) -> str:
    """
    Save a NetworkCollection into SQLite WITHOUT saving graphs.

    What gets stored:
      - collection_id (generated if not given)
      - batch_id
      - created_at
      - for each record: network_id, metadata_json, heterogeneity_json

    Returns:
      collection_id
    """

    if not isinstance(collection, NetworkCollection):
        raise TypeError(f"Expected NetworkCollection, got {type(collection)}")

    init_collections_db(db_path=db_path)

    if collection_id is None:
        # Stable + readable, not necessarily unique across machines; good enough if you include timestamp
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        collection_id = f"{ts}_{collection.batch_id}"

    info(f"[db] Saving NetworkCollection index: collection_id={collection_id}, batch_id={collection.batch_id}")

    conn = connect_sqlite(db_path)
    cur = conn.cursor()

    # Insert collection row (idempotent)
    execute_with_retry(
        cur,
        """
        INSERT OR REPLACE INTO network_collections (collection_id, batch_id, created_at)
        VALUES (?, ?, ?)
        """,
        (collection_id, collection.batch_id, collection.created_at),
    )

    # Insert records
    inserted = 0
    for rec in collection.records:
        # rec.graph intentionally ignored

        metadata_json = None
        if rec.metadata is not None:
            metadata_json = json.dumps(rec.metadata.to_dict())

        hetero_json = None
        if rec.heterogeneity is not None:
            hetero_json = json.dumps(rec.heterogeneity.to_dict())

        execute_with_retry(
            cur,
            """
            INSERT OR REPLACE INTO network_collection_records
                (collection_id, network_id, metadata_json, heterogeneity_json)
            VALUES (?, ?, ?, ?)
            """,
            (collection_id, rec.network_id, metadata_json, hetero_json),
        )
        inserted += 1

    conn.commit()
    conn.close()

    info(f"[db] Saved {inserted} network record(s) for collection_id={collection_id}")
    return collection_id

def init_network_in_db_with_id(
    G: nx.Graph,
    network_id: str,
    db_root: Path = DB_ROOT,
    overwrite: bool = False,
) -> str:
    """
    Save one network graph into:
      database/<network_id>/
        network.gpickle
        network.png
        network.pdf

    Uses the provided network_id (important for NetworkCollection import).
    """
    network_dir = db_root / network_id

    if network_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Network directory already exists: {network_dir} "
                f"(set overwrite=True if you want to replace it)"
            )
        # overwrite=True: keep folder, replace files
        warn(f"[network-init] Overwriting existing network dir: {network_dir}")
    else:
        network_dir.mkdir(parents=True, exist_ok=False)

    info(f"[network-init] Saving network {network_id} -> {network_dir}")

    # Plot
    network_plot_base = network_dir / "network"
    plot_network(
        graph=G,
        path=str(network_plot_base),
        plot_nodes=False,
        edge_width=0.3,
        color="black",
        node_size=1.0,
    )

    # Pickle graph
    graph_path = network_dir / "network.gpickle"
    with graph_path.open("wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

    info(f"[network-init] Finished saving network {network_id}")
    return network_id


def import_network_collection_and_seed_jobs(
    collection_pickle_path: Path,
    *,
    save_index_to_db: bool = True,
    db_path: Path = DB_PATH,
    db_root: Path = DB_ROOT,
    overwrite_network_dirs: bool = False,
    seed_passive_jobs: bool = True,
    collection_id: Optional[str] = None,
) -> str:
    """
    Load a NetworkCollection pickle, save its index to SQLite (optional),
    import each graph into database/<network_id>/, and seed passive jobs (optional).

    Returns:
      collection_id (the SQLite collection_id if saved; otherwise a generated string)
    """
    collection_pickle_path = Path(collection_pickle_path)
    if not collection_pickle_path.exists():
        raise FileNotFoundError(f"Collection pickle not found: {collection_pickle_path}")

    info(f"[collection] Loading NetworkCollection from: {collection_pickle_path}")
    collection = NetworkCollection.from_pickle(collection_pickle_path)
    info(f"[collection] Loaded collection batch_id={collection.batch_id}, records={len(collection.records)}")

    # 1) Save collection index (NO graphs) into SQLite
    if save_index_to_db:
        cid = save_network_collection_index(
            collection=collection,
            db_path=db_path,
            collection_id=collection_id,
        )
    else:
        # still return something sensible
        cid = collection_id or f"no_db_{collection.batch_id}"
        warn(f"[collection] save_index_to_db=False; not writing collection index to SQLite (cid={cid})")

    # 2) Import graphs into filesystem DB + seed jobs
    imported = 0
    job_seeded = 0

    for rec in collection.records:
        network_id = rec.network_id
        G = rec.graph

        try:
            init_network_in_db_with_id(
                G=G,
                network_id=network_id,
                db_root=db_root,
                overwrite=overwrite_network_dirs,
            )
            imported += 1

            net_log = get_network_logger(network_id)
            net_log.info(f"Imported from collection_id={cid}, batch_id={collection.batch_id}")

            if seed_passive_jobs:
                init_jobs_db()
                job = NetsaltJob(
                    network_id=network_id,
                    pattern_id=None,
                    pattern_idx=None,
                    pattern_size=None,
                    job_type="passive",
                )
                insert_job(job, db_path=db_path)  # if your insert_job accepts db_path; if not, remove arg
                job_seeded += 1
                dbg(f"[collection] Seeded passive job: job_id={job.job_id} network_id={network_id}")

        except Exception as e:
            err(f"[collection] ERROR importing network_id={network_id}: {e}")
            continue

    if imported == 0:
        raise RuntimeError("[collection] No networks were successfully imported from the collection.")

    info(f"[collection] Done. Imported={imported}, passive_jobs_seeded={job_seeded}, collection_id={cid}")
    return cid