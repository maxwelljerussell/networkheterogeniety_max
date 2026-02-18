import pickle
from pathlib import Path
from typing import Tuple

import networkx as nx

from classes.netsalt_job import NetsaltJob
from classes.passive_modes import PassiveModes
from classes.lasing_modes import LasingModes
from classes.pattern import Pattern
from netsalt_pipeline.simulation import compute_passive_modes, compute_lasing_modes_sweep
from netsalt_pipeline.nonlinearity import pump_trajectories_sweep

from manager.pattern_io import pattern_size_to_str, pattern_from_bitmap
from manager.network_io import _network_dir
from manager.database.db import DB_ROOT
from manager.log import info, dbg, warn, err, get_network_logger

def _ensure_exists(path: Path) -> Path:
    if not path.exists():
        err(f"[IO] Expected file not found: {path}")
        raise FileNotFoundError(f"Expected file not found: {path}")
    return path

def run_passive_job(job: NetsaltJob):
    log = get_network_logger(job.network_id)
    info(f"[passive] Starting passive job for network={job.network_id}")
    log.info("Starting passive job")

    G = load_network(job.network_id)
    passive = compute_passive_modes(G, job.config)

    path = save_passive_modes(job.network_id, passive)

    info(f"[passive] Saved passive modes for {job.network_id} to {path}")
    log.info(f"Passive modes saved to {path}")
    return passive

def run_lasing_job(job: NetsaltJob):
    log = get_network_logger(job.network_id)
    info(f"[lasing] Starting lasing job for network={job.network_id}, pattern_idx={job.pattern_idx}")
    log.info(f"Starting lasing job (pattern_idx={job.pattern_idx})")

    G = load_network(job.network_id)
    passive_modes = load_passive_modes(job.network_id)

    idx = 0 if job.pattern_idx is None else job.pattern_idx
    pattern = load_pattern(job.pattern_id, job.pattern_size, idx=idx)

    lasing_modes = compute_lasing_modes_sweep(
        G, passive_modes, pattern, job.config
    )

    path = save_lasing_modes(job.network_id, job.pattern_idx, lasing_modes)
    info(f"[lasing] Saved lasing modes for network={job.network_id}, idx={idx} to {path}")
    log.info(f"Lasing modes saved to {path}")

    return lasing_modes

def run_trajectory_job(job: NetsaltJob):
    log = get_network_logger(job.network_id)
    info(f"[traj] Starting trajectory job for network={job.network_id}, pattern_idx={job.pattern_idx}")
    log.info(f"Starting trajectory job (pattern_idx={job.pattern_idx})")

    lasing_modes = load_lasing_modes(job.network_id, job.pattern_idx)

    D0_traj, K, K_approx, modes_traj = pump_trajectories_sweep(
        lasing_modes, return_approx=False
    )

    path = save_trajectories(
        job.network_id,
        job.pattern_idx,
        D0_traj,
        K,
        K_approx,
        modes_traj,
    )
    
    info(f"[traj] Saved trajectories for network={job.network_id}, idx={job.pattern_idx} to {path}")
    log.info(f"Trajectories saved to {path}")
    return {
        "D0_traj": D0_traj,
        "K": K,
        "K_approx": K_approx,
        "modes_traj": modes_traj,
    }

def load_network(network_id: str, db_root: Path = DB_ROOT) -> nx.Graph:
    path = db_root / network_id / "network.gpickle"  # Auto Slash Convert with Pathlib
    _ensure_exists(path)

    with path.open("rb") as f:
        return pickle.load(f)

def load_passive_modes(network_id: str, db_root: Path = DB_ROOT) -> PassiveModes:
    path = db_root / network_id / "passive_modes.pkl"
    _ensure_exists(path)

    with path.open("rb") as f:
        return pickle.load(f)

def load_pattern(
        pattern_id: str,
        pattern_size: Tuple[int, int],
        idx: int,
        db_root: Path = DB_ROOT,
) -> Pattern:
    size_str = pattern_size_to_str(pattern_size)

    path = db_root / "patterns" / size_str / pattern_id / f"{idx}.bmp"
    _ensure_exists(path)

    pattern = pattern_from_bitmap(path)

    return pattern

def load_lasing_modes(
        network_id: str,
        pattern_idx: str, 
        db_root: Path = DB_ROOT
) -> LasingModes:
    path = db_root / network_id / "lasing_modes" / f"{pattern_idx}.pkl"
    _ensure_exists(path)

    with path.open("rb") as f:
        return pickle.load(f)

def save_passive_modes(network_id: str, passive_modes) -> Path:
    network_dir = _network_dir(network_id)
    path = network_dir / "passive_modes.pkl"
    with path.open("wb") as f:
        pickle.dump(passive_modes, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path

def save_lasing_modes(network_id: str, pattern_idx: int, lasing_modes) -> Path:
    network_dir = _network_dir(network_id)
    out_dir = network_dir / "lasing_modes"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{pattern_idx}.pkl"
    with path.open("wb") as f:
        pickle.dump(lasing_modes, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path

def save_trajectories(
    network_id: str,
    pattern_idx: int,
    D0_traj,
    K,
    K_approx,
    modes_traj,
) -> Path:
    network_dir = _network_dir(network_id)
    out_dir = network_dir / "freq_traj"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{pattern_idx}.pkl"
    with path.open("wb") as f:
        pickle.dump(
            {
                "D0_traj": D0_traj,
                "K": K,
                "K_approx": K_approx,
                "modes_traj": modes_traj,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    return path