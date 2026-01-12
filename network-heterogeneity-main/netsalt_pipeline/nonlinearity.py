from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import netsalt

from classes.lasing_modes import LasingModes


# ---------------------------------------------------------------------
# Small debug helper
# ---------------------------------------------------------------------
def dbg(msg: str, debug: bool) -> None:
    if debug:
        print(f"[nonlinearity] {msg}")


# ---------------------------------------------------------------------
# 1. Internal SALT intensity sweep (compute_modal_intensities)
# ---------------------------------------------------------------------
def internal_intensity_sweep(
    lasing: LasingModes,
    D0_max: Optional[float] = None,
    debug: bool = False,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Use netSALT.compute_modal_intensities() to perform an *internal*
    pump sweep from threshold up to D0_max.

    Parameters
    ----------
    lasing : LasingModes
        Output of your compute_lasing_modes() call. We use
        lasing.threshold_modes and lasing.competition_matrix.
    D0_max : float, optional
        Max pump to sweep to. If None, uses
        lasing.config.compute_modal_intensities.D0_max.
    debug : bool
        Print some info if True.

    Returns
    -------
    D0_internal : np.ndarray, shape (n_D0,)
        The (possibly non-uniform) pump values used internally.
    I_matrix : np.ndarray, shape (n_D0, n_modes)
        Intensities for each mode and each D0.
        Row = D0, column = mode index.
    modes_swept : pd.DataFrame
        The full modes DataFrame returned by netSALT
        (with 'modal_intensities' and 'interacting_lasing_thresholds').
    """
    if D0_max is None:
        D0_max = lasing.config.compute_modal_intensities.D0_max

    if lasing.competition_matrix is None:
        raise ValueError(
            "lasing.competition_matrix is None → no lasing modes. "
            "Cannot run internal intensity sweep."
        )

    dbg(f"Starting internal intensity sweep up to D0_max={D0_max:.4g}", debug)

    # Work on a copy so we don't mutate lasing.threshold_modes
    modes_df = lasing.threshold_modes.copy()

    # netSALT will:
    #  - start at first lasing threshold
    #  - walk up to D0_max, adding modes / removing vanishing modes
    modes_swept = netsalt.compute_modal_intensities(
        modes_df,
        D0_max,
        lasing.competition_matrix,
    )

    # Extract the "modal_intensities" sub-DataFrame:
    #   rows   = modes
    #   cols   = pump intensities (D0_internal)
    MI = modes_swept["modal_intensities"]

    if isinstance(MI.columns, pd.MultiIndex):
        raw_D0 = np.array([c[1] for c in MI.columns], dtype=float)
    else:
        raw_D0 = MI.columns.to_numpy(dtype=float)
    
    sort_idx = np.argsort(raw_D0)
    D0_internal = raw_D0[sort_idx]

    # Shape (n_modes, n_D0) → transpose to (n_D0, n_modes)
    I_matrix = MI.to_numpy()[:, sort_idx].T

    dbg(
        f"Internal sweep produced {len(D0_internal)} D0 points "
        f"and {I_matrix.shape[1]} modes.",
        debug,
    )

    return D0_internal, I_matrix, modes_swept


# ---------------------------------------------------------------------
# 2. Pump trajectories (k(D0) for each mode)
# ---------------------------------------------------------------------
def pump_trajectories_sweep(
    lasing: LasingModes,
    return_approx: bool = False,
    debug: bool = False,
):
    """
    Run netsalt.pump_trajectories() to get mode trajectories
    k(D0) in the complex plane.

    Parameters
    ----------
    lasing : LasingModes
        Must contain a quantum_graph_with_pump and threshold_modes.
    return_approx : bool
        If True, also return the linear approximate trajectories.
    debug : bool
        Print information if True.

    Returns
    -------
    D0_traj : np.ndarray, shape (n_D0,)
        Pump values used for trajectories.
    K : np.ndarray, shape (n_D0, n_modes)
        Complex frequencies k(D0) for each mode.
    K_approx : np.ndarray or None
        Same shape as K, containing the approximate linear trajectories
        if return_approx is True, else None.
    modes_traj : pd.DataFrame
        The DataFrame returned by pump_trajectories().
    """
    dbg("Starting pump_trajectories sweep...", debug)

    modes_df = lasing.threshold_modes.copy()
    qg = lasing.quantum_graph_with_pump
    quality_method = lasing.config.mode_search_config.quality_method

    modes_traj = netsalt.pump_trajectories(
        modes_df,
        qg,
        return_approx=return_approx,
        quality_method=quality_method,
    )

    MT = modes_traj["mode_trajectories"]
    # columns = D0 values
    if isinstance(MT.columns, pd.MultiIndex):
        D0_traj = np.array([c[1] for c in MT.columns], dtype=float)
    else:
        D0_traj = MT.columns.to_numpy(dtype=float)

    # shape (n_modes, n_D0) → (n_D0, n_modes)
    K = MT.to_numpy().T.astype(complex)

    K_approx = None
    if return_approx and ("mode_trajectories_approx" in modes_traj.columns):
        MTa = modes_traj["mode_trajectories_approx"]
        if isinstance(MTa.columns, pd.MultiIndex):
            _ = [c[1] for c in MTa.columns]  # same D0s
        K_approx = MTa.to_numpy().T.astype(complex)

    dbg(
        f"Trajectories: {len(D0_traj)} D0 points, {K.shape[1]} modes.",
        debug,
    )

    return D0_traj, K, K_approx, modes_traj


# ---------------------------------------------------------------------
# 3. Simple analysis helpers
# ---------------------------------------------------------------------
def mode_turn_on_off(
    D0: np.ndarray,
    I_mode: np.ndarray,
    threshold: float = 1e-6,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Estimate turn-on and turn-off D0 for a single mode.

    Parameters
    ----------
    D0 : array, shape (n_D0,)
        Pump values.
    I_mode : array, shape (n_D0,)
        Intensities of a single mode.
    threshold : float
        Intensity threshold for "active" mode.

    Returns
    -------
    D0_on : float or None
        First D0 where I_mode > threshold (None if never lasers).
    D0_off : float or None
        Last D0 where I_mode > threshold (None if never turns off).
    """
    active = I_mode > threshold
    if not np.any(active):
        return None, None

    idx = np.where(active)[0]
    return float(D0[idx[0]]), float(D0[idx[-1]])


def total_intensity(I_matrix: np.ndarray) -> np.ndarray:
    """
    Sum intensities over modes for each D0.

    Parameters
    ----------
    I_matrix : array, shape (n_D0, n_modes)

    Returns
    -------
    total : array, shape (n_D0,)
    """
    return np.nansum(I_matrix, axis=1)


def top_k_modes_by_max_intensity(
    I_matrix: np.ndarray,
    k: int = 5,
) -> List[int]:
    """
    Return indices of the k modes with largest max intensity
    over the sweep.
    """
    max_per_mode = np.nanmax(I_matrix, axis=0)
    valid = np.isfinite(max_per_mode) & (max_per_mode > 0)

    if not np.any(valid):
        # Nothing ever lases → just return empty list
        return []
    valid_indices = np.nonzero(valid)[0]
    valid_max = max_per_mode[valid]

    # Sort valid modes by descending max intensity
    order = np.argsort(valid_max)[::-1]
    top_valid = valid_indices[order[:k]]

    return top_valid.tolist()


# ---------------------------------------------------------------------
# 4. Plotting helpers (optional, for quick inspection)
# ---------------------------------------------------------------------
def plot_total_intensity(
    D0: np.ndarray,
    I_matrix: np.ndarray,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot total intensity vs D0.
    """
    if ax is None:
        fig, ax = plt.subplots()

    I_tot = total_intensity(I_matrix)
    ax.plot(D0, I_tot, marker="o")
    ax.set_xlabel("D0 (pump)")
    ax.set_ylabel("Total intensity")
    ax.set_title("Total lasing intensity vs D0")
    ax.grid(True)
    return ax


def plot_mode_intensities(
    D0: np.ndarray,
    I_matrix: np.ndarray,
    mode_indices: Sequence[int],
    ax: Optional[plt.Axes] = None,
    title: str = "Mode intensities vs D0",
) -> plt.Axes:
    """
    Plot intensity vs D0 for selected modes.
    """
    if ax is None:
        fig, ax = plt.subplots()

    for m in mode_indices:
        ax.plot(D0, I_matrix[:, m], marker="o", label=f"mode {m}")

    ax.set_xlabel("D0 (pump)")
    ax.set_ylabel("Intensity")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    return ax


def plot_top_k_modes(
    D0: np.ndarray,
    I_matrix: np.ndarray,
    k: int = 5,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    top = top_k_modes_by_max_intensity(I_matrix, k=k)
    return plot_mode_intensities(
        D0,
        I_matrix,
        mode_indices=top,
        ax=ax,
        title=f"Top {k} lasing modes",
    )

def plot_frequency_pulling(
    D0: np.ndarray,
    K: np.ndarray,
    mode_indices: Sequence[int],
    ax: Optional[plt.Axes] = None,
    title: str = "Frequency pulling vs D0",
) -> plt.Axes:
    """
    Plot Re(k) vs D0 for selected modes.
    K should be shape (n_D0, n_modes), complex.
    """
    if ax is None:
        fig, ax = plt.subplots()

    for m in mode_indices:
        ax.plot(D0, np.real(K[:, m]), marker="o", label=f"mode {m}")

    ax.set_xlabel("D0 (pump)")
    ax.set_ylabel("Re(k)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    return ax
