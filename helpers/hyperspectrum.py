"""
hyperspectrum.py

Standalone utilities for constructing and plotting a (k, y) hyperspectrum
from netSALT lasing modes.
"""

from typing import Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import netsalt

from classes.lasing_modes import LasingModes

def _get_modal_intensities(
    lasing_modes: LasingModes,
) -> np.ndarray:
    try:
        return np.real(lasing_modes.lasing_modes["modal_intensities"].iloc[:, -1])
    except:
        return np.real(lasing_modes.lasing_modes["modal_intensities"])

def get_hyperspectrum(
    lasing_modes: LasingModes,
    max_extent: float,
    lasing_width: float = 0.005,  # Width of the lasing peak in k space
    #
    edge_leakage_factor: float = 1.0,
    edge_leakage_radius: float = 0.5,  # Radius of the edge leakage blur
    #
    node_leakage_factor: float = 0.0,
    node_leakage_radius: float = 1.0,  # Radius of the node leakage blob in y space
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct a (k, y) hyperspectrum by summing over lasing modes.

    Returns
    -------
    ks : np.ndarray
        Wavenumber grid
    ys : np.ndarray
        Vertical spatial grid
    hyperspectrum : np.ndarray
        2D intensity map with shape (len(ks), len(ys))
    """

    graph = lasing_modes.quantum_graph_with_pump

    # ------------------------------------------------------------------
    # Initialize grids
    # ------------------------------------------------------------------
    ks, _ = netsalt.get_scan_grid(graph)
    ys = np.linspace(-max_extent / 2, max_extent / 2, num=1000)

    hyperspectrum_edges = np.zeros((len(ks), len(ys)))
    hyperspectrum_nodes = np.zeros((len(ks), len(ys)))

    # ------------------------------------------------------------------
    # Precompute y-profiles for edges
    # ------------------------------------------------------------------
    if edge_leakage_factor > 0:
        edge_y_factors = []

        for edge in graph.edges:
            pos1 = graph.nodes[edge[0]]["position"]
            pos2 = graph.nodes[edge[1]]["position"]

            length = np.linalg.norm(np.array(pos2) - np.array(pos1))
            min_y = min(pos1[1], pos2[1])
            max_y = max(pos1[1], pos2[1])
            length_y = max_y - min_y

            y_factor = np.zeros(len(ys))
            norm = length / (
                length_y + np.sqrt(2 * np.pi * edge_leakage_radius**2)
            )

            # Below edge
            idx = ys < min_y
            y_factor[idx] = norm * np.exp(
                -((min_y - ys[idx]) ** 2) / (2 * edge_leakage_radius**2)
            )

            # On edge
            idx = (ys >= min_y) & (ys <= max_y)
            y_factor[idx] = norm

            # Above edge
            idx = ys > max_y
            y_factor[idx] = norm * np.exp(
                -((max_y - ys[idx]) ** 2) / (2 * edge_leakage_radius**2)
            )

            edge_y_factors.append(y_factor)

        edge_y_factors_matrix = np.vstack(edge_y_factors)

    # ------------------------------------------------------------------
    # Precompute y-profiles for nodes
    # ------------------------------------------------------------------
    if node_leakage_factor > 0:
        node_y_factors = []

        for node in graph.nodes:
            # Skip degree-2 nodes
            if graph.degree(node) == 2:
                node_y_factors.append(np.zeros(len(ys)))
                continue

            pos = graph.nodes[node]["position"]
            y_factor = np.zeros(len(ys))

            for i, y in enumerate(ys):
                dy = y - pos[1]
                if abs(dy) <= node_leakage_radius:
                    y_factor[i] = np.sqrt(
                        node_leakage_radius**2 - dy**2
                    ) / (np.pi / 2 * node_leakage_radius)

            node_y_factors.append(y_factor)

        node_y_factors_matrix = np.vstack(node_y_factors)

    # ------------------------------------------------------------------
    # Iterate over lasing modes
    # ------------------------------------------------------------------
    modes_df = lasing_modes.lasing_modes
    threshold_modes = np.real(modes_df["threshold_lasing_modes"])

    intensities = _get_modal_intensities(lasing_modes)
    intensities = np.nan_to_num(intensities, nan=0.0)

    for mode_index, (mode, intensity) in enumerate(
        zip(threshold_modes, intensities)
    ):
        if intensity == 0:
            continue

        # Set pump level for correct mode profile
        graph.graph["params"]["D0"] = modes_df["lasing_thresholds"][mode_index]

        try:
            edge_solutions = netsalt.mean_mode_on_edges(mode, graph)
        except Exception:
            edge_solutions = None

        try:
            node_solutions = np.real(
                netsalt.mode_on_nodes(mode, graph)
            ) ** 2
        except Exception:
            node_solutions = None

        # Lorentzian spectral profile
        k_factor = lasing_width**2 / ((ks - mode) ** 2 + lasing_width**2)

        edge_sum = 0.0
        if edge_leakage_factor > 0 and edge_solutions is not None:
            edge_sum = edge_solutions @ edge_y_factors_matrix

        node_sum = 0.0
        if node_leakage_factor > 0 and node_solutions is not None:
            node_sum = node_solutions @ node_y_factors_matrix

        hyperspectrum_edges += intensity * k_factor[:, None] * edge_sum
        hyperspectrum_nodes += intensity * k_factor[:, None] * node_sum

    # ------------------------------------------------------------------
    # Normalize and combine
    # ------------------------------------------------------------------
    if edge_leakage_factor > 0 and hyperspectrum_edges.sum() > 0:
        hyperspectrum_edges /= hyperspectrum_edges.sum()

    if node_leakage_factor > 0 and hyperspectrum_nodes.sum() > 0:
        hyperspectrum_nodes /= hyperspectrum_nodes.sum()

    hyperspectrum = (
        edge_leakage_factor * hyperspectrum_edges
        + node_leakage_factor * hyperspectrum_nodes
    )

    return ks, ys, hyperspectrum


def plot_hyperspectrum(
    lasing_modes: LasingModes,
    ax: plt.Axes,
    max_extent: float,
    lasing_width: float = 0.005,
    cmap: str = "viridis",
    data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    vmax: Optional[float] = None,
    log_scale: bool = False,
):
    """
    Plot a hyperspectrum on a given Matplotlib Axes.
    """

    if data is not None:
        ks, ys, hyperspectrum = data
    else:
        ks, ys, hyperspectrum = get_hyperspectrum(
            lasing_modes,
            max_extent,
            lasing_width=lasing_width,
        )

    im = ax.imshow(
        hyperspectrum.T,
        aspect="auto",
        extent=[ks[0], ks[-1], ys[0], ys[-1]],
        origin="lower",
        cmap=cmap,
        vmin=None if log_scale else 0,
        vmax=vmax,
        norm=LogNorm(vmax=vmax) if log_scale else None,
    )

    ax.set_xlabel("k")
    ax.set_ylabel("y")
    ax.set_title("Hyperspectrum")

    return im

def crop_and_pool_hyperspectrum(
    data: Tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    # k-crop controls (energy in S(k)=sum_y H)
    q_lo: float = 0.005,
    q_hi: float = 0.995,
    pad_frac: float = 0.05,
    # target pooled size
    out_k: int = 128,
    out_y: int = 64,
    # optional extra processing
    normalize: bool = False,
    log1p: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Input: (ks, ys, H) from get_hyperspectrum.

    Output: (ks2, ys2, H2) where H2 has shape (out_k, out_y) via:
      1) crop in k using cumulative-energy quantiles on S(k)=sum_y H(k,y)
      2) block-mean pool to (out_k, out_y)

    Special cases / guarantees:
      - If total energy is zero (all H==0), returns an all-zero (out_k,out_y) hyperspectrum.
      - If the cropped k-region has fewer than out_k samples, we DO NOT shrink below out_k.
        Instead we keep at least out_k k-samples by expanding the crop window around the
        lasing region (or falling back to the full ks range).
      - If input K or Y are smaller than out_k/out_y, we raise (should never happen in your pipeline).
      - Returns binned axes (block means) so len(ks2)==out_k and len(ys2)==out_y.

    Notes:
      - Pooling trims edges so dimensions divide cleanly.
    """
    ks, ys, H = data
    ks = np.asarray(ks)
    ys = np.asarray(ys)
    H = np.asarray(H, dtype=np.float32)

    out_k = int(out_k)
    out_y = int(out_y)
    if out_k <= 0 or out_y <= 0:
        raise ValueError("out_k/out_y must be > 0")

    K_full, Y_full = H.shape
    if K_full < out_k or Y_full < out_y:
        raise ValueError(f"Input H is too small: {H.shape}, need at least ({out_k},{out_y})")

    # Optional pre-processing
    if normalize:
        s = float(H.sum())
        if s > 0:
            H = H / s
    if log1p:
        H = np.log1p(H)

    # ---- total energy check ----
    total_energy = float(H.sum())
    if (not np.isfinite(total_energy)) or total_energy <= 0.0:
        # Return empty pooled hyperspectrum, with simple evenly-spaced axes
        # (axes only matter for plotting; features just see zeros)
        ks2 = np.linspace(float(ks[0]), float(ks[-1]), out_k, dtype=np.float32)
        ys2 = np.linspace(float(ys[0]), float(ys[-1]), out_y, dtype=np.float32)
        H2 = np.zeros((out_k, out_y), dtype=np.float32)
        return ks2, ys2, H2

    # ---- crop in k by cumulative energy on S(k) ----
    S = H.sum(axis=1)  # (K,)
    total_S = float(S.sum())
    if total_S <= 0.0 or not np.isfinite(total_S):
        # same as "no energy" in practice
        ks2 = np.linspace(float(ks[0]), float(ks[-1]), out_k, dtype=np.float32)
        ys2 = np.linspace(float(ys[0]), float(ys[-1]), out_y, dtype=np.float32)
        H2 = np.zeros((out_k, out_y), dtype=np.float32)
        return ks2, ys2, H2

    cdf = np.cumsum(S) / total_S
    i_lo = int(np.searchsorted(cdf, float(q_lo), side="left"))
    i_hi = int(np.searchsorted(cdf, float(q_hi), side="right")) - 1

    i_lo = max(0, min(i_lo, len(ks) - 1))
    i_hi = max(0, min(i_hi, len(ks) - 1))
    if i_hi <= i_lo:
        # fallback to full range
        j_lo, j_hi = 0, len(ks) - 1
    else:
        width = i_hi - i_lo + 1
        pad = int(round(float(pad_frac) * width))
        j_lo = max(0, i_lo - pad)
        j_hi = min(len(ks) - 1, i_hi + pad)

    # Ensure we keep at least out_k k-samples (don't shrink below out_k)
    crop_len = j_hi - j_lo + 1
    if crop_len < out_k:
        need = out_k - crop_len
        left_extra = need // 2
        right_extra = need - left_extra

        j_lo2 = max(0, j_lo - left_extra)
        j_hi2 = min(len(ks) - 1, j_hi + right_extra)

        # If still short due to boundaries, push the other side
        crop_len2 = j_hi2 - j_lo2 + 1
        if crop_len2 < out_k:
            remaining = out_k - crop_len2
            # try extend left, then right
            j_lo2 = max(0, j_lo2 - remaining)
            crop_len3 = j_hi2 - j_lo2 + 1
            if crop_len3 < out_k:
                j_hi2 = min(len(ks) - 1, j_hi2 + (out_k - crop_len3))

        j_lo, j_hi = j_lo2, j_hi2

    ks_c = ks[j_lo : j_hi + 1]
    H_c = H[j_lo : j_hi + 1, :]

    # ---- pool to (out_k, out_y) via block mean ----
    Kc, Yc = H_c.shape

    # Pooling requires output <= input
    if Kc < out_k:
        # Should be prevented by expansion above; but keep safe.
        # Fallback: don't crop (use full)
        ks_c = ks
        H_c = H
        Kc, Yc = H_c.shape
        if Kc < out_k:
            raise ValueError(f"Even full ks length {Kc} < out_k {out_k}")

    if Yc < out_y:
        raise ValueError(f"ys length {Yc} < out_y {out_y}")

    k_block = Kc // out_k
    y_block = Yc // out_y
    K2 = k_block * out_k
    Y2 = y_block * out_y

    H_trim = H_c[:K2, :Y2]
    H2 = H_trim.reshape(out_k, k_block, out_y, y_block).mean(axis=(1, 3)).astype(np.float32, copy=False)

    # ---- binned axes to match pooled grid ----
    ks_trim = ks_c[:K2]
    ks2 = ks_trim.reshape(out_k, k_block).mean(axis=1).astype(np.float32, copy=False)

    ys_trim = ys[:Y2]
    ys2 = ys_trim.reshape(out_y, y_block).mean(axis=1).astype(np.float32, copy=False)

    return ks2, ys2, H2
