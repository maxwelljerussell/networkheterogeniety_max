from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Dict

import numpy as np
import networkx as nx

from network_generation.generate_hexgraph import generate_hex_network
from classes.network_constraints import NetworkConstraints

constraints_hex = NetworkConstraints(
    max_extent=50, 
    min_edge_length = 0.1,
    max_edge_count=np.inf, 
    max_node_count=np.inf,
    min_distance_between_node_and_edge=0.1
)

@dataclass(frozen=True)
class VoidMap:    
    """
    Spatial occupancy map over the masked region.

    node_counts[y, x] = number of nodes in bin (x, y)
    void_mask[y, x]   = True if bin is considered "void" (empty/sparse)

    Notes:
    - Coordinate system assumes the final masked region is centered at (0,0)
      and bounded by [-radius, radius] in x and y (square mask).
    """
    radius: float
    cell_size: float
    bins: int
    node_counts: np.ndarray # shape (bins, bins), dtype=int
    void_mask: np.ndarray   # shape (bins, bins), dtype=bool


def _dilate_bool_mask(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """
    Simple 8-neighborhood dilation without scipy.
    Expands True regions by 'iterations' steps.
    """
    if iterations <= 0:
        return mask
    
    out = mask.copy()
    for _ in range(iterations):
        m = out
        # OR of 8 neighbors + self
        out = (
            m
            | np.roll(m, 1, axis=0)
            | np.roll(m, -1, axis=0)
            | np.roll(m, 1, axis=1)
            | np.roll(m, -1, axis=1)
            | np.roll(np.roll(m, 1, axis=0), 1, axis=1)
            | np.roll(np.roll(m, 1, axis=0), -1, axis=1)
            | np.roll(np.roll(m, -1, axis=0), 1, axis=1)
            | np.roll(np.roll(m, -1, axis=0), -1, axis=1)
        )
    return out

def _relabel_network(G: nx.Graph):
    G_ = nx.convert_node_labels_to_integers(G)
    return G_

def compute_void_map(
    network: nx.Graph,
    radius: float,
    cell_size: float,
    *,
    void_threshold: int = 0,
    mask_type: Literal["square", "circle"] = "square",
    dilate_void_bins: int = 0,
    account_for_edges: bool = True,
) -> VoidMap:
    """
    Build a binned occupancy map of node positions and label "void" bins.

    Parameters
    ----------
    network:
        Graph whose nodes have node["position"] as (x,y) array-like.
    radius:
        Half-extent for the target region (same as mask_network radius).
        For square: valid region is x,y in [-radius, radius].
        For circle: valid region is x^2 + y^2 <= radius^2 (bins outside will be forced non-void).
    cell_size:
        Bin size in same units as positions.
    void_threshold:
        A bin is void if node_count <= void_threshold.
        - 0 means "truly empty"
        - 1 means "0 or 1 node is considered void/sparse"
    mask_type:
        "square" or "circle". For circle we prevent filling outside the circle.
    dilate_void_bins:
        Optional dilation steps to expand void regions a bit (helps avoid sharp patch edges).

    Returns
    -------
    VoidMap
    """
    if cell_size <= 0:
        raise ValueError("cell_size must be > 0.")
    if radius <= 0:
        raise ValueError("radius must be > 0.")
    if void_threshold < 0:
        raise ValueError("void_threshold must be >= 0.")

    # Number of bins per axis
    bins = int(np.ceil((2.0 * radius) / cell_size))
    bins = max(bins, 1)

    node_counts = np.zeros((bins, bins), dtype=int)

    # Collect positions
    # (network.nodes.values() is used elsewhere in your repo)
    positions = [node.get("position", None) for node in network.nodes.values()]
    positions = [p for p in positions if p is not None]
    if len(positions) == 0:
        # no nodes => everything inside mask is void
        void_mask = np.ones((bins, bins), dtype=bool)
        if mask_type == "circle":
            void_mask = _apply_circle_clip(void_mask, radius, cell_size)
        if dilate_void_bins > 0:
            void_mask = _dilate_bool_mask(void_mask, dilate_void_bins)
        return VoidMap(radius, cell_size, bins, node_counts, void_mask)

    pos = np.asarray(positions, dtype=float)  # shape (N,2)

    # Map positions from [-radius, radius] to bin indices [0, bins-1]
    # x=-radius -> 0, x=radius -> bins-1 (clipped)
    x = pos[:, 0]
    y = pos[:, 1]

    # normalized in [0, 2*radius)
    fx = (x + radius) / (2.0 * radius)
    fy = (y + radius) / (2.0 * radius)

    ix = np.floor(fx * bins).astype(int)
    iy = np.floor(fy * bins).astype(int)

    # clip to bounds (some nodes might lie slightly outside due to numerical ops)
    ix = np.clip(ix, 0, bins - 1)
    iy = np.clip(iy, 0, bins - 1)

    # Increment counts
    for j in range(ix.shape[0]):
        node_counts[iy[j], ix[j]] += 1

    void_mask = node_counts <= void_threshold

    # If circle mask, bins outside circle should NOT be considered "void fill targets"
    # (because they are outside the valid region entirely)
    if mask_type == "circle":
        void_mask = _apply_circle_clip(void_mask, radius, cell_size)

    # Optional dilation (expands void regions slightly)
    if dilate_void_bins > 0:
        void_mask = _dilate_bool_mask(void_mask, dilate_void_bins)

    voidmap = VoidMap(radius, cell_size, bins, node_counts, void_mask)

    if account_for_edges:
        voidmap = refine_void_mask_with_edges(network, voidmap, sample_step=0.5 * cell_size)

    return voidmap


def _apply_circle_clip(void_mask: np.ndarray, radius: float, cell_size: float) -> np.ndarray:
    """
    For a circular mask, mark bins outside the circle as NOT void
    (so we won't try to fill outside the domain).
    """
    bins = void_mask.shape[0]
    # Bin centers in coordinate space
    # center_x in [-radius + cell/2, radius - cell/2]
    centers = (np.arange(bins) + 0.5) * (2.0 * radius / bins) - radius
    X, Y = np.meshgrid(centers, centers)  # shape (bins,bins)
    inside = (X * X + Y * Y) <= (radius * radius)
    # Only keep void where inside; outside becomes False
    return void_mask & inside


def bin_index_of_point(
    point_xy: np.ndarray,
    radius: float,
    bins: int,
) -> Tuple[int, int]:
    """
    Utility: map a point (x,y) to (iy, ix) bin index, compatible with compute_void_map.
    Useful later when you want to test "is this edge midpoint in a void bin?".
    """
    x, y = float(point_xy[0]), float(point_xy[1])
    fx = (x + radius) / (2.0 * radius)
    fy = (y + radius) / (2.0 * radius)
    ix = int(np.floor(fx * bins))
    iy = int(np.floor(fy * bins))
    ix = max(0, min(bins - 1, ix))
    iy = max(0, min(bins - 1, iy))
    return iy, ix

import math

def refine_void_mask_with_edges(
    network: nx.Graph,
    voidmap: VoidMap,
    *,
    sample_step: float | None = None,
) -> VoidMap:
    """
    Refine a VoidMap made from node occupancy by removing bins that are crossed by edges.

    Idea:
      - Start from void bins (node_counts <= threshold)
      - If any edge passes through a bin, that bin should NOT be considered void.

    Implementation:
      - Sample points along each edge segment
      - Mark visited bins as edge-covered
      - void_mask := void_mask AND (NOT edge_covered)

    Parameters
    ----------
    network:
        Graph with node["position"].
    voidmap:
        Output of compute_void_map(...).
    sample_step:
        Distance between samples along an edge in coordinate units.
        If None: uses 0.5 * voidmap.cell_size (recommended).

    Returns
    -------
    New VoidMap with updated void_mask (node_counts unchanged).
    """
    R = voidmap.radius
    bins = voidmap.bins
    cell = voidmap.cell_size

    if sample_step is None:
        sample_step = 0.5 * cell
    if sample_step <= 0:
        raise ValueError("sample_step must be > 0.")

    edge_covered = np.zeros((bins, bins), dtype=bool)

    # Preload positions for speed
    pos = {u: network.nodes[u].get("position", None) for u in network.nodes}
    # If some nodes lack positions, skip those edges safely
    for u, v in network.edges():
        pu = pos.get(u, None)
        pv = pos.get(v, None)
        if pu is None or pv is None:
            continue

        pu = np.asarray(pu, dtype=float)
        pv = np.asarray(pv, dtype=float)

        d = pv - pu
        length = float(np.hypot(d[0], d[1]))
        if length == 0:
            iy, ix = bin_index_of_point(pu, R, bins)
            edge_covered[iy, ix] = True
            continue

        # How many samples along the segment?
        n_steps = int(math.ceil(length / sample_step))
        # sample at t = 0..1 inclusive
        for i in range(n_steps + 1):
            t = i / n_steps
            p = pu + t * d
            iy, ix = bin_index_of_point(p, R, bins)
            edge_covered[iy, ix] = True

    new_void_mask = voidmap.void_mask & (~edge_covered)

    return VoidMap(
        radius=voidmap.radius,
        cell_size=voidmap.cell_size,
        bins=voidmap.bins,
        node_counts=voidmap.node_counts,
        void_mask=new_void_mask,
    )

def generate_hex_backfill_in_voids(
    voidmap: VoidMap,
    constraints,
    *,
    hex_length: Optional[float] = None,
    std_rel: float = 0.1,
    keep_edge_if: str = "midpoint",
) -> nx.Graph:
    """
    Generate a background hex network and keep only edges that lie inside void bins.

    Parameters
    ----------
    voidmap:
        VoidMap returned by compute_void_map (optionally refined by edges).
    constraints:
        Your NetworkConstraints object (needed by generate_hex_network).
    hex_length:
        Hex grid characteristic length. If None, uses 2.5 * constraints.min_edge_length.
    std_rel:
        Passed to generate_hex_network (randomness of hex grid).
    keep_edge_if:
        "midpoint" (recommended) or "either_endpoint".
        - midpoint: keep edge if its midpoint is in a void bin
        - either_endpoint: keep edge if either endpoint is in a void bin (more aggressive)

    Returns
    -------
    fill_graph:
        A graph containing only hex edges in void regions.
    """
    if hex_length is None:
        hex_length = 2.5 * constraints.min_edge_length

    # 1) Generate full hex grid across domain (it may already respect mask via constraints)
    hexG = generate_hex_network(length=hex_length, std_rel=std_rel, constraints=constraints)

    # 2) Decide which hex edges to keep based on void bins
    R = voidmap.radius
    bins = voidmap.bins

    keep_edges = []
    positions = {u: np.asarray(hexG.nodes[u]["position"], dtype=float) for u in hexG.nodes}

    for u, v in hexG.edges():
        pu, pv = positions[u], positions[v]

        if keep_edge_if == "midpoint":
            p = 0.5 * (pu + pv)
            iy, ix = bin_index_of_point(p, R, bins)
            if voidmap.void_mask[iy, ix]:
                keep_edges.append((u, v))

        elif keep_edge_if == "either_endpoint":
            iy1, ix1 = bin_index_of_point(pu, R, bins)
            iy2, ix2 = bin_index_of_point(pv, R, bins)
            if voidmap.void_mask[iy1, ix1] or voidmap.void_mask[iy2, ix2]:
                keep_edges.append((u, v))

        else:
            raise ValueError('keep_edge_if must be "midpoint" or "either_endpoint".')

    # 3) Build filtered fill graph
    fillG = nx.Graph()
    # add only nodes touched by kept edges
    used_nodes = set()
    for u, v in keep_edges:
        used_nodes.add(u)
        used_nodes.add(v)

    for u in used_nodes:
        fillG.add_node(u, **hexG.nodes[u])

    fillG.add_edges_from(keep_edges)

    # 4) Remove isolates (just in case)
    isolates = [n for n in fillG.nodes if fillG.degree(n) == 0]
    fillG.remove_nodes_from(isolates)

    return fillG


def merge_graphs_with_relabel(
    mainG: nx.Graph,
    addG: nx.Graph,
    *,
    relabel_prefix: str = "fill",
) -> Tuple[nx.Graph, Dict]:
    """
    Merge addG into mainG safely by relabeling addG node ids.

    Returns
    -------
    mergedG, mapping
        mapping maps old addG node -> new node in mergedG
    """
    merged = mainG.copy()

    # Create new ids that will not collide:
    # Use tuple ids (prefix, old_id) which is collision-proof.
    mapping = {u: (relabel_prefix, u) for u in addG.nodes}
    addG2 = nx.relabel_nodes(addG, mapping, copy=True)

    merged.add_nodes_from(addG2.nodes(data=True))
    merged.add_edges_from(addG2.edges(data=True))

    return merged, mapping


def selective_backfill_step(
    mainG: nx.Graph,
    voidmap: VoidMap,
    constraints,
    *,
    hex_length: Optional[float] = None,
    std_rel: float = 0.1,
    keep_edge_if: str = "midpoint",
    relabel_prefix: str = "fill",
) -> Tuple[nx.Graph, nx.Graph, Dict]:
    """
    Convenience wrapper:
      - create fill graph from hex grid in void bins
      - merge into main graph

    Returns
    -------
    mergedG, fillG, mapping
    """
    fillG = generate_hex_backfill_in_voids(
        voidmap,
        constraints,
        hex_length=hex_length,
        std_rel=std_rel,
        keep_edge_if=keep_edge_if,
    )

    mergedG, mapping = merge_graphs_with_relabel(mainG, fillG, relabel_prefix=relabel_prefix)
    return mergedG, fillG, mapping

def hex_fill_network(
    G: nx.Graph,
    *,
    constraints_main,           # used for voidmap radius/cell scaling (your original constraints)
    constraints_hex=constraints_hex,       # used for hex generation (you used constraints_2)
    cell_factor: float = 4.8,
    void_threshold: int = 0,
    mask_type: str = "square",
    dilate_void_bins: int = 0,  # recommend 0 for your case
    account_for_edges: bool = True,
    hex_length: Optional[float] = 10.0,
    std_rel: float = 0.1,
    keep_edge_if: str = "either_endpoint",
    relabel_prefix: str = "hexfill",
    return_debug: bool = False,
    relabel_network: bool = True,
) -> nx.Graph | Tuple[nx.Graph, nx.Graph, "VoidMap"]:
    """
    Take an arbitrary networkx graph G (nodes must have G.nodes[u]["position"])
    and return a hex-filled graph.

    Parameters
    ----------
    constraints_main:
        The constraints that define the domain extent/min_edge_length for void detection.
    constraints_hex:
        Constraints to generate the hex grid. If None, uses constraints_main.
    cell_factor:
        Bin size = cell_factor * constraints_main.min_edge_length
    dilate_void_bins:
        Usually keep 0; dilation can cause hex to appear too widely.
    hex_length:
        Hex edge length/spacing. If None, set relative to min_edge_length.
    return_debug:
        If True, also returns (fillG, voidmap).

    Returns
    -------
    G_filled
    or (G_filled, fillG, voidmap)
    """
    if constraints_hex is None:
        constraints_hex = constraints_main

    R = constraints_main.max_extent / 2
    cell = cell_factor * constraints_main.min_edge_length

    voidmap = compute_void_map(
        G,
        radius=R,
        cell_size=cell,
        void_threshold=void_threshold,
        mask_type=mask_type,
        dilate_void_bins=dilate_void_bins,
        account_for_edges=account_for_edges,
    )

    # choose a default hex length if not provided
    if hex_length is None:
        hex_length = 10.0

    G_filled, fillG, _ = selective_backfill_step(
        G,
        voidmap,
        constraints=constraints_hex,
        hex_length=hex_length,
        std_rel=std_rel,
        keep_edge_if=keep_edge_if,
        relabel_prefix=relabel_prefix,
    )

    if relabel_network:
        G_filled = _relabel_network(G_filled)

    if return_debug:
        return G_filled, fillG, voidmap
    return G_filled
