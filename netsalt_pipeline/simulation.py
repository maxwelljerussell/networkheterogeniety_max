from typing import Optional, Sequence, Tuple, List
import time
import numpy as np
import networkx as nx
import pandas as pd
import netsalt
from netsalt import FindThresholdLasingModesException

from classes.spectrum import Spectrum
from classes.netsalt_config import NetsaltConfig
from classes.passive_modes import PassiveModes
from classes.lasing_modes import LasingModes
from classes.pattern import Pattern

from helpers.pixelation import pixelate_network

def dbg(msg, debug:bool):
    if debug:
        print(f"[netSALT DEBUG] {msg}")

def compute_pump_profile(network: nx.Graph, pattern: Pattern):
    pixel_assignment = network.graph["pixel_assignment"]
    weights = np.zeros(len(network.edges))

    for x in range(pattern.width):
        for y in range(pattern.height):
            for edge in pixel_assignment[y][x]:
                weights[edge] = pattern[y][x]

    return weights

def _update_mode_search_config(
    qg,
    config: NetsaltConfig,
    for_passive_modes: bool = False,
    n_workers: int = 1,
):
    """Attach mode-search parameters to qg.graph['params']."""
    mode_search_config = config.mode_search_config
    qg.graph.setdefault("params", {})
    qg.graph["params"].update(
        {
            "n_workers": n_workers,
            "k_n": mode_search_config.k_n,
            "k_min": mode_search_config.k_min,
            "k_max": mode_search_config.k_max,
            "alpha_n": mode_search_config.alpha_n,
            "alpha_min": mode_search_config.alpha_min,
            "alpha_max": mode_search_config.alpha_max,
            "quality_threshold": (
                mode_search_config.quality_threshold_passive_modes
                if for_passive_modes
                else mode_search_config.quality_threshold
            ),
            "max_steps": mode_search_config.max_steps,
            "max_tries_reduction": mode_search_config.max_tries_reduction,
            "reduction_factor": mode_search_config.reduction_factor,
            "search_stepsize": mode_search_config.search_stepsize,
        }
    )
    return qg


def _update_config_with_pump(
    graph,
    pump: np.ndarray,
    config: NetsaltConfig,
    for_passive_modes: bool = False,
    n_workers: int = 1,
):
    """
    Attach mode-search parameters and pump-related parameters
    (D0_max, D0_steps, pump) to the graph.
    """
    qg = _update_mode_search_config(
        graph, 
        config, 
        for_passive_modes=for_passive_modes,
        n_workers=n_workers,
    )
    qg.graph.setdefault("params", {})
    qg.graph["params"].update(
        {
            "D0_max": config.pump_config.D0_max,
            "D0_steps": config.pump_config.D0_steps,
            "pump": np.array(pump),
        }
    )
    return qg


def create_quantum_graph(
        input_graph,
        config: NetsaltConfig,
        refraction_index: Optional[np.ndarray] = None,
):
    """
    Turn a plain network graph (with node['position']) into a netsalt
    quantum graph using parameters stored in config.create_quantum_graph.
    """

    cg = config.create_quantum_graph

    params = {
        "open_model": cg.graph_mode,
        cg.dielectric_mode: {
            "method": cg.method,
            "inner_value": cg.inner_value,
            "loss": cg.loss,
            "outer_value": cg.outer_value,
        },
        "node_loss": cg.node_loss,
        "plot_edgesize": cg.edge_size,
        "k_a": cg.k_a,
        "gamma_perp": cg.gamma_perp,
    }

    qg = input_graph.copy()

    if cg.method != "custom" and not cg.keep_degree_two:
        qg = netsalt.quantum_graph.simplify_graph(qg)

    positions = np.array([qg.nodes[u]["position"] for u in qg.nodes])

    netsalt.create_quantum_graph(
        qg,
        params,
        positions=positions,
        noise_level=cg.noise_level
    )

    netsalt.set_total_length(
        qg,
        cg.inner_total_length,
        max_extent=cg.max_extent,
        inner=True
    )

    netsalt.set_dielectric_constant(qg, params, custom_values=refraction_index)

    netsalt.set_dispersion_relation(qg, netsalt.physics.dispersion_relation_pump)

    netsalt.update_parameters(qg, params)

    qg = netsalt.oversample_graph(qg, params["plot_edgesize"])

    return qg

def compute_passive_modes(
        input_graph,
        config: NetsaltConfig,
        refraction_index: Optional[np.ndarray] = None,
        debug: bool = False,
) -> PassiveModes:
    """
    1) Build quantum graph without pump
    2) Scan k/alpha space
    3) Extract passive modes with netSALT.find_modes
    """
    dbg("=== COMPUTE PASSIVE MODES ===", debug)
    dbg(f"Input graph: {input_graph.number_of_nodes()} nodes, {input_graph.number_of_edges()} edges", debug)
    t0 = time.time()

    qg = create_quantum_graph(input_graph, config, refraction_index)
    qg = _update_mode_search_config(qg, config, for_passive_modes=True)
    ms = config.mode_search_config
    dbg(f"Quantum graph created in {time.time() - t0:.2f}s", debug)

    dbg("Scanning frequencies...", debug)
    qualities = netsalt.scan_frequencies(
        qg,
        quality_method=ms.quality_method,
    )

    dbg("Finding passive modes...", debug)
    passive_modes = netsalt.find_modes(
        qg,
        qualities,
        quality_method=ms.quality_method,
        min_distance=ms.min_distance,
        threshold_abs=ms.threshold_abs,
    )
    dbg(f"Found {len(passive_modes)} passive modes", debug)

    return PassiveModes(config, qg, qualities, passive_modes)

def _setup_lasing_problem(
        input_graph: nx.Graph,
        passive_modes: PassiveModes,
        pattern: Pattern,
        config: NetsaltConfig,
        refraction_index: Optional[np.ndarray] = None,
):
    """
    1. Pixelate network to pattern resolution.
    2. Compute pump from pattern.
    3. Build quantum graph with pump.
    4. Find threshold modes.
    5. Compute mode competition matrix.

    Returns
    -------
    quantum_graph_with_pump : nx.Graph
    pump : np.ndarray
    threshold_modes : object
    competition_matrix : np.ndarray or None
    """
    pixelated_graph = pixelate_network(
        input_graph,
        width=pattern.width,
        height=pattern.height,
    )

    pump = compute_pump_profile(pixelated_graph, pattern)

    graph_with_pump = pixelated_graph.copy()
    graph_with_pump.graph.setdefault("params", {})
    graph_with_pump.graph["params"]["pump"] = pump

    qg = create_quantum_graph(
        graph_with_pump,
        config=config,
        refraction_index=refraction_index,
    )

    qg = _update_config_with_pump(
        qg,
        pump=pump,
        config=config,
        for_passive_modes=False,
    )

    ms = config.mode_search_config
    mode_trajectories = passive_modes.passive_modes

    try:
        threshold_modes = netsalt.find_threshold_lasing_modes(
            mode_trajectories,
            qg,
            quality_method=ms.quality_method,
            config={
                "new_D0_method": ms.new_D0_method,
                "kill_modes": ms.kill_modes,
            },
        )
    except FindThresholdLasingModesException as e:
        raise e #TODO: add logging

    min_threshold = np.min(threshold_modes["lasing_thresholds"])
    if np.isinf(min_threshold):
        return qg, pump, threshold_modes, None

    competition_matrix = netsalt.compute_mode_competition_matrix(qg, threshold_modes)

    return qg, pump, threshold_modes, competition_matrix

def compute_lasing_modes(
    input_graph: nx.Graph,
    passive_modes: PassiveModes,
    pattern: Pattern,
    config: NetsaltConfig,
    refraction_index: Optional[np.ndarray] = None,
    debug: bool = False,
) -> LasingModes:
    """
    Compute lasing modes at a single D0:
      D0 = config.compute_modal_intensities.D0_max
    """
    dbg("=== COMPUTE LASING MODES ===", debug)
    dbg("Setting up lasing problem...", debug)
    qg, pump, threshold_modes, competition_matrix = _setup_lasing_problem(
        input_graph=input_graph,
        passive_modes=passive_modes,
        pattern=pattern,
        config=config,
        refraction_index=refraction_index,
    )

    thresholds = threshold_modes["lasing_thresholds"]
    dbg(f"Min threshold: {np.min(thresholds):.4e}", debug)
    dbg(f"Max threshold: {np.max(thresholds):.4e}", debug)

    # no lasing at all → NaN intensities
    if competition_matrix is None:
        lasing_modes_dict = threshold_modes
        lasing_modes_dict["modal_intensities"] = np.full(
            len(lasing_modes_dict), np.nan
        )
        return LasingModes(
            config=config,
            pattern=pattern,
            pump=pump,
            refraction_index=refraction_index,
            quantum_graph_with_pump=qg,
            threshold_modes=threshold_modes,
            competition_matrix=None,
            lasing_modes=lasing_modes_dict,
        )

    D0 = config.compute_modal_intensities.D0_max

    lasing_modes_dict = netsalt.compute_modal_intensities(
        threshold_modes,
        D0,
        competition_matrix,
    )
    dbg(f"Total intensity at D0={D0:.4f}: {np.nansum(lasing_modes_dict['modal_intensities']):.4e}", debug)

    return LasingModes(
        config=config,
        pattern=pattern,
        pump=pump,
        refraction_index=refraction_index,
        quantum_graph_with_pump=qg,
        threshold_modes=threshold_modes,
        competition_matrix=competition_matrix,
        lasing_modes=lasing_modes_dict,
        D0=D0,
    )

def compute_lasing_modes_sweep(
    input_graph: nx.Graph,
    passive_modes,
    pattern: Pattern,
    config: NetsaltConfig,
    refraction_index=None,
    D0_max=None,
    debug=False,
) -> LasingModes:

    dbg("=== COMPUTE LASING MODES SWEEP ===", debug)

    qg, pump, threshold_modes, competition_matrix = _setup_lasing_problem(
        input_graph=input_graph,
        passive_modes=passive_modes,
        pattern=pattern,
        config=config,
        refraction_index=refraction_index,
    )
    if D0_max is None:
        D0_max = config.compute_modal_intensities.D0_max

    if competition_matrix is None:
        n_modes = len(threshold_modes["lasing_thresholds"])
        n_steps = max(config.pump_config.D0_steps, 2)
        D0_list = np.linspace(0.0, float(D0_max), n_steps)

        lasing_modes_dict = threshold_modes.copy()
        lasing_modes_dict["modal_intensities"] = pd.DataFrame(
            np.full((n_modes, len(D0_list)), np.nan),
            columns=D0_list,
        )

        return LasingModes(
            config=config,
            pattern=pattern,
            pump=pump,
            refraction_index=refraction_index,
            quantum_graph_with_pump=qg,
            threshold_modes=threshold_modes,
            competition_matrix=None,
            lasing_modes=lasing_modes_dict,
            D0=D0_list,
        )

    modes_df = threshold_modes.copy()
    modes_swept = netsalt.compute_modal_intensities(
        modes_df,
        D0_max,
        competition_matrix,
    )

    MI = modes_swept["modal_intensities"]       # shape (n_modes, n_D0)
    D0_list = np.array(MI.columns, dtype=float) # list of pump powers

    # RETURN ONE OBJECT — SAME CLASS
    return LasingModes(
        config=config,
        pattern=pattern,
        pump=pump,
        refraction_index=refraction_index,
        quantum_graph_with_pump=qg,
        threshold_modes=threshold_modes,
        competition_matrix=competition_matrix,
        lasing_modes=modes_swept,   # contains ALL intensities
        D0=D0_list,                 # contains FULL pump sweep
    )

def sweep_d0_spectra(
    input_graph: nx.Graph,
    passive_modes: PassiveModes,
    pattern: Pattern,
    config: NetsaltConfig,
    refraction_index: Optional[np.ndarray] = None,
    D0_values: Optional[Sequence[float]] = None,
    only_lasing_modes: bool = False,
    debug: bool = False,
) -> Tuple[np.ndarray, List[Spectrum]]:
    """
    Sweep D0 and return a Spectrum for each D0.

    1) Setup lasing problem once (pixelation + pump + qg + thresholds + competition).
    2) For each D0 in D0_values, compute modal intensities.
    3) Wrap into LasingModes and call get_spectrum().
    """
    dbg("=== D0 SWEEP START ===", debug)

    if D0_values is None:
        n_steps = max(config.pump_config.D0_steps, 2)
        D0_values = np.linspace(
            0.0,
            config.compute_modal_intensities.D0_max,
            n_steps,
        )
    D0_array = np.asarray(D0_values, dtype=float)

    dbg(f"D0 sweep range: {D0_array[0]:.4f} → {D0_array[-1]:.4f}", debug)
    dbg(f"Number of D0 points: {len(D0_array)}", debug)
    dbg("Setting up lasing problem (thresholds + competition)...", debug)
    
    qg, pump, threshold_modes, competition_matrix = _setup_lasing_problem(
        input_graph=input_graph,
        passive_modes=passive_modes,
        pattern=pattern,
        config=config,
        refraction_index=refraction_index,
    )

    dbg(f"Pump min/max: {pump.min():.3e} / {pump.max():.3e}", debug)

    if competition_matrix is None:
        dbg("No finite thresholds found → no lasing possible.", debug)
    
    spectra: List[Spectrum] = []

    if competition_matrix is None:
        for D0 in D0_array:
            dbg(f"D0={D0:.4f} → no lasing (all NaN)", debug)
            lasing_modes_dict = threshold_modes.copy()
            lasing_modes_dict["modal_intensities"] = np.full(
                len(lasing_modes_dict), np.nan
            )
            lm = LasingModes(
                config=config,
                pattern=pattern,
                pump=pump,
                refraction_index=refraction_index,
                quantum_graph_with_pump=qg,
                threshold_modes=threshold_modes,
                competition_matrix=None,
                lasing_modes=lasing_modes_dict,
            )
            spectra.append(lm.get_spectrum(only_lasing_modes=only_lasing_modes))
        dbg("=== D0 SWEEP COMPLETE ===", debug)
        return D0_array, spectra

    for i, D0 in enumerate(D0_array):
        lasing_modes_dict = netsalt.compute_modal_intensities(
            threshold_modes,
            D0,
            competition_matrix,
        )

        mi = lasing_modes_dict["modal_intensities"]          # DataFrame (n_modes, n_cols)
        last_col = mi.columns[-1]                            # e.g. pump intensity closest to D0
        total_I = np.nansum(mi[last_col])                    # sum over modes at that D0 only
        dbg(
            f"[{i+1}/{len(D0_array)}] D0={D0:.4f} → Total Intensity = {total_I:.4e}",
            debug,
        )

        lm = LasingModes(
            config=config,
            pattern=pattern,
            pump=pump,
            refraction_index=refraction_index,
            quantum_graph_with_pump=qg,
            threshold_modes=threshold_modes,
            competition_matrix=competition_matrix,
            lasing_modes=lasing_modes_dict,
        )
        spec = lm.get_spectrum(only_lasing_modes=only_lasing_modes)
        spectra.append(spec)
    dbg("=== D0 SWEEP COMPLETE ===", debug)
    return D0_array, spectra
