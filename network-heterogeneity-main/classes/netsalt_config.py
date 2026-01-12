import json


class CreateQuantumGraphConfig:
    def __init__(self, **kwargs):
        self.graph_mode = kwargs.get("graph_mode", "open")
        self.inner_total_length = kwargs.get("inner_total_length", None)
        self.dielectric_mode = kwargs.get("dielectric_mode", "refraction_params")
        self.method = kwargs.get("method", "uniform")
        self.inner_value = kwargs.get("inner_value", 1.5)
        self.loss = kwargs.get("loss", 0.005)
        self.outer_value = kwargs.get("outer_value", 1.0)
        self.edge_size = kwargs.get("edge_size", 0.1)
        self.k_a = kwargs.get("k_a", 15.0)  # Center of the lorentzian pump
        self.gamma_perp = kwargs.get("gamma_perp", 3.0)  # Width of the lorentzian pump
        self.node_loss = kwargs.get("node_loss", 0.0)
        self.noise_level = kwargs.get("noise_level", 0.001)
        self.keep_degree_two = kwargs.get("keep_degree_two", True)
        self.max_extent = kwargs.get("max_extent", None)


class ModeSearchConfig:
    def __init__(self, **kwargs):
        self.k_n = kwargs.get("k_n", 100)
        self.k_min = kwargs.get("k_min", 10)
        self.k_max = kwargs.get("k_max", 12)
        self.alpha_n = kwargs.get("alpha_n", 100)
        self.alpha_min = kwargs.get("alpha_min", 0.0)
        self.alpha_max = kwargs.get("alpha_max", 0.1)
        self.quality_threshold_passive_modes = kwargs.get(
            "quality_threshold_passive_modes", 1e-4
        )
        self.quality_threshold = kwargs.get("quality_threshold", 1e-3)
        self.search_stepsize = kwargs.get("search_stepsize", 0.001)
        self.max_steps = kwargs.get("max_steps", 1000)
        self.max_tries_reduction = kwargs.get("max_tries_reduction", 50)
        self.reduction_factor = kwargs.get("reduction_factor", 1.0)
        self.quality_method = kwargs.get("quality_method", "eigenvalue")
        self.min_distance = kwargs.get("min_distance", 2)
        self.threshold_abs = kwargs.get("threshold_abs", 0.1)
        self.new_D0_method = kwargs.get("new_D0_method", "linear_approx")
        self.kill_modes = kwargs.get("kill_modes", True)


class PumpConfig:
    def __init__(self, **kwargs):
        self.D0_max = kwargs.get("D0_max", 0.05)
        self.D0_steps = kwargs.get("D0_steps", 10)


class ComputeModalIntensitiesConfig:
    def __init__(self, **kwargs):
        self.D0_max = kwargs.get("D0_max", 0.1)


class ComputeModeTrajectoriesConfig:
    def __init__(self, **kwargs):
        self.skip = kwargs.get("skip", False)


class NetsaltConfig:
    def __init__(self, **kwargs):
        self.create_quantum_graph = CreateQuantumGraphConfig(
            **kwargs.get("create_quantum_graph", {})
        )
        self.mode_search_config = ModeSearchConfig(
            **kwargs.get("mode_search_config", {})
        )
        self.pump_config = PumpConfig(**kwargs.get("pump_config", {}))
        self.compute_modal_intensities = ComputeModalIntensitiesConfig(
            **kwargs.get("compute_modal_intensities", {})
        )
        self.compute_mode_trajectories = ComputeModeTrajectoriesConfig(
            **kwargs.get("compute_mode_trajectories", {})
        )

    def to_dict(self):
        return {
            "create_quantum_graph": vars(self.create_quantum_graph),
            "mode_search_config": vars(self.mode_search_config),
            "pump_config": vars(self.pump_config),
            "compute_modal_intensities": vars(self.compute_modal_intensities),
            "compute_mode_trajectories": vars(self.compute_mode_trajectories),
        }

    def __eq__(self, value):
        return self.to_dict() == value.to_dict()

    def to_json(self):
        return json.dumps(self.to_dict())

    @staticmethod
    def from_json(json_str):
        return NetsaltConfig(**json.loads(json_str))


default_config_1 = NetsaltConfig(
    create_quantum_graph={
        # "max_extent": 50,
        "inner_value": 3.4,
        "loss": 0.001,  # TODO: check this value
        "outer_value": 1.0,
        "edge_size": 10,
        "k_a": 7.0,
        "gamma_perp": 0.35,
        "noise_level": 0.0,
        "keep_degree_two": True,
    },
    mode_search_config={
        "k_n": 4000,
        "k_min": 6.8,
        "k_max": 7.2,
        "alpha_n": 100,
        "alpha_min": 0,
        "alpha_max": 0.01,
        "quality_threshold_passive_modes": 1e-5,
        "quality_threshold": 1e-4,  # TODO: check this values
        "search_stepsize": 0.000005,
        "max_steps": 10000,
        "max_tries_reduction": 50,
        "reduction_factor": 0.8,
    },
    pump_config={"D0_max": 0.03, "D0_steps": 14},
)