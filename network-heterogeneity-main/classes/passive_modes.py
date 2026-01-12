from classes.netsalt_config import NetsaltConfig


class PassiveModes:
    def __init__(
        self,
        config: NetsaltConfig,
        quantum_graph_without_pump,
        qualities,
        passive_modes,
    ):
        self.config = config
        self.quantum_graph_without_pump = quantum_graph_without_pump
        self.qualities = qualities
        self.passive_modes = passive_modes
