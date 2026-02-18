from typing import Dict, Any
from datetime import datetime

class NetworkMetadata:
    """
    Class to store metadata of a network
    """

    generation_method: str
    generation_parameters: Dict[str, Any]
    network_closed: bool

    def __init__(
        self,
        generation_method: str,
        network_closed: bool,
        generation_parameters: Dict[str, Any] = {},
    ):
        self.generation_method = generation_method
        self.generation_parameters = generation_parameters
        self.network_closed = network_closed

    def to_dict(self) -> dict:
        return {
            "generation_method": self.generation_method,
            "network_closed": self.network_closed,
            "generatrion_parameters": self.generation_parameters,
        }
