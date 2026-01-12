from dataclasses import dataclass, field
from typing import Optional, Tuple
from classes.netsalt_config import NetsaltConfig, default_config_1
from classes.job_status import JobStatus
import uuid
import copy


@dataclass
class NetsaltJob:
    network_id: str
    pattern_id: Optional[str]
    pattern_idx: Optional[int]
    pattern_size: Optional[Tuple[int, int]]
    
    job_type: str

    needs_trajectory: bool = False
    status: JobStatus = "pending"
    error_message: Optional[str] = None
    execution_time: Optional[float] = None

    config: NetsaltConfig = field(default_factory=lambda: copy.deepcopy(default_config_1))

    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __str__(self):
        if self.pattern_size is None:
            size_str = "None"
        else:
            size_str = f"{self.pattern_size[0]}x{self.pattern_size[1]}"

        return (
            f"NetsaltJob(network_id={self.network_id}, "
            f"pattern_id={self.pattern_id}, "
            f"pattern_idx={self.pattern_idx}, "
            f"type={self.job_type}, "
            f"size={size_str}, "
            f"status={self.status})"
        )
