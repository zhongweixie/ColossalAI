from dataclasses import dataclass
from typing import List


class ModelParameters:
    param_idx = 0
    fp16_params = []
    fp32_master_params = []

@dataclass
class NodeInfo:
    has_param: bool = False
    param_size: float = 0
    offload_param_flag: bool = False
    param_indices: List = None
    runtime_fwd_mem: float = 0
    runtime_bwd_mem: float = 0
