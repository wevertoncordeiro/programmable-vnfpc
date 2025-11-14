from typing import List
from dataclasses import dataclass

@dataclass
class VirtualLink:
    source: str
    target: str
    bandwidth_demand: float
    vs_type: str
    route: List[int]
