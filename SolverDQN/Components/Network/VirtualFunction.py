from dataclasses import dataclass

@dataclass
class VirtualFunction:
    name: str
    cpu_demand: float
    mem_demand: float
    assigned_npop: str = None
