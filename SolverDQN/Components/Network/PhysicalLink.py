from dataclasses import dataclass

@dataclass
class PhysicalLink:
    id: int
    source: str
    target: str
    bandwidth: float
