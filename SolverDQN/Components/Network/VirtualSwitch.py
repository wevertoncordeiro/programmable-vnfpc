from dataclasses import dataclass

@dataclass
class VirtualSwitch:
    name: str
    stage_demand: int = 1
