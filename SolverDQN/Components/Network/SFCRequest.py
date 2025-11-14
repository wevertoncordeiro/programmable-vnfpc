from typing import List
from dataclasses import dataclass

from Components.Network import VirtualLink, VirtualFunction, VirtualSwitch


@dataclass
class SFCRequest:
    name: str
    endpoints: List[str]
    vnfs: List[VirtualFunction]
    virtual_switches: List[VirtualSwitch]
    virtual_links: List[VirtualLink]
