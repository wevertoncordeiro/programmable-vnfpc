import re

import numpy
import torch

from Components.Network.SFCRequest import SFCRequest
from Components.Network.VirtualFunction import VirtualFunction
from Components.Network.VirtualLink import VirtualLink
from Components.Network.VirtualSwitch import VirtualSwitch


class NetworkParser:

    def __init__(self):
        self.endpoints = []
        self.physical_switches = []
        self.npops = []
        self.links = []
        self.sfc_requests = []
        self.link_counter = 0

    def parse_file(self, filename: str):
        """Parse network definition from file"""
        with open(filename, 'r') as f:
            content = f.read()
        return self.parse_text(content)

    def parse_text(self, text: str):
        """Parse network definition from text"""
        lines = text.strip().split('\n')
        current_sfc = None
        current_sfc_data = {}
        global_vs_stages = {}

        for line in lines:
            line = re.sub(r'#.*$', '', line).strip()
            if not line:
                continue

            parts = line.split()
            if not parts:
                continue

            command = parts[0].lower()

            if parts[0] in ['vs_int', 'vs_scion'] and not current_sfc:
                vs_type = parts[0]
                if len(parts) >= 3 and parts[1].lower() == 'stage_demand':
                    stage_demand = int(parts[2])
                    global_vs_stages[vs_type] = stage_demand
                continue

            if command == 'sfc':
                if current_sfc:
                    self._build_sfc(current_sfc, current_sfc_data)

                current_sfc = parts[1]
                current_sfc_data = {
                    'endpoints': [],
                    'vnfs': [],
                    'vlinks': [],
                    'vs_stages': global_vs_stages.copy()
                }

            elif current_sfc:
                if command == 'endpoint':
                    current_sfc_data['endpoints'] = parts[1:]

                elif command == 'virtual_function':
                    name = parts[1]
                    cpu = float(parts[2])
                    mem = float(parts[3])
                    assigned = parts[4] if len(parts) > 4 else None
                    current_sfc_data['vnfs'].append((name, cpu, mem, assigned))

                elif command == 'virtual_link':
                    src = parts[1]
                    tgt = parts[2]
                    bw = float(parts[3])
                    vs_type = parts[4] if len(parts) > 4 else None
                    current_sfc_data['vlinks'].append((src, tgt, bw, vs_type))

                elif parts[0] in ['vs_int', 'vs_scion']:
                    vs_type = parts[0]
                    if len(parts) >= 3 and parts[1].lower() == 'stage_demand':
                        stage_demand = int(parts[2])
                        current_sfc_data['vs_stages'][vs_type] = stage_demand

            else:
                if command == 'topology':
                    continue

                elif command == 'endpoint':
                    self.endpoints.extend(parts[1:])

                elif command == 'physical_switch':
                    name = parts[1]
                    capacity = int(parts[2])
                    self.physical_switches.append((name, capacity))

                elif command == 'network_pop':
                    name = parts[1]
                    cpu = float(parts[2])
                    mem = float(parts[3])
                    self.npops.append((name, cpu, mem))

                elif command == 'link':
                    src = parts[1]
                    tgt = parts[2]
                    bw = float(parts[3])
                    self.link_counter += 1
                    self.links.append((self.link_counter, src, tgt, bw))

        if current_sfc:
            self._build_sfc(current_sfc, current_sfc_data)

        return self

    def _build_sfc(self, sfc_name: str, sfc_data: dict):
        """Build SFC request from parsed data"""
        vnfs = []
        for name, cpu, mem, assigned in sfc_data['vnfs']:
            vnfs.append(VirtualFunction(name, cpu, mem, assigned))

        vs_types = set()
        for _, _, _, vs_type in sfc_data['vlinks']:
            if vs_type:
                vs_types.add(vs_type)

        virtual_switches = []
        for vs_type in vs_types:
            stage_demand = sfc_data['vs_stages'].get(vs_type, 1)
            virtual_switches.append(VirtualSwitch(vs_type, stage_demand))

        virtual_links = []
        for src, tgt, bw, vs_type in sfc_data['vlinks']:
            virtual_links.append(VirtualLink(src, tgt, bw, vs_type, []))

        sfc = SFCRequest(
            name=sfc_name,
            endpoints=sfc_data['endpoints'],
            vnfs=vnfs,
            virtual_switches=virtual_switches,
            virtual_links=virtual_links
        )

        self.sfc_requests.append(sfc)
