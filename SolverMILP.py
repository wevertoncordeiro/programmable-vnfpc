#!/usr/bin/env python3
"""
Service Function Chaining (SFC) Deployment Solver with Text Parser
Accepts network topology and SFC definitions in text format
"""

objective_choices=['Maximize_SFCs', 'Maximize_SFCs_Minimize_Resources']
DEFAULT_OBJECTIVE = objective_choices[1]


try:
    import traceback
    import sys
    import argparse
    import pulp
    from typing import Dict, List, Tuple, Set
    from dataclasses import dataclass, asdict
    from collections import defaultdict
    import re
    import json
    import time
    import os


except Exception as e:
    print(f"\n✗ ERROR: {e}")
    traceback.print_exc()
    print("python3 -m venv venv")
    print("source venv/bin/activate")
    print("pip install -r requirements.txt")



@dataclass
class PhysicalSwitch:
    name: str
    stage_capacity: int


@dataclass
class NetworkPoP:
    name: str
    cpu_capacity: float
    mem_capacity: float


@dataclass
class PhysicalLink:
    id: int
    source: str
    target: str
    bandwidth: float


@dataclass
class VirtualFunction:
    name: str
    cpu_demand: float
    mem_demand: float
    assigned_npop: str = None


@dataclass
class VirtualSwitch:
    name: str
    stage_demand: int = 1


@dataclass
class VirtualLink:
    source: str
    target: str
    bandwidth_demand: float
    vs_type: str
    route: List[int]


@dataclass
class SFCRequest:
    name: str
    endpoints: List[str]
    vnfs: List[VirtualFunction]
    virtual_switches: List[VirtualSwitch]
    virtual_links: List[VirtualLink]


class NetworkParser:
    """Parser for network topology and SFC definitions"""

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
        global_vs_stages = {}  # Store global stage_demand definitions

        for line in lines:
            # Remove comments and strip whitespace
            line = re.sub(r'#.*$', '', line).strip()
            if not line:
                continue

            parts = line.split()
            if not parts:
                continue

            command = parts[0].lower()

            # Handle global VS stage_demand definitions (outside SFC blocks)
            if parts[0] in ['vs_int', 'vs_scion'] and not current_sfc:
                vs_type = parts[0]
                if len(parts) >= 3 and parts[1].lower() == 'stage_demand':
                    stage_demand = int(parts[2])
                    global_vs_stages[vs_type] = stage_demand
                continue

            # Check if we're in SFC definition mode
            if command == 'sfc':
                # Save previous SFC if exists
                if current_sfc:
                    self._build_sfc(current_sfc, current_sfc_data)

                # Start new SFC
                current_sfc = parts[1]
                current_sfc_data = {
                    'endpoints': [],
                    'vnfs': [],
                    'vlinks': [],
                    'vs_stages': global_vs_stages.copy()  # Initialize with global values
                }

            elif current_sfc:
                # We're inside an SFC definition
                if command == 'endpoint':
                    current_sfc_data['endpoints'] = parts[1:]

                elif command == 'virtual_function':
                    # virtual_function fw 0.1 0.1 [npop1]
                    name = parts[1]
                    cpu = float(parts[2])
                    mem = float(parts[3])
                    assigned = parts[4] if len(parts) > 4 else None
                    current_sfc_data['vnfs'].append((name, cpu, mem, assigned))

                elif command == 'virtual_link':
                    # virtual_link A fw 10 vs_int
                    src = parts[1]
                    tgt = parts[2]
                    bw = float(parts[3])
                    vs_type = parts[4] if len(parts) > 4 else None
                    current_sfc_data['vlinks'].append((src, tgt, bw, vs_type))

                elif parts[0] in ['vs_int', 'vs_scion']:
                    # Override global stage_demand for this specific SFC
                    vs_type = parts[0]
                    if len(parts) >= 3 and parts[1].lower() == 'stage_demand':
                        stage_demand = int(parts[2])
                        current_sfc_data['vs_stages'][vs_type] = stage_demand

            else:
                # We're in topology definition mode
                if command == 'topology':
                    continue

                elif command == 'endpoint':
                    # endpoint e1 e2 e3 ...
                    self.endpoints.extend(parts[1:])

                elif command == 'physical_switch':
                    # physical_switch ps1 20
                    name = parts[1]
                    capacity = int(parts[2])
                    self.physical_switches.append((name, capacity))

                elif command == 'network_pop':
                    # network_pop npop1 1 1
                    name = parts[1]
                    cpu = float(parts[2])
                    mem = float(parts[3])
                    self.npops.append((name, cpu, mem))

                elif command == 'link':
                    # link e1 ps1 500
                    src = parts[1]
                    tgt = parts[2]
                    bw = float(parts[3])
                    self.link_counter += 1
                    self.links.append((self.link_counter, src, tgt, bw))

        # Save last SFC if exists
        if current_sfc:
            self._build_sfc(current_sfc, current_sfc_data)

        return self

    def _build_sfc(self, sfc_name: str, sfc_data: dict):
        """Build SFC request from parsed data"""
        # Create VNFs
        vnfs = []
        for name, cpu, mem, assigned in sfc_data['vnfs']:
            vnfs.append(VirtualFunction(name, cpu, mem, assigned))

        # Collect VS types from virtual links
        vs_types = set()
        for _, _, _, vs_type in sfc_data['vlinks']:
            if vs_type:
                vs_types.add(vs_type)

        # Create virtual switches with stage_demand from vs_stages or default to 1
        virtual_switches = []
        for vs_type in vs_types:
            stage_demand = sfc_data['vs_stages'].get(vs_type, 1)  # Default to 1 if not specified
            virtual_switches.append(VirtualSwitch(vs_type, stage_demand))

        # Create virtual links
        virtual_links = []
        for src, tgt, bw, vs_type in sfc_data['vlinks']:
            virtual_links.append(VirtualLink(src, tgt, bw, vs_type, []))

        # Create SFC request
        sfc = SFCRequest(
            name=sfc_name,
            endpoints=sfc_data['endpoints'],
            vnfs=vnfs,
            virtual_switches=virtual_switches,
            virtual_links=virtual_links
        )

        self.sfc_requests.append(sfc)

    def apply_to_solver(self, solver):
        """Apply parsed configuration to solver"""
        # Add endpoints
        for ep in self.endpoints:
            solver.add_endpoint(ep)

        # Add physical switches
        for name, capacity in self.physical_switches:
            solver.add_physical_switch(name, capacity)

        # Add network PoPs
        for name, cpu, mem in self.npops:
            solver.add_npop(name, cpu, mem)

        # Add physical links
        for link_id, src, tgt, bw in self.links:
            solver.add_physical_link(link_id, src, tgt, bw)

        # Add SFC requests
        for sfc in self.sfc_requests:
            solver.add_sfc_request(sfc)

        return solver

class SFCILPSolver:
    """Solver for Service Function Chaining deployment using ILP"""

    def __init__(self, solver, objective):
        self.endpoints: Set[str] = set()
        self.physical_switches: Dict[str, PhysicalSwitch] = {}
        self.npops: Dict[str, NetworkPoP] = {}
        self.physical_links: Dict[int, PhysicalLink] = {}
        self.link_by_nodes: Dict[Tuple[str, str], PhysicalLink] = {}
        self.sfc_requests: List[SFCRequest] = []
        self.vs_types: Set[str] = set()
        self.solver = solver
        self.objective = objective

    def add_endpoint(self, name: str):
        """Add physical endpoint"""
        self.endpoints.add(name)

    def add_physical_switch(self, name: str, stage_capacity: int):
        """Add programmable physical switch"""
        self.physical_switches[name] = PhysicalSwitch(name, stage_capacity)

    def add_npop(self, name: str, cpu: float, mem: float):
        """Add Network Point of Presence"""
        self.npops[name] = NetworkPoP(name, cpu, mem)

    def add_physical_link(self, link_id: int, source: str, target: str, bandwidth: float):
        """Add bidirectional physical link"""
        # Forward direction
        link_fwd = PhysicalLink(link_id, source, target, bandwidth)
        self.physical_links[link_id] = link_fwd
        self.link_by_nodes[(source, target)] = link_fwd

        # Backward direction
        link_bwd = PhysicalLink(-link_id, target, source, bandwidth)
        self.physical_links[-link_id] = link_bwd
        self.link_by_nodes[(target, source)] = link_bwd

    def add_sfc_request(self, request: SFCRequest):
        """Add SFC request"""
        self.sfc_requests.append(request)
        for vs in request.virtual_switches:
            self.vs_types.add(vs.name)

    def get_all_nodes(self) -> List[str]:
        """Get all physical nodes"""
        return list(self.endpoints) + list(self.physical_switches.keys()) + list(self.npops.keys())

    def get_outgoing_links(self, node: str) -> List[PhysicalLink]:
        """Get outgoing links from a node"""
        return [link for link in self.physical_links.values() if link.source == node]

    def get_incoming_links(self, node: str) -> List[PhysicalLink]:
        """Get incoming links to a node"""
        return [link for link in self.physical_links.values() if link.target == node]

    def calculate_residual_capacity(self, results: Dict) -> Dict:
        """Calculate residual capacity for all resources"""
        residual = {
            'npops': {},
            'physical_switches': {},
            'physical_links': {}
        }

        # Calculate NPoP residual capacity
        for npop_name, npop in self.npops.items():
            cpu_used = 0
            mem_used = 0

            for q in self.sfc_requests:
                if results['sfc_acceptance'].get(q.name, False):
                    for vf in q.vnfs:
                        assigned_npop = results['vnf_assignments'].get((q.name, vf.name))
                        if assigned_npop == npop_name:
                            cpu_used += vf.cpu_demand
                            mem_used += vf.mem_demand

            residual['npops'][npop_name] = {
                'cpu_capacity': npop.cpu_capacity,
                'cpu_used': cpu_used,
                'cpu_residual': npop.cpu_capacity - cpu_used,
                'cpu_utilization_percent': (cpu_used / npop.cpu_capacity * 100) if npop.cpu_capacity > 0 else 0,
                'mem_capacity': npop.mem_capacity,
                'mem_used': mem_used,
                'mem_residual': npop.mem_capacity - mem_used,
                'mem_utilization_percent': (mem_used / npop.mem_capacity * 100) if npop.mem_capacity > 0 else 0
            }

        # Calculate Physical Switch residual capacity
        for ps_name, ps in self.physical_switches.items():
            stages_used = 0

            for q in self.sfc_requests:
                if results['sfc_acceptance'].get(q.name, False):
                    for vs in q.virtual_switches:
                        vs_key = f"{q.name}_{vs.name}"
                        assignments = results['vs_assignments'].get(vs_key, [])
                        if ps_name in assignments:
                            stages_used += vs.stage_demand

            residual['physical_switches'][ps_name] = {
                'stage_capacity': ps.stage_capacity,
                'stages_used': stages_used,
                'stages_residual': ps.stage_capacity - stages_used,
                'utilization_percent': (stages_used / ps.stage_capacity * 100) if ps.stage_capacity > 0 else 0
            }

        # Calculate Physical Link residual bandwidth
        for link_id, link in self.physical_links.items():
            if link_id > 0:  # Only forward links
                bw_used = 0

                for q in self.sfc_requests:
                    if results['sfc_acceptance'].get(q.name, False):
                        for vl in q.virtual_links:
                            bw_used += vl.bandwidth_demand

                residual['physical_links'][link_id] = {
                    'source': link.source,
                    'target': link.target,
                    'bandwidth_capacity': link.bandwidth,
                    'bandwidth_used': bw_used,
                    'bandwidth_residual': link.bandwidth - bw_used,
                    'utilization_percent': (bw_used / link.bandwidth * 100) if link.bandwidth > 0 else 0
                }

        return residual

    @staticmethod
    def display_residual_capacity(residual: Dict):
        """Display residual capacity information"""
        print("\n" + "=" * 80)
        print("RESIDUAL CAPACITY")
        print("=" * 80)

        print("\n--- Network PoPs ---")
        for npop_name, data in residual['npops'].items():
            print(f"\n{npop_name}:")
            print(f"  CPU: {data['cpu_used']:.2f}/{data['cpu_capacity']:.2f} " +
                  f"(Residual: {data['cpu_residual']:.2f}, {data['cpu_utilization_percent']:.1f}% used)")
            print(f"  MEM: {data['mem_used']:.2f}/{data['mem_capacity']:.2f} " +
                  f"(Residual: {data['mem_residual']:.2f}, {data['mem_utilization_percent']:.1f}% used)")

        print("\n--- Physical Switches ---")
        for ps_name, data in residual['physical_switches'].items():
            print(f"{ps_name}: {data['stages_used']}/{data['stage_capacity']} stages " +
                  f"(Residual: {data['stages_residual']}, {data['utilization_percent']:.1f}% used)")

        print("\n--- Physical Links ---")
        for link_id, data in residual['physical_links'].items():
            print(f"Link {link_id} ({data['source']} → {data['target']}): " +
                  f"{data['bandwidth_used']:.2f}/{data['bandwidth_capacity']:.2f} BW " +
                  f"(Residual: {data['bandwidth_residual']:.2f}, {data['utilization_percent']:.1f}% used)")

    def export_topology_to_json(self,  results: Dict, residual: Dict,
                                  input_file, parser, elapsed_time, solver, output_file, time_limit, mip_gap):
        """Export  results and complete topology information to JSON file"""

        # Build topology data
        topology_data = {
            "metadata": {
                "solver": solver,
                "status": results.get('status', 'Unknown'),
                "objective_value": results.get('objective'),
                "timestamp": None,
                "input_file": input_file,
                "output_file": output_file,
                "elapsed_time": elapsed_time,
                "time_limit": time_limit,
                "mip_gap": mip_gap,

                "physical_topology_summary": {
                    "number_endpoints": len(list(self.endpoints)),
                    "number_physical_switches": len(parser.physical_switches),
                    "number_network_pops": len(parser.npops),
                    "number_physical_links":len(parser.links),
                    "number_sfc_requests": len(parser.sfc_requests)
                },
            },
            "solution": {
                "accepted_sfcs": results.get('accepted_sfcs', 0),
                "total_sfcs": results.get('total_sfcs', 0),
                "sum_vnf_vs_instances_count": results.get('sum_vnf_vs_instances_count', 0),
                "vnf_instances_count": results.get('vnf_instances_count', 0),
                "vs_instances_count": results.get('vs_instances_count', 0),
                "objective": self.objective,
                "sfc_acceptance": results.get('sfc_acceptance', {}),
                "vnf_assignments": {
                    f"{k[0]}_{k[1]}": v
                    for k, v in results.get('vnf_assignments', {}).items()
                },
                "vs_assignments": results.get('vs_assignments', {})
            },
            "residual_capacity": residual,
            "physical_topology": {
                "endpoints": list(self.endpoints),
                "physical_switches": [
                    {
                        "name": ps.name,
                        "stage_capacity": ps.stage_capacity
                    }
                    for ps in self.physical_switches.values()
                ],
                "network_pops": [
                    {
                        "name": npop.name,
                        "cpu_capacity": npop.cpu_capacity,
                        "mem_capacity": npop.mem_capacity
                    }
                    for npop in self.npops.values()
                ],
                "physical_links": [
                    {
                        "id": link.id,
                        "source": link.source,
                        "target": link.target,
                        "bandwidth": link.bandwidth
                    }
                    for link in self.physical_links.values()
                    if link.id > 0
                ]
            },
            "sfc_requests": [
                {
                    "name": sfc.name,
                    "endpoints": sfc.endpoints,
                    "vnfs": [
                        {
                            "name": vf.name,
                            "cpu_demand": vf.cpu_demand,
                            "mem_demand": vf.mem_demand,
                            "assigned_npop": vf.assigned_npop
                        }
                        for vf in sfc.vnfs
                    ],
                    "virtual_switches": [
                        {
                            "name": vs.name,
                            "stage_demand": vs.stage_demand
                        }
                        for vs in sfc.virtual_switches
                    ],
                    "virtual_links": [
                        {
                            "source": vl.source,
                            "target": vl.target,
                            "bandwidth_demand": vl.bandwidth_demand,
                            "vs_type": vl.vs_type
                        }
                        for vl in sfc.virtual_links
                    ]
                }
                for sfc in self.sfc_requests
            ]
        }

        # Save to JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(topology_data, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Topology data exported to: {output_file}")

        return topology_data

    def display_topology_info(self):
        """Display complete topology information"""
        print("\n" + "=" * 80)
        print("TOPOLOGY INFORMATION")
        print("=" * 80)

        print(f"\n--- Physical Infrastructure ---")
        print(f"Endpoints: {len(self.endpoints)}")
        for ep in sorted(self.endpoints):
            print(f"  • {ep}")

        print(f"\nPhysical Switches: {len(self.physical_switches)}")
        for ps in self.physical_switches.values():
            print(f"  • {ps.name} (Stage Capacity: {ps.stage_capacity})")

        print(f"\nNetwork PoPs: {len(self.npops)}")
        for npop in self.npops.values():
            print(f"  • {npop.name} (CPU: {npop.cpu_capacity}, MEM: {npop.mem_capacity})")

        print(f"\nPhysical Links: {len([l for l in self.physical_links.values() if l.id > 0])}")
        for link in sorted([l for l in self.physical_links.values() if l.id > 0], key=lambda x: x.id):
            print(f"  • Link {link.id}: {link.source} ↔ {link.target} (BW: {link.bandwidth})")

        print(f"\n--- SFC Requests ---")
        print(f"Total SFC Requests: {len(self.sfc_requests)}")
        for sfc in self.sfc_requests:
            print(f"\n{sfc.name}:")
            print(f"  Endpoints: {', '.join(sfc.endpoints)}")
            print(f"  VNFs: {len(sfc.vnfs)}")
            for vf in sfc.vnfs:
                assigned_info = f" → {vf.assigned_npop}" if vf.assigned_npop else ""
                print(f"    • {vf.name} (CPU: {vf.cpu_demand}, MEM: {vf.mem_demand}){assigned_info}")
            print(f"  Virtual Switches: {len(sfc.virtual_switches)}")
            for vs in sfc.virtual_switches:
                print(f"    • {vs.name} (Stage Demand: {vs.stage_demand})")
            print(f"  Virtual Links: {len(sfc.virtual_links)}")
            for vl in sfc.virtual_links:
                vs_info = f" via {vl.vs_type}" if vl.vs_type else ""
                print(f"    • {vl.source} → {vl.target} (BW: {vl.bandwidth_demand}){vs_info}")

    def solve(self) -> Dict:
        """Solve the ILP problem"""
        print("=" * 80)
        print("SERVICE FUNCTION CHAINING ILP SOLVER - MULTI-OBJECTIVE")
        print("=" * 80)

        model = pulp.LpProblem("SFC_Deployment", pulp.LpMaximize)

        # Decision Variables
        z = {}
        for q in self.sfc_requests:
            z[q.name] = pulp.LpVariable(f"z_{q.name}", cat='Binary')

        x = {}
        for q in self.sfc_requests:
            for vf in q.vnfs:
                for npop_name in self.npops.keys():
                    x[(q.name, vf.name, npop_name)] = pulp.LpVariable(
                        f"x_{q.name}_{vf.name}_{npop_name}", cat='Binary'
                    )

        y = {}
        for q in self.sfc_requests:
            for vs in q.virtual_switches:
                for ps_name in self.physical_switches.keys():
                    y[(q.name, vs.name, ps_name)] = pulp.LpVariable(
                        f"y_{q.name}_{vs.name}_{ps_name}", cat='Binary'
                    )

        u = {}
        vf_types = set()
        for q in self.sfc_requests:
            for vf in q.vnfs:
                vf_types.add(vf.name)

        for vf_type in vf_types:
            for npop_name in self.npops.keys():
                u[(vf_type, npop_name)] = pulp.LpVariable(
                    f"u_{vf_type}_{npop_name}", cat='Binary'
                )

        v = {}
        for vs_type in self.vs_types:
            for ps_name in self.physical_switches.keys():
                v[(vs_type, ps_name)] = pulp.LpVariable(
                    f"v_{vs_type}_{ps_name}", cat='Binary'
                )

        f = {}
        for q in self.sfc_requests:
            for vl_idx, vl in enumerate(q.virtual_links):
                for link_id in self.physical_links.keys():
                    f[(q.name, vl_idx, link_id)] = pulp.LpVariable(
                        f"f_{q.name}_vl{vl_idx}_{link_id}", lowBound=0, upBound=1, cat='Continuous'
                    )

        # Objective Function

        if self.objective == "Maximize_SFCs_Minimize_Resources":
            M = 1000

            accepted_sfcs = pulp.lpSum([z[q.name] for q in self.sfc_requests])
            resources_used = pulp.lpSum([u[key] for key in u]) + pulp.lpSum([v[key] for key in v])

            objective = M * accepted_sfcs -  resources_used
            model += objective, self.objective

            print(f"\nObjective: Maximize [{M} * SFCs_accepted - resources]")
        elif self.objective == "Maximize_SFCs":
            accepted_sfcs = pulp.lpSum([z[q.name] for q in self.sfc_requests])

            objective =  accepted_sfcs
            model += objective, self.objective

            print(f"\nObjective: Maximize [SFCs_accepted]")
        else:
            print(f"ERROR: objective not found {self.objective}")
            sys.exit(1)

        print(f"  SFC requests: {len(self.sfc_requests)}")
        print(f"  VNF types: {len(vf_types)}")
        print(f"  VS types: {len(self.vs_types)}")

        constraint_count = 0

        # SFC Acceptance Constraints
        for q in self.sfc_requests:
            for vf in q.vnfs:
                for npop_name in self.npops.keys():
                    model += (
                        x[(q.name, vf.name, npop_name)] <= z[q.name],
                        f"SFC_Accept_VNF_{q.name}_{vf.name}_{npop_name}"
                    )
                    constraint_count += 1

            for vs in q.virtual_switches:
                for ps_name in self.physical_switches.keys():
                    model += (
                        y[(q.name, vs.name, ps_name)] <= z[q.name],
                        f"SFC_Accept_VS_{q.name}_{vs.name}_{ps_name}"
                    )
                    constraint_count += 1

            for vl_idx, vl in enumerate(q.virtual_links):
                total_flow = pulp.lpSum([
                    f[(q.name, vl_idx, link_id)]
                    for link_id in self.physical_links.keys()
                ])
                model += (
                    total_flow <= z[q.name] * len(self.physical_links),
                    f"SFC_Accept_Flow_{q.name}_vl{vl_idx}"
                )
                constraint_count += 1

        # VNF Deployment Constraints
        for q in self.sfc_requests:
            for vf in q.vnfs:
                if vf.assigned_npop is None:
                    model += (
                        pulp.lpSum([x[(q.name, vf.name, npop_name)]
                                    for npop_name in self.npops.keys()]) == z[q.name],
                        f"VNF_Assignment_{q.name}_{vf.name}"
                    )
                    constraint_count += 1
                else:
                    model += (
                        x[(q.name, vf.name, vf.assigned_npop)] == z[q.name],
                        f"VNF_PreAssignment_{q.name}_{vf.name}"
                    )
                    constraint_count += 1

        for q in self.sfc_requests:
            for vf in q.vnfs:
                for npop_name in self.npops.keys():
                    model += (
                        x[(q.name, vf.name, npop_name)] <= u[(vf.name, npop_name)],
                        f"VNF_Instance_Coupling_{q.name}_{vf.name}_{npop_name}"
                    )
                    constraint_count += 1

        for npop_name, npop in self.npops.items():
            cpu_usage = pulp.lpSum([
                x[(q.name, vf.name, npop_name)] * vf.cpu_demand
                for q in self.sfc_requests
                for vf in q.vnfs
            ])
            model += (
                cpu_usage <= npop.cpu_capacity,
                f"NPoP_CPU_Capacity_{npop_name}"
            )
            constraint_count += 1

        for npop_name, npop in self.npops.items():
            mem_usage = pulp.lpSum([
                x[(q.name, vf.name, npop_name)] * vf.mem_demand
                for q in self.sfc_requests
                for vf in q.vnfs
            ])
            model += (
                mem_usage <= npop.mem_capacity,
                f"NPoP_MEM_Capacity_{npop_name}"
            )
            constraint_count += 1

        # Virtual Switch Deployment Constraints
        for q in self.sfc_requests:
            for vs in q.virtual_switches:
                model += (
                    pulp.lpSum([y[(q.name, vs.name, ps_name)]
                                for ps_name in self.physical_switches.keys()]) >= z[q.name],
                    f"VS_Assignment_{q.name}_{vs.name}"
                )
                constraint_count += 1

        for q in self.sfc_requests:
            for vs in q.virtual_switches:
                for ps_name in self.physical_switches.keys():
                    model += (
                        y[(q.name, vs.name, ps_name)] <= v[(vs.name, ps_name)],
                        f"VS_Instance_Coupling_{q.name}_{vs.name}_{ps_name}"
                    )
                    constraint_count += 1

        for ps_name, ps in self.physical_switches.items():
            stage_usage = pulp.lpSum([
                y[(q.name, vs.name, ps_name)] * vs.stage_demand
                for q in self.sfc_requests
                for vs in q.virtual_switches
            ])
            model += (
                stage_usage <= ps.stage_capacity,
                f"PS_Stage_Capacity_{ps_name}"
            )
            constraint_count += 1

        # Flow Conservation Constraints
        all_nodes = self.get_all_nodes()

        for q in self.sfc_requests:
            for vl_idx, vl in enumerate(q.virtual_links):
                for node in all_nodes:
                    outgoing = self.get_outgoing_links(node)
                    incoming = self.get_incoming_links(node)

                    flow_balance = (
                            pulp.lpSum([f[(q.name, vl_idx, link.id)] for link in outgoing]) -
                            pulp.lpSum([f[(q.name, vl_idx, link.id)] for link in incoming])
                    )

                    balance_terms = []

                    if vl.source in [vf.name for vf in q.vnfs]:
                        if node in self.npops:
                            balance_terms.append(x[(q.name, vl.source, node)])
                    elif vl.source in q.endpoints:
                        if node == vl.source:
                            balance_terms.append(z[q.name])

                    if vl.target in [vf.name for vf in q.vnfs]:
                        if node in self.npops:
                            balance_terms.append(-x[(q.name, vl.target, node)])
                    elif vl.target in q.endpoints:
                        if node == vl.target:
                            balance_terms.append(-z[q.name])

                    if balance_terms:
                        balance = pulp.lpSum(balance_terms)
                    else:
                        balance = 0

                    model += (
                        flow_balance == balance,
                        f"Flow_Conservation_{q.name}_vl{vl_idx}_{node}"
                    )
                    constraint_count += 1

        # Bandwidth Capacity Constraints
        for link_id, link in self.physical_links.items():
            bandwidth_usage = pulp.lpSum([
                f[(q.name, vl_idx, link_id)] * vl.bandwidth_demand
                for q in self.sfc_requests
                for vl_idx, vl in enumerate(q.virtual_links)
            ])
            model += (
                bandwidth_usage <= link.bandwidth,
                f"Link_BW_Capacity_{link_id}"
            )
            constraint_count += 1

        # Path Consistency Constraints
        for q in self.sfc_requests:
            for vl_idx, vl in enumerate(q.virtual_links):
                vs_type = vl.vs_type
                if vs_type:
                    vs_obj = None
                    for vs in q.virtual_switches:
                        if vs.name == vs_type:
                            vs_obj = vs
                            break

                    if vs_obj:
                        for link_id, link in self.physical_links.items():
                            if link.source in self.physical_switches:
                                model += (
                                    f[(q.name, vl_idx, link_id)] <= y[(q.name, vs_obj.name, link.source)],
                                    f"VS_Path_Consistency_{q.name}_vl{vl_idx}_{link_id}_src"
                                )
                                constraint_count += 1

                            if link.target in self.physical_switches:
                                model += (
                                    f[(q.name, vl_idx, link_id)] <= y[(q.name, vs_obj.name, link.target)],
                                    f"VS_Path_Consistency_{q.name}_vl{vl_idx}_{link_id}_tgt"
                                )
                                constraint_count += 1

        print(f"\nTotal constraints: {constraint_count}")
        print(f"Total variables: {len(z) + len(x) + len(y) + len(u) + len(v) + len(f)}")

        # Solve
        print("\n" + "=" * 80)
        print("SOLVING...")
        print("=" * 80)

        #solver = pulp.CPLEX_CMD(msg=1, timeLimit=300, threads=4)
        #solver = pulp.CPLEX_PY(msg=1, timeLimit=300, threads=8)
        #solver = pulp.PULP_CBC_CMD(msg=1, timeLimit=300)
        model.writeLP("sfc_model.lp")
        model.solve(self.solver)

        # Extract Results
        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)

        status = pulp.LpStatus[model.status]
        print(f"\nStatus: {status}")

        if model.status == pulp.LpStatusOptimal:
            print(f"Objective: {pulp.value(model.objective):.2f}")

            accepted_count = sum([pulp.value(z[q.name]) for q in self.sfc_requests])
            total_count = len(self.sfc_requests)
            vnf_inst_count = sum([pulp.value(u[key]) for key in u])
            vs_inst_count = sum([pulp.value(v[key]) for key in v])

            print(f"\nSFCs Accepted: {int(accepted_count)}/{total_count}")
            print(f"VNF Instances: {int(vnf_inst_count)}")
            print(f"VS Instances: {int(vs_inst_count)}")

            results = {
                'status': status,
                'objective': pulp.value(model.objective),
                'accepted_sfcs': int(accepted_count),
                'total_sfcs': total_count,
                'vnf_instances_count': int(vnf_inst_count),
                'vs_instances_count': int(vs_inst_count),
                'sum_vnf_vs_instances_count': int(vnf_inst_count) + int(vs_inst_count),
                'sfc_acceptance': {},
                'vnf_assignments': {},
                'vs_assignments': {},
            }

            print("\n--- SFC Status ---")
            for q in self.sfc_requests:
                accepted = pulp.value(z[q.name]) > 0.5
                print(f"{q.name}: {'✓ ACCEPTED' if accepted else '✗ REJECTED'}")
                results['sfc_acceptance'][q.name] = accepted

            print("\n--- VNF Assignments ---")
            for q in self.sfc_requests:
                if pulp.value(z[q.name]) > 0.5:
                    print(f"\n{q.name}:")
                    for vf in q.vnfs:
                        for npop_name in self.npops.keys():
                            if pulp.value(x[(q.name, vf.name, npop_name)]) > 0.5:
                                print(f"  {vf.name} → {npop_name}")
                                results['vnf_assignments'][(q.name, vf.name)] = npop_name

            print("\n--- VS Assignments ---")
            vs_assignments_dict = {}
            for q in self.sfc_requests:
                if pulp.value(z[q.name]) > 0.5:
                    print(f"\n{q.name}:")
                    for vs in q.virtual_switches:
                        assignments = []
                        for ps_name in self.physical_switches.keys():
                            if pulp.value(y[(q.name, vs.name, ps_name)]) > 0.5:
                                assignments.append(ps_name)
                        if assignments:
                            print(f"  {vs.name} → {', '.join(assignments)}")
                            vs_assignments_dict[f"{q.name}_{vs.name}"] = assignments

            results['vs_assignments'] = vs_assignments_dict

            # Calculate and display residual capacity
            residual = self.calculate_residual_capacity(results)
            self.display_residual_capacity(residual)
            results['residual_capacity'] = residual

            return results
        else:
            return {'status': status, 'objective': None}

def format_time(seconds):
    """Converte segundos para formato legível"""
    if seconds < 60:
        return f"{seconds:.2f} segundos"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes} minuto(s) e {secs:.2f} segundos"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours} hours(s), {minutes} minutes(s) e {secs:.2f} seconds"

def create_arg_parser():
    arg_parser = argparse.ArgumentParser(
        description="SFC SOLVER WITH TEXT PARSER",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
File format:
  Topology
  endpoint e1 e2 e3 ...
  physical_switch ps1 20
  network_pop npop1 1.0 1.0
  link e1 ps1 500

  SFC Q1
  endpoint e1 e7 e8
  virtual_function fw 0.1 0.1 npop1
  virtual_link e1 fw 10 vs_int
  vs_int stage_demand 5
  vs_scion stage_demand 3
{"=" * 80}
        """
    )

    # Required positional arguments
    arg_parser.add_argument(
        '--input', '-i',
        help='Input network topology file'
    )

    # Optional output file
    arg_parser.add_argument(
        '--output', '-o',
        nargs='?',
        default='topology_results.json',
        help='Output JSON file (default: topology_results.json)'
    )

    # Solver selection
    arg_parser.add_argument(
        '--solver', '-s',
        choices=['cbc', 'cplex', 'gurobi', 'glpk', 'scip'],
        default='cbc',
        help='MILP solver to use (default: cbc)'
    )

    arg_parser.add_argument(
        '--objective', '-b',
        choices=['Maximize_SFCs', 'Maximize_SFCs_Minimize_Resources'],
        default=DEFAULT_OBJECTIVE,
        help='Objective function to use'
    )

    # Solver-specific options
    arg_parser.add_argument(
        '--time-limit', '-t',
        type=int,
        default=None,
        help='Time limit for solver in seconds'
    )

    arg_parser.add_argument(
        '--mip-gap',
        type=float,
        default=None,
        help='MIP gap tolerance (e.g., 0.01 for 1%%)'
    )

    arg_parser.add_argument(
        '--threads',
        type=int,
        default=None,
        help='Number of threads for solver to use'
    )

    # TODO: implement logging with different verbose
    # parser.add_argument(
    #     '--verbose', '-v',
    #     action='store_true',
    #     help='Enable verbose solver output'
    # )

    # parser.add_argument(
    #     '--log-file',
    #     help='Solver log file path'
    # )
    try:
        args = arg_parser.parse_args()
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)

    return args

# ========================
# UTILITY FUNCTIONS
# ========================
def print_all_settings(arguments):
    """Logs the full configuration of the current experiment based on parsed arguments."""
    print(f"Command:\n\t{0}\n".format(" ".join([x for x in sys.argv])))
    print(f"Settings:")
    lengths = [len(x) for x in vars(arguments).keys()]
    max_length = max(lengths)

    for key_item, values in sorted(vars(arguments).items()):
        message = "\t"
        message += key_item.ljust(max_length, " ")
        message += " : {}".format(values)
        print(message)

    print("")

def check_solver_availability(solver_name):
    """
    Verifica se um solver específico está disponível
    Retorna (disponivel, comando_instalacao)
    """
    solvers_info = {
        'cbc': {
            'test': lambda: pulp.PULP_CBC_CMD().available(),
            'install_commands': {
                'pip': 'pip install pulp',
                'conda': 'conda install -c conda-forge coincbc',
                'apt': 'sudo apt install coinor-cbc'
            }
        },
        'gurobi': {
            'test': lambda: pulp.GUROBI().available(),
            'install_commands': {
                'pip': 'pip install gurobipy',
                'website': 'https://www.gurobi.com/downloads/ (requires license)'
            }
        },
        'cplex': {
            'test': lambda: pulp.CPLEX_CMD().available(),
            'install_commands': {
                'website': 'https://www.ibm.com/products/ilog-cplex-optimization-studio (requires license)'
            }
        },
        'glpk': {
            'test': lambda: pulp.GLPK_CMD().available(),
            'install_commands': {
                'apt': 'sudo apt install glpk-utils',
                'brew': 'brew install glpk',
                'yum': 'sudo yum install glpk-utils'
            }
        },
        'scip': {
            'test': lambda: pulp.SCIP_CMD().available(),
            'install_commands': {
                'website': 'https://www.scipopt.org/index.php#download',
                'conda': 'conda install -c conda-forge scip'
            }
        }
    }

    if solver_name not in solvers_info:
        return False, f"Solver '{solver_name}' não reconhecido"

    try:
        available = solvers_info[solver_name]['test']()
        install_commands = solvers_info[solver_name]['install_commands']
        if not available:
            print(f"⚠️  Solver '{solver_name}' não está disponível!")
            print("Comandos de instalação:")

            if isinstance(install_commands, dict):
                for method, command in install_commands.items():
                    print(f"   {method}: {command}")
            else:
                print(f"   {install_commands}")

        return available
    except Exception as e:
        return False, solvers_info[solver_name]['install_commands']

def main():
    """Main function - reads network definition from file parameter"""

    arg_parser = create_arg_parser()
    print_all_settings(arg_parser)

    filename = arg_parser.input
    output_json = arg_parser.output

    solver_kwargs = {}
    # Add common options
    if arg_parser.time_limit:
        solver_kwargs['timeLimit'] = arg_parser.time_limit
    if arg_parser.mip_gap:
        solver_kwargs['gapRel'] = arg_parser.mip_gap
    if arg_parser.threads:
        solver_kwargs['threads'] = arg_parser.threads
    # if arg_parser.log_file:
    #     solver_kwargs['logPath'] = arg_parser.log_file
    # if hasattr(args, 'verbose') and arg_parser.verbose:
    #     solver_kwargs['msg'] = True

    # Create solver instance based on selection
    try:
        if arg_parser.solver == 'cbc':
            pulp_solver = pulp.PULP_CBC_CMD(**solver_kwargs)
        elif arg_parser.solver == 'gurobi':
            licence_file_name = "gurobi.lic"
            if os.path.exists(licence_file_name):
                import gurobipy
                wls_params={}
                with open(licence_file_name, 'r') as f:
                    for l in f:
                        l = l.strip()
                        if l and not l.startswith('#') and '=' in l:
                            key, value = l.split('=', 1)
                            value = value.strip()
                            if key == "LICENSEID":
                                value = int(value.strip())

                            wls_params[key.strip()] = value

                print(wls_params)

                env = gurobipy.Env(params=wls_params)
                pulp_solver = pulp.GUROBI(env=env, **solver_kwargs)
            else:
                pulp_solver = pulp.GUROBI(**solver_kwargs)
        elif arg_parser.solver == 'cplex':
            pulp_solver = pulp.CPLEX_CMD(**solver_kwargs)
        elif arg_parser.solver == 'glpk':
            pulp_solver = pulp.GLPK_CMD(**solver_kwargs)
        elif arg_parser.solver == 'scip':
            pulp_solver = pulp.SCIP_CMD(**solver_kwargs)
        else:
            print(f"ERROR: Unknown solver {arg_parser.solver}")
            sys.exit()
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)

    if not check_solver_availability(arg_parser.solver):
        sys.exit(1)

    print("=" * 80)
    print("Solver using MILP based approaches")
    print("=" * 80)
    print(f"\nReading from file: {filename}")

    try:
        # Parse network definition from file
        print("\nParsing network definition...")
        parser = NetworkParser()
        parser.parse_file(filename)

        print(f"  ✓ Endpoints: {len(parser.endpoints)}")
        print(f"  ✓ Physical Switches: {len(parser.physical_switches)}")
        print(f"  ✓ Network PoPs: {len(parser.npops)}")
        print(f"  ✓ Physical Links: {len(parser.links)}")
        print(f"  ✓ SFC Requests: {len(parser.sfc_requests)}")

        # Create solver and apply configuration
        print("\nConfiguring solver...")
        solver = SFCILPSolver(pulp_solver, arg_parser.objective)
        parser.apply_to_solver(solver)
        print("  ✓ Solver configured")

        # Display topology information
        solver.display_topology_info()

        # Solve
        print("\nSolving...")
        start_time = time.time()
        results = solver.solve()
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time solving: {format_time(elapsed_time)}")

        # Export to JSON
        if results['status'] == 'Optimal':
            print("\nExporting results...")
            residual = results.get('residual_capacity', solver.calculate_residual_capacity(results))

            # (self, output_file: str = "topology_results.json", results: Dict, residual: Dict,
            #                       input_file, parser, elapsed_time,solver):
            topology_data = solver.export_topology_to_json(output_file=output_json,
                                                           results=results,
                                                           residual=residual,
                                                           input_file=filename,
                                                           parser=parser,
                                                           elapsed_time=elapsed_time,
                                                           solver=arg_parser.solver,
                                                           time_limit=arg_parser.time_limit,
                                                           mip_gap=arg_parser.mip_gap)

        # Summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        if results['status'] == 'Optimal':
            print(f"\n✓ Solution found!")
            print(f"  Accepted: {results['accepted_sfcs']}/{results['total_sfcs']} SFCs")
            print(f"  Instantiated Sum: {results['sum_vnf_vs_instances_count']} VNF:{results['vnf_instances_count']} VS:{results['vs_instances_count']}")

            print(f"  Objective Value: {results['objective']:.2f}")


            print("\n  SFC Status:")
            for sfc_name, accepted in results['sfc_acceptance'].items():
                status = "✓ Accepted" if accepted else "✗ Rejected"
                print(f"    {sfc_name}: {status}")

            print(f"\n  Results saved to: {output_json}")
        else:
            print(f"\n✗ No optimal solution found")
            print(f"  Status: {results['status']}")

        print("\n" + "=" * 80)
        print("COMPLETED")
        print("=" * 80)

        return results

    except FileNotFoundError:
        print(f"\n✗ ERROR: File '{filename}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
