from collections import defaultdict
from typing import Dict, Tuple

class ResourceManager:


    def __init__(self):
        # VS: TRUE sharing (consumed once)
        self.vs_instances = {}

        # VNF: ACCUMULATED per SFC (NOT shared)
        self.vnf_deployments = {}

        # VL: Track bandwidth usage per physical link (ACCUMULATED)
        self.link_bandwidth = defaultdict(lambda: {'total_used': 0.0, 'sfc_demands': {}})


    def can_place_vs(self, vs_type: str, ps_name: str, stage_demand: int,
                     stages_available: int) -> bool:

        key = (vs_type, ps_name, stage_demand)

        if key in self.vs_instances:
            return True  # Sharing existing instance

        return stages_available >= stage_demand  # New instance

    def add_vs_user(self, vs_type: str, ps_name: str, sfc_name: str,
                    stage_demand: int) -> int:

        key = (vs_type, ps_name, stage_demand)

        if key not in self.vs_instances:
            # NEW instance - consume stages ONCE
            self.vs_instances[key] = {
                'stages_used': stage_demand,
                'sfc_users': set()
            }
            stages_consumed = stage_demand
        else:
            # EXISTING instance - NO additional consumption
            stages_consumed = 0

        self.vs_instances[key]['sfc_users'].add(sfc_name)
        return stages_consumed

    def remove_vs_user(self, vs_type: str, ps_name: str, sfc_name: str,
                       stage_demand: int) -> int:

        key = (vs_type, ps_name, stage_demand)

        if key in self.vs_instances:
            self.vs_instances[key]['sfc_users'].discard(sfc_name)

            if not self.vs_instances[key]['sfc_users']:
                stages_freed = self.vs_instances[key]['stages_used']
                del self.vs_instances[key]
                return stages_freed

        return 0

    def can_place_vnf(self, vnf_type: str, npop_name: str,
                      cpu_demand: float, mem_demand: float,
                      cpu_available: float, mem_available: float) -> bool:

        return (cpu_available >= cpu_demand and
                mem_available >= mem_demand)

    def add_vnf_deployment(self, vnf_type: str, npop_name: str, sfc_name: str,
                           cpu_demand: float, mem_demand: float) -> Tuple[float, float]:

        key = (vnf_type, npop_name)

        if key not in self.vnf_deployments:
            self.vnf_deployments[key] = {
                'sfc_demands': {},
                'total_cpu': 0.0,
                'total_mem': 0.0
            }

        # Store this SFC's demand
        self.vnf_deployments[key]['sfc_demands'][sfc_name] = (cpu_demand, mem_demand)

        # ACCUMULATE to total
        self.vnf_deployments[key]['total_cpu'] += cpu_demand
        self.vnf_deployments[key]['total_mem'] += mem_demand


        # Return consumption (always the full demand)
        return cpu_demand, mem_demand

    def remove_vnf_deployment(self, vnf_type: str, npop_name: str,
                              sfc_name: str) -> Tuple[float, float]:

        key = (vnf_type, npop_name)

        if key not in self.vnf_deployments:
            return 0.0, 0.0

        if sfc_name not in self.vnf_deployments[key]['sfc_demands']:
            return 0.0, 0.0

        # Get this SFC's demand
        cpu_demand, mem_demand = self.vnf_deployments[key]['sfc_demands'][sfc_name]

        # SUBTRACT from total
        self.vnf_deployments[key]['total_cpu'] -= cpu_demand
        self.vnf_deployments[key]['total_mem'] -= mem_demand

        # Remove record
        del self.vnf_deployments[key]['sfc_demands'][sfc_name]

        # Clean up if no more SFCs
        if not self.vnf_deployments[key]['sfc_demands']:
            del self.vnf_deployments[key]


        return cpu_demand, mem_demand

    def add_link_usage(self, link_id: int, sfc_name: str, bandwidth_demand: float) -> float:

        if sfc_name not in self.link_bandwidth[link_id]['sfc_demands']:
            self.link_bandwidth[link_id]['sfc_demands'][sfc_name] = []

        self.link_bandwidth[link_id]['sfc_demands'][sfc_name].append(bandwidth_demand)
        self.link_bandwidth[link_id]['total_used'] += bandwidth_demand

        return bandwidth_demand

    def remove_link_usage(self, link_id: int, sfc_name: str) -> float:

        if link_id not in self.link_bandwidth:
            return 0.0

        if sfc_name not in self.link_bandwidth[link_id]['sfc_demands']:
            return 0.0

        demands = self.link_bandwidth[link_id]['sfc_demands'][sfc_name]
        total_freed = sum(demands)

        self.link_bandwidth[link_id]['total_used'] -= total_freed
        del self.link_bandwidth[link_id]['sfc_demands'][sfc_name]

        if not self.link_bandwidth[link_id]['sfc_demands']:
            del self.link_bandwidth[link_id]

        return total_freed

    def get_link_usage(self, link_id: int) -> float:
        """Get total bandwidth usage on a link."""
        return self.link_bandwidth[link_id]['total_used'] if link_id in self.link_bandwidth else 0.0

    def get_stats(self) -> Dict:
        """Get resource usage statistics."""
        return {
            'vs_instances': len(self.vs_instances),
            'vs_total_users': sum(len(inst['sfc_users']) for inst in self.vs_instances.values()),
            'vnf_deployments': len(self.vnf_deployments),
            'vnf_total_sfcs': sum(len(dep['sfc_demands']) for dep in self.vnf_deployments.values()),
            'links_used': len(self.link_bandwidth),
            'total_links_sfcs': sum(len(link['sfc_demands']) for link in self.link_bandwidth.values())
        }
