import logging
from collections import defaultdict
from typing import Tuple

import numpy
import torch
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv

from Components.Network.PathFinder import PathFinder
from Components.Network.ResourceManager import ResourceManager

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sfc_dqn_execution.log', mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {DEVICE}")
if DEVICE.type == 'cuda':
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")



REWARD_SFC_ACCEPTED = 10.0
REWARD_SFC_REJECTED = -10.0
REWARD_INVALID_ACTION = -30.0
REWARD_NEW_NPOP_PENALTY = -1
REWARD_REUSE_NPOP_BONUS = 5.0
REWARD_CONSOLIDATION_BONUS = 1.5
REWARD_STEP_PENALTY = 0.0
REWARD_RESOURCE_USAGE = 0.0
REWARD_VNF_PLACEMENT_SUCCESS = 0.0

REWARD_VS_REUSE_BONUS = 0.5


class SFCEnvironment:

    def __init__(self, topology_data: dict):
        self.topology = topology_data
        self.path_finder = PathFinder(topology_data)
        self.resource_manager = ResourceManager()

        self.endpoint_to_node = {}
        for link_id, data in self.topology['physical_links'].items():
            source = data['source']
            target = data['target']
            if source in self.topology['endpoints']:
                self.endpoint_to_node[source] = target
            if target in self.topology['endpoints']:
                self.endpoint_to_node[target] = source

        self.reset()

    def reset(self):
        """Reset environment to initial state"""
        self.npop_cpu = {name: data['cpu_capacity'] for name, data in self.topology['npops'].items()}
        self.npop_mem = {name: data['mem_capacity'] for name, data in self.topology['npops'].items()}
        self.ps_stages = {name: data['stage_capacity'] for name, data in self.topology['physical_switches'].items()}
        self.link_bw = {link_id: data['bandwidth'] for link_id, data in self.topology['physical_links'].items()}

        self.deployed_sfcs = []
        self.vnf_placements = {}
        self.vs_placements = {}
        self.vlink_routes = {}

        self.npop_cpu_used = defaultdict(float)
        self.npop_mem_used = defaultdict(float)
        self.ps_stages_used = defaultdict(int)
        self.link_bw_used = defaultdict(float)

        self.npops_used_globally = set()
        self.npops_used_current_sfc = set()

        self.resource_manager = ResourceManager()

        self.current_sfc_idx = 0
        self.current_vnf_idx = 0
        self.current_sfc = None
        self.sfc_requests = self.topology['sfc_requests']

        return self._get_state()

    def _get_state(self):
        """Get current state representation"""
        if self.current_sfc_idx >= len(self.sfc_requests):
            return None

        self.current_sfc = self.sfc_requests[self.current_sfc_idx]

        infra_features = self._build_infra_features()
        infra_edges = self._build_infra_edges()
        sfc_features = self._build_sfc_features()

        return {
            'infra_data': Data(
                x=torch.FloatTensor(infra_features).to(DEVICE),
                edge_index=torch.LongTensor(infra_edges).to(DEVICE)
            ),
            'sfc_data': Data(
                x=torch.FloatTensor(sfc_features).to(DEVICE)
            ),
            'num_npops': len(self.topology['npops']),
            'vnf_idx': self.current_vnf_idx,
            'sfc_name': self.current_sfc['name']
        }

    def _build_infra_features(self):

        features = []

        # 1. NPoPs (can host VNFs)
        for name, npop in self.topology['npops'].items():
            # Global utilization (all SFCs)
            cpu_util_global = self.npop_cpu_used[name] / npop['cpu_capacity'] if npop['cpu_capacity'] > 0 else 0
            mem_util_global = self.npop_mem_used[name] / npop['mem_capacity'] if npop['mem_capacity'] > 0 else 0
            is_used = 1.0 if name in self.npops_used_globally else 0.0

            cpu_used_current_sfc = 0.0
            mem_used_current_sfc = 0.0

            if self.current_sfc:
                for vnf_idx in range(self.current_vnf_idx):
                    if vnf_idx < len(self.current_sfc['vnfs']):
                        vnf = self.current_sfc['vnfs'][vnf_idx]
                        key = (self.current_sfc['name'], vnf['name'])
                        if key in self.vnf_placements and self.vnf_placements[key] == name:
                            cpu_used_current_sfc += vnf['cpu_demand']
                            mem_used_current_sfc += vnf['mem_demand']

            cpu_util_current = cpu_used_current_sfc / npop['cpu_capacity'] if npop['cpu_capacity'] > 0 else 0
            mem_util_current = mem_used_current_sfc / npop['mem_capacity'] if npop['mem_capacity'] > 0 else 0

            features.append([
                1.0,  # node_type: NPoP
                cpu_util_global,  # Global CPU utilization
                mem_util_global,  # Global Memory utilization
                npop['cpu_capacity'] - self.npop_cpu_used[name],  # Available CPU
                is_used,  # Is being used globally
                0.0,  # Not an endpoint
                cpu_util_current,  # ✅ NEW: Current SFC CPU utilization
                mem_util_current  # ✅ NEW: Current SFC Memory utilization
            ])

        for name, ps in self.topology['physical_switches'].items():
            stage_util_global = self.ps_stages_used[name] / ps['stage_capacity'] if ps['stage_capacity'] > 0 else 0

            features.append([
                0.0,  # node_type: Physical Switch
                stage_util_global,  # Global stage utilization
                0.0,  # No memory
                ps['stage_capacity'] - self.ps_stages_used[name],  # Available stages
                0.0,  # Not tracked for usage
                0.0,  # Not an endpoint
                0.0,  # Current SFC: no stages yet
                0.0  # Current SFC: no memory
            ])

        # 3. Endpoints (access points for SFCs)
        for endpoint in self.topology['endpoints']:
            # Check if this endpoint is used in current SFC
            is_sfc_endpoint = 0.0
            if self.current_sfc and endpoint in self.current_sfc['endpoints']:
                is_sfc_endpoint = 1.0

            features.append([
                -1.0,  # node_type: Endpoint (negative to distinguish)
                0.0,  # No global utilization
                0.0,  # No memory
                0.0,  # No capacity
                is_sfc_endpoint,  # Is used in current SFC
                1.0,  # IS an endpoint
                0.0,  # Current SFC: no resources
                0.0  # Current SFC: no resources
            ])

        return features

    def _build_infra_edges(self):

        node_map = {}
        idx = 0

        # Map NPoPs
        for name in self.topology['npops'].keys():
            node_map[name] = idx
            idx += 1

        # Map Physical Switches
        for name in self.topology['physical_switches'].keys():
            node_map[name] = idx
            idx += 1

        # Map Endpoints
        for endpoint in self.topology['endpoints']:
            node_map[endpoint] = idx
            idx += 1

        # Build edges (bidirectional)
        edges = [[], []]
        for link in self.topology['physical_links'].values():
            src = link['source']
            tgt = link['target']

            if src in node_map and tgt in node_map:
                src_idx = node_map[src]
                tgt_idx = node_map[tgt]
                # Bidirectional edges
                edges[0].extend([src_idx, tgt_idx])
                edges[1].extend([tgt_idx, src_idx])

        return edges

    def _build_sfc_features(self):

        features = []

        # Get NPoP name list for indexing
        npop_names = list(self.topology['npops'].keys())

        # VNF features
        for idx, vnf in enumerate(self.current_sfc['vnfs']):
            is_current = 1.0 if idx == self.current_vnf_idx else 0.0
            is_placed = 1.0 if idx < self.current_vnf_idx else 0.0  # ✅ NEW: Already placed?

            placed_npop_idx = -1.0  # -1 means not placed
            if is_placed:
                key = (self.current_sfc['name'], vnf['name'])
                if key in self.vnf_placements:
                    npop_name = self.vnf_placements[key]
                    if npop_name in npop_names:
                        # Normalize to [0, 1] range
                        placed_npop_idx = npop_names.index(npop_name) / max(1, len(npop_names) - 1)

            features.append([
                1.0,  # is_vnf
                vnf['cpu_demand'],
                vnf['mem_demand'],
                is_current,
                0.0,
                0.0,
                is_placed,
                placed_npop_idx
            ])

        for vl in self.current_sfc['virtual_links']:
            src_is_endpoint = 1.0 if vl['source'] in self.topology['endpoints'] else 0.0
            tgt_is_endpoint = 1.0 if vl['target'] in self.topology['endpoints'] else 0.0
            touches_endpoint = max(src_is_endpoint, tgt_is_endpoint)

            features.append([
                0.0,  # Not a VNF
                vl['bandwidth_demand'],
                0.0,
                0.0,
                1.0 if vl['vs_type'] else 0.0,
                touches_endpoint,
                0.0,
                -1.0
            ])

        return features

    def get_valid_actions(self):
        """Get mask of valid actions for current VNF"""
        if self.current_vnf_idx >= len(self.current_sfc['vnfs']):
            return None

        vnf = self.current_sfc['vnfs'][self.current_vnf_idx]
        mask = []

        for name, npop in self.topology['npops'].items():
            cpu_available = npop['cpu_capacity'] - self.npop_cpu_used[name]
            mem_available = npop['mem_capacity'] - self.npop_mem_used[name]

            # VNF uses accumulated model - ALWAYS check capacity
            can_place = self.resource_manager.can_place_vnf(
                vnf['name'], name, vnf['cpu_demand'], vnf['mem_demand'],
                cpu_available, mem_available
            )
            mask.append(can_place)

        return torch.BoolTensor(mask).to(DEVICE)

    def step(self, action: int):
        if self.current_vnf_idx >= len(self.current_sfc['vnfs']):
            return None, 0, True

        vnf = self.current_sfc['vnfs'][self.current_vnf_idx]
        npop_names = list(self.topology['npops'].keys())

        valid_actions = self.get_valid_actions()
        if not valid_actions[action]:
            return self._get_state(), REWARD_INVALID_ACTION, False

        npop_name = npop_names[action]
        consolidation_reward = 0.0

        if npop_name not in self.npops_used_globally:
            consolidation_reward += REWARD_NEW_NPOP_PENALTY
        else:
            consolidation_reward += REWARD_REUSE_NPOP_BONUS

        self.vnf_placements[(self.current_sfc['name'], vnf['name'])] = npop_name

        cpu_consumed, mem_consumed = self.resource_manager.add_vnf_deployment(
            vnf['name'], npop_name, self.current_sfc['name'],
            vnf['cpu_demand'], vnf['mem_demand']
        )

        self.npop_cpu_used[npop_name] += cpu_consumed
        self.npop_mem_used[npop_name] += mem_consumed

        self.npops_used_globally.add(npop_name)
        self.npops_used_current_sfc.add(npop_name)

        reward = consolidation_reward + REWARD_RESOURCE_USAGE + REWARD_STEP_PENALTY

        self.current_vnf_idx += 1
        done = False

        if self.current_vnf_idx >= len(self.current_sfc['vnfs']):
            num_npops_used = len(self.npops_used_current_sfc)
            num_vnfs = len(self.current_sfc['vnfs'])

            if num_npops_used > 0:
                consolidation_ratio = 1.0 / num_npops_used
                reward += REWARD_CONSOLIDATION_BONUS * consolidation_ratio * num_vnfs

            can_deploy, vs_reward = self._check_vs_and_links_with_paths()

            if can_deploy:
                self._deploy_vs_and_links_with_paths()
                self.deployed_sfcs.append(self.current_sfc['name'])
                reward += REWARD_SFC_ACCEPTED + vs_reward
            else:
                self._rollback_vnf_placements()
                reward += REWARD_SFC_REJECTED

            self.current_sfc_idx += 1
            self.current_vnf_idx = 0
            self.npops_used_current_sfc = set()

            if self.current_sfc_idx >= len(self.sfc_requests):
                done = True

        next_state = self._get_state() if not done else None
        return next_state, reward, done

    def _check_vs_and_links_with_paths(self) -> Tuple[bool, float]:
        """Check if VS and virtual links can be placed with proper path finding"""
        vs_reward = 0.0
        vlink_paths = {}

        # Calculate ACTUAL bandwidth availability from resource manager
        link_bw_available = {}
        for link_id, link in self.topology['physical_links'].items():
            total_used = self.resource_manager.get_link_usage(link_id)
            available = link['bandwidth'] - total_used
            link_bw_available[link_id] = available

        for vlink_idx, vl in enumerate(self.current_sfc['virtual_links']):
            src_node = self._get_node_for_vlink_endpoint(vl['source'])
            tgt_node = self._get_node_for_vlink_endpoint(vl['target'])


            if src_node is None or tgt_node is None:
                return False, 0.0

            node_path, link_path = self.path_finder.find_path(
                src_node, tgt_node, vl['bandwidth_demand'], link_bw_available
            )

            if node_path is None:
                # Log detailed network state for debugging
                return False, 0.0

            vlink_paths[vlink_idx] = (node_path, link_path)

        # Check VS placement
        required_vs_per_ps = defaultdict(set)

        for vlink_idx, (node_path, link_path) in vlink_paths.items():
            vl = self.current_sfc['virtual_links'][vlink_idx]
            if vl['vs_type']:
                switches_on_path = self.path_finder.get_switches_on_path(node_path)
                for ps in switches_on_path:
                    required_vs_per_ps[ps].add(vl['vs_type'])

        for vs in self.current_sfc['virtual_switches']:
            vs_type = vs['name']
            stage_demand = vs['stage_demand']

            switches_needing_vs = [ps for ps, vs_set in required_vs_per_ps.items()
                                   if vs_type in vs_set]

            if not switches_needing_vs:
                continue

            for ps_name in switches_needing_vs:
                ps = self.topology['physical_switches'][ps_name]
                stages_available = ps['stage_capacity'] - self.ps_stages_used[ps_name]

                can_place = self.resource_manager.can_place_vs(
                    vs_type, ps_name, stage_demand, stages_available
                )

                if not can_place:
                    return False, 0.0

                vs_key = (vs_type, ps_name, stage_demand)
                if vs_key in self.resource_manager.vs_instances:
                    vs_reward += REWARD_VS_REUSE_BONUS

        self._temp_vlink_paths = vlink_paths
        self._temp_required_vs_per_ps = required_vs_per_ps

        return True, vs_reward

    def _log_network_state(self, source: str, target: str, bw_demand: float):
        """Log detailed network state for debugging path failures"""

        # Log links and their usage
        logger.info("Physical links and usage:")
        for link_id, link in self.topology['physical_links'].items():
            used = self.resource_manager.get_link_usage(link_id)
            available = link['bandwidth'] - used


        if source in self.path_finder.graph:
            for neighbor, bw, link_id in self.path_finder.graph[source]:
                available = link['bandwidth'] - self.resource_manager.get_link_usage(link_id)

    def _deploy_vs_and_links_with_paths(self):
        sfc_name = self.current_sfc['name']

        vs_deployments = defaultdict(list)

        # Deploy VS instances (TRUE sharing)
        for vs in self.current_sfc['virtual_switches']:
            vs_type = vs['name']
            stage_demand = vs['stage_demand']

            switches_needing_vs = [ps for ps, vs_set in self._temp_required_vs_per_ps.items()
                                   if vs_type in vs_set]

            for ps_name in switches_needing_vs:
                stages_consumed = self.resource_manager.add_vs_user(
                    vs_type, ps_name, sfc_name, stage_demand
                )

                if stages_consumed > 0:
                    self.ps_stages_used[ps_name] += stages_consumed

                vs_deployments[vs_type].append(ps_name)

        for vs_type, ps_list in vs_deployments.items():
            key = f"{sfc_name}_{vs_type}"
            self.vs_placements[key] = ps_list

        for vlink_idx, (node_path, link_path) in self._temp_vlink_paths.items():
            vl = self.current_sfc['virtual_links'][vlink_idx]

            for link_id in link_path:
                bw_consumed = self.resource_manager.add_link_usage(
                    link_id, sfc_name, vl['bandwidth_demand']
                )
                self.link_bw_used[link_id] += bw_consumed

            self.vlink_routes[(sfc_name, vlink_idx)] = (node_path, link_path)

        del self._temp_vlink_paths
        del self._temp_required_vs_per_ps

    def _rollback_vnf_placements(self):
        sfc_name = self.current_sfc['name']

        for vnf in self.current_sfc['vnfs']:
            key = (sfc_name, vnf['name'])
            if key in self.vnf_placements:
                npop_name = self.vnf_placements[key]
                cpu_freed, mem_freed = self.resource_manager.remove_vnf_deployment(
                    vnf['name'], npop_name, sfc_name
                )

                self.npop_cpu_used[npop_name] -= cpu_freed
                self.npop_mem_used[npop_name] -= mem_freed

                if self.npop_cpu_used[npop_name] <= 1e-6 and self.npop_mem_used[npop_name] <= 1e-6:
                    self.npops_used_globally.discard(npop_name)

                del self.vnf_placements[key]

    def _get_node_for_vlink_endpoint(self, endpoint_name: str) -> str:
        """Get the physical node for a virtual link endpoint"""


        # Check if it's a VNF that's been placed
        if self.current_sfc:
            for vnf in self.current_sfc['vnfs']:
                if vnf['name'] == endpoint_name:
                    key = (self.current_sfc['name'], vnf['name'])
                    if key in self.vnf_placements:
                        npop = self.vnf_placements[key]
                        return npop

        # Check if it's directly a physical node
        if endpoint_name in self.topology['npops']:
            return endpoint_name
        if endpoint_name in self.topology['physical_switches']:
            return endpoint_name

        # Check if it's a physical endpoint
        if endpoint_name in self.topology['endpoints']:
            connected_node = self.endpoint_to_node.get(endpoint_name)
            return connected_node

        return None

    def get_results(self):
        """Get deployment results"""
        stats = self.resource_manager.get_stats()
        return {
            'deployed_sfcs': self.deployed_sfcs,
            'vnf_placements': dict(self.vnf_placements),
            'vs_placements': dict(self.vs_placements),
            'vlink_routes': dict(self.vlink_routes),
            'accepted_sfcs': len(self.deployed_sfcs),
            'total_sfcs': len(self.sfc_requests),
            'npops_used': len(self.npops_used_globally),
            'vnf_instances': stats['vnf_deployments'],
            'vs_instances': stats['vs_instances']
        }
