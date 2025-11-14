import heapq
from collections import defaultdict
from typing import List, Dict, Tuple

import numpy


class PathFinder:
    """Handles path computation using Dijkstra's algorithm"""

    def __init__(self, topology: dict):
        self.topology = topology
        self._build_graph()
        self._build_endpoint_mapping()

    def _build_endpoint_mapping(self):
        """Build mapping from endpoints to connected switches"""
        self.endpoint_to_switch = {}
        for link_id, link in self.topology['physical_links'].items():
            src = link['source']
            dst = link['target']
            # If source is endpoint, map it to destination (should be a switch/npop)
            if src in self.topology['endpoints']:
                self.endpoint_to_switch[src] = dst
            # If destination is endpoint, map it to source
            if dst in self.topology['endpoints']:
                self.endpoint_to_switch[dst] = src

    def _build_graph(self):
        """Build complete adjacency list including ALL nodes"""
        self.graph = defaultdict(list)
        self.link_map = {}
        self.all_nodes = set()

        # Add all nodes first
        for node in list(self.topology['npops'].keys()) + \
                    list(self.topology['physical_switches'].keys()) + \
                    self.topology['endpoints']:
            self.all_nodes.add(node)

        # Build connections
        for link_id, link in self.topology['physical_links'].items():
            src = link['source']
            dst = link['target']
            bw = link['bandwidth']

            # Add bidirectional connections
            self.graph[src].append((dst, bw, link_id))
            self.graph[dst].append((src, bw, link_id))

            self.link_map[(src, dst)] = link_id
            self.link_map[(dst, src)] = link_id

    def find_path(self, source: str, target: str,
                  bandwidth_demand: float,
                  link_bw_available: Dict[int, float]) -> Tuple[List[str], List[int]]:
        """Find shortest path with sufficient bandwidth using proper Dijkstra"""

        # Handle endpoint mapping
        actual_source = self.endpoint_to_switch.get(source, source)
        actual_target = self.endpoint_to_switch.get(target, target)


        if actual_source == actual_target:
            return [source, target], []

        # Dijkstra's algorithm with bandwidth constraint
        dist = {node: float('inf') for node in self.all_nodes}
        prev = {node: None for node in self.all_nodes}
        dist[actual_source] = 0

        pq = [(0, actual_source)]

        while pq:
            current_dist, current = heapq.heappop(pq)

            if current == actual_target:
                break

            if current_dist > dist[current]:
                continue

            for neighbor, link_bw, link_id in self.graph.get(current, []):
                # Check bandwidth availability
                available_bw = link_bw_available.get(link_id, link_bw)
                if available_bw < bandwidth_demand:
                    continue

                new_dist = current_dist + 1  # Hop count as metric

                if new_dist < dist[neighbor]:
                    dist[neighbor] = new_dist
                    prev[neighbor] = (current, link_id)
                    heapq.heappush(pq, (new_dist, neighbor))

        # Reconstruct path
        if dist[actual_target] == float('inf'):
            return None, None

        path_nodes = []
        path_links = []
        current = actual_target

        while current != actual_source:
            path_nodes.append(current)
            if prev[current] is not None:
                prev_node, link_id = prev[current]
                path_links.append(link_id)
                current = prev_node
            else:
                break

        path_nodes.append(actual_source)
        path_nodes.reverse()
        path_links.reverse()

        # Add endpoints if they were mapped
        if source != actual_source:
            path_nodes = [source] + path_nodes
        if target != actual_target:
            path_nodes = path_nodes + [target]

        return path_nodes, path_links

    def get_switches_on_path(self, node_path: List[str]) -> List[str]:
        """Extract physical switches from a node path"""
        switches = []
        for node in node_path:
            if node in self.topology['physical_switches']:
                switches.append(node)
        return switches
