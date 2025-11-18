#!/usr/bin/env python3

"""
Main script for running evaluation campaigns for NOMS'26 network optimization study.
The campaign workflow:
1. Generate network topology using BRITE
  
The script manages parameter combinations and orchestrates the execution flow.
"""

import os
import sys
import shlex
import logging
import argparse
import datetime
import itertools
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import re
import random
from dataclasses import dataclass, asdict, field
import networkx as nx

#para cada switch conecta npop  (100%) (npops=nswitchs)
#sorteia x% dos switchs (50%) conectam ou não com um endpoint (nendpoints = nswitchs/2)

# Default configuration
DEFAULT_VERBOSITY_LEVEL = logging.INFO
TIME_FORMAT = '%Y-%m-%d_%H:%M:%S'
PATH_LOG = 'logs'
PATH_OUTPUTS = 'outputs'
PATHS = [PATH_LOG, PATH_OUTPUTS]
PROBABILITY_ENDPOINT = .5
BANDWIDTH_Mbps_PHYSICAL = 10000 #10Gbps
BANDWIDTH_Mbps_SFC=100 #100 Mbps
# Campaign parameter combinations
#   50 100 200  
#  125 250 500 
#1  10  20  30  
BRITE_NUMBER_OF_NODES = [50, 100]#, 200]#,20, 30,40,50]# , 50, 100, 200]#, 30, 40, 50, 75, 100, 500, 1000]  # (N) Number of nodes 
BRITE_LINKS_PER_NODE = [4]  # (m) Number of links added per new node
NUMBER_SFC = [1, 5, 10, 15, 20, 25] #100, 1000, 10000] #=-1 means equal to number of nodes
# NUMBER_SFC = [30, 35, 40, 45, 50]
TOLERANCE_TEST_NODE_TYPES = 0.1 
# Constants for resource types
NUMBER_OF_STAGES = "NUMBER_OF_STAGES"
COMPUTING_POWER_NORMALIZED = "COMPUTING_POWER_NORMALIZED"
MEMORY_NORMALIZED = "MEMORY_NORMALIZED"


class NetworkNodeType:
    """Base class representing a type of network node with its probability and capacity."""
    # Class-level counter dictionary for each node type
    _type_counters = {}

    def __init__(self, type: str = "", prefix: str= "", probability: float = 0.0, capacity: Optional[Dict[str, Any]] = None):
        self.name = self.__class__.__name__
        self.type = type
        self.prefix = prefix 
        self.probability = probability
        self.capacity = capacity or {}
        # Initialize counter for this type if it doesn't exist
        if self.__class__ not in NetworkNodeType._type_counters:
            NetworkNodeType._type_counters[self.__class__] = 0

    def reset_id(self): 
        NetworkNodeType._type_counters[self.__class__] = 0 

    def get_id(self): 
        # Increment the counter for this specific type
        NetworkNodeType._type_counters[self.__class__] += 1
        return f"{self.prefix}{NetworkNodeType._type_counters[self.__class__]}" 
    
    def __repr__(self):
        return f"{self.__class__.__name__}(probability={self.probability}, capacity={self.capacity})"

class EndPoint(NetworkNodeType):
    """End point node type with no specific capacity requirements."""
    def __init__(self):
        super().__init__(type="endpoint", prefix="e", probability=0.5, capacity={})

class PhysicalSwitch(NetworkNodeType):
    """Physical Programmable switch node type with stage capacity."""
    def __init__(self):
        super().__init__(type="physical_switch", prefix="ps", probability=0.25, capacity={NUMBER_OF_STAGES: 128})

class NPop(NetworkNodeType):
    """Network Point of Presence node type with CPU and memory capacity."""
    def __init__(self):
        super().__init__(type="network_pop", prefix="npop", probability=0.25, capacity={COMPUTING_POWER_NORMALIZED: 1, MEMORY_NORMALIZED: 1})

#NPOP nao pode conectar com Endpoint
#NPOP somentee com switch
#endpoint somente uma aresta 

# Create singleton instances
END_POINT = EndPoint()
PROGRAMMABLE_SWITCH = PhysicalSwitch()
NPOP = NPop()
NETWORK_TYPES = [END_POINT, PROGRAMMABLE_SWITCH, NPOP]

def validate_probabilities(node_types: List[NetworkNodeType]) -> bool:
    """Validate that probabilities sum to 1."""
    total_prob = sum(node.probability for node in node_types)
    return abs(total_prob - 1.0) < 1e-10  # Using small epsilon for float comparison

validate_probabilities(NETWORK_TYPES)

def generate_network_nodes(count: int) -> List[NetworkNodeType]:
    """Generate network nodes based on type probabilities."""
    if not validate_probabilities(NETWORK_TYPES):
        total = sum(node.probability for node in NETWORK_TYPES)
        raise ValueError(f"Network type probabilities must sum to 1.0. Current sum: {total}")
    
    # Create weights list for random.choices
    weights = [node.probability for node in NETWORK_TYPES]
    
    # Generate nodes using probability distribution
    selected_types = random.choices(NETWORK_TYPES, weights=weights, k=count)
    
    # Create new instances of each selected type
    nodes = []
    for i, node_type in enumerate(selected_types):
        # Create a new instance of the same class
        new_node = node_type.__class__()
        nodes.append(new_node)
        logging.debug(f"Generated node {i+1}: {new_node}")
    
    return nodes

# Example usage of network node generation
logging.info("Generating 10 network nodes based on probability distribution")
try:
    generated_nodes = generate_network_nodes(10)
    for i, node in enumerate(generated_nodes, 1):
        logging.info(f"Node {i}: {node}")
except ValueError as e:
    logging.error(f"Error generating nodes: {e}")



class VirtualNetworkType:
    """Class representing a type of Virtual Network Function with its probability and resource requirement."""
    def __init__(self, type:str = "", prefix:str ="", requirement: Optional[Dict[str, Any]] = None):
        self.name = self.__class__.__name__
        self.type = type
        self.prefix = prefix
        self.requirement = requirement or {}
   
    def __repr__(self):
        return f"{self.__class__.__name__}" ##(type='{self.type}', prefix='{self.prefix}')

class VirtualEndPoint(VirtualNetworkType):
    """Virtual Firewall with stage requirement."""
    def __init__(self):
        super().__init__(type="endpoint", prefix="")

class VirtualSwitchINT(VirtualNetworkType):
    """Virtual Firewall with stage requirement."""
    def __init__(self):
        super().__init__(type="virtual_switch", prefix="vs_int", requirement={NUMBER_OF_STAGES: 1})

class VirtualSwitchSCION(VirtualNetworkType):
    """Virtual Firewall with stage requirement."""
    def __init__(self):
        super().__init__(type="virtual_switch", prefix="vs_scion", requirement={NUMBER_OF_STAGES: 1})

class VirtualFirewall(VirtualNetworkType):
    """Virtual Firewall with cpu and memory requirements."""
    def __init__(self):
        super().__init__(
            type="virtual_function", 
            prefix="fw",  # adding prefix for node identification
            requirement={COMPUTING_POWER_NORMALIZED:0.1, MEMORY_NORMALIZED: 0.1}
        )

class VirtualLoadBalancer(VirtualNetworkType):
    """Virtual Load Balancer with cpu and memory requirements."""
    def __init__(self):
        super().__init__(
            type="virtual_function", 
            prefix="lb",  # adding prefix for node identification
            requirement={COMPUTING_POWER_NORMALIZED:0.1, MEMORY_NORMALIZED: 0.1}
        )

class VirtualCaching(VirtualNetworkType):
    """Virtual Caching with cpu and memory requirements."""
    def __init__(self):
        super().__init__(type="virtual_function", prefix="ca",requirement={COMPUTING_POWER_NORMALIZED:0.5, MEMORY_NORMALIZED: 0.5})

# Create singleton instances
VIRTUAL_END_POINT = VirtualEndPoint()
VIRTUAL_SWITCH_INT = VirtualSwitchINT()
VIRTUAL_SWITCH_SCION = VirtualSwitchSCION()
VIRTUAL_FIREWALL = VirtualFirewall()
VIRTUAL_LOAD_BALANCER = VirtualLoadBalancer()
VIRTUAL_CACHING = VirtualCaching()
VN_TYPES = [VIRTUAL_SWITCH_INT, VIRTUAL_SWITCH_SCION, VIRTUAL_FIREWALL, VIRTUAL_LOAD_BALANCER, VIRTUAL_CACHING]
  
  
def number_to_letter_id(n: int) -> str:
    """Convert a number to a letter-based ID (A, B, C, ..., AA, AB, AC, ...)"""
    result = ""
    while n > 0:
        n -= 1
        result = chr(65 + (n % 26)) + result  # 65 is ASCII for 'A'
        n //= 26
    return result

def create_nodes(G: nx.Graph, node_quantities: Dict, number_end_points=1) -> None:
    """Create nodes in the graph according to the specified quantities for each type.
    
    Args:
        G: NetworkX graph to add nodes to
        node_quantities: Dictionary mapping node types to their quantities
    """
    nodes_dict = {}
    count_end_point = 1
    print(node_quantities)
    for node_type, quantity in node_quantities.items():
        for i in range(1, quantity + 1):
            if isinstance(node_type, VirtualEndPoint):
                node_abc = number_to_letter_id(i)
                  
                node_id = f"e{random.randint(1, number_end_points)}"
                tries = 0
                while node_id in nodes_dict.values():
                    node_id = f"e{random.randint(1, number_end_points)}"
                    tries +=1
                    if tries > 10000:
                        #TODO implement a better version of this
                        print("error: num max of tries for alocating virtual end points to real end points.")
                        sys.exit()
                nodes_dict[node_abc] = node_id
                #print(f"END POINT {i} {node_abc} {node_id}")
            else:
                node_id = f"{node_type.prefix}{i}" if quantity > 1 else node_type.prefix
            G.add_node(node_id, type=node_type)
    print(f"create_nodes {nodes_dict}")
     
    return nodes_dict 
 

def print_topology(G: nx.Graph, title: str = "", out_path: str = "", append: bool = False, end_points = None) -> None:
    """Write topology in the specified format to `input_test.txt` (appends)."""
    
    if title:
        header = f"\n{title}\n"
    else:
        header = ""

    # Collect lines in a buffer and write once to file (append)
    lines: List[str] = []
    if header:
        lines.append(header)

    # Find nodes by type
    endpoints = []
    switches = []
    npops = []
    vfws = []
    vlbs = []

    for node, attrs in G.nodes(data=True):
        node_type = attrs.get('type')
        if not node_type:
            continue
         

        if isinstance(node_type, NetworkNodeType):
            if node_type.type == "endpoint":
                endpoints.append(node)
            elif node_type.type == "physical_switch":
                switches.append(node)
            elif node_type.type == "network_pop":
                npops.append(node)

        elif isinstance(node_type, VirtualNetworkType):
            if isinstance(node_type, VirtualFirewall):
                vfws.append(node)
            elif isinstance(node_type, VirtualLoadBalancer):
                vlbs.append(node)
            elif isinstance(node_type, VirtualEndPoint):
                endpoints.append(node)

     
    if end_points != None:
        if end_points != len(endpoints):
            print(f"ERRRO endpoints {end_points} != {len(endpoints)}")
            sys.exit()
    # Nodes
    if endpoints:
        lines.append("endpoint " + " ".join(sorted(endpoints)))
     
    for node in sorted(switches):
        node_type = G.nodes[node]['type']
        lines.append(f"physical_switch {node} {node_type.capacity.get(NUMBER_OF_STAGES, 20)}")

    for node in sorted(npops):
        node_type = G.nodes[node]['type']
        lines.append(f"network_pop {node} {node_type.capacity.get(COMPUTING_POWER_NORMALIZED, 1)} {node_type.capacity.get(MEMORY_NORMALIZED, 1)}")

    for node in sorted(vfws):
        node_type = G.nodes[node]['type']
        lines.append(f"virtual_function fw {node_type.requirement.get(COMPUTING_POWER_NORMALIZED, 0.1)} {node_type.requirement.get(MEMORY_NORMALIZED, 0.1)}")

    for node in sorted(vlbs):
        node_type = G.nodes[node]['type']
        lines.append(f"virtual_function lb {node_type.requirement.get(COMPUTING_POWER_NORMALIZED, 0.1)} {node_type.requirement.get(MEMORY_NORMALIZED, 0.1)}")

    # Edges
    edge_count = 1
    for u, v, attrs in sorted(G.edges(data=True)):
        type =  attrs.get('type', 'N/A')
        bw =   attrs.get('bw', 'N/A')
        switch =  attrs.get('switch', '')
        lines.append(f"{type} {u} {v} {bw} {switch}\t#{edge_count}")
        
        edge_count += 1

    lines.append("")

    # Write (append) to file
    try:
        if out_path != "":
            
            with open(out_path, 'a' if append else 'w') as fh:
                for ln in lines:
                    fh.write(ln + "\n")
                    
        for ln in lines:
                    print(ln)
    except Exception as e:
        logging.error(f"Failed to write topology to {out_path}: {e}")


# SFC Q1 - Load balancer chain

def generate_sfc_q1(number_physical_end_points):
    print("\n\n")
    print("##### TIPO Q1 - 3 end points")

    q1 = nx.Graph()
    end_points = 3
    nodes = {VIRTUAL_END_POINT:end_points, VIRTUAL_FIREWALL:1, VIRTUAL_LOAD_BALANCER:1}
    nodes_dict = create_nodes(q1, nodes, number_physical_end_points)
    print(nodes_dict)
    # Virtual links with comments showing physical path
    q1.add_edge(nodes_dict["A"], "fw",  bw=BANDWIDTH_Mbps_SFC, type="virtual_link", switch=VIRTUAL_SWITCH_INT.prefix)      # e1 -> ps1 -> npop1
    q1.add_edge("fw", "lb",             bw=BANDWIDTH_Mbps_SFC, type="virtual_link", switch=VIRTUAL_SWITCH_INT.prefix)   # npop1 -> ps1 -> ps5 -> ps7 -> ps6 -> ps4 -> npop2
    q1.add_edge("lb", nodes_dict["B"],  bw=BANDWIDTH_Mbps_SFC/2, type="virtual_link", switch=VIRTUAL_SWITCH_INT.prefix)     # npop2 -> ps4 -> e7
    q1.add_edge("lb", nodes_dict["C"],  bw=BANDWIDTH_Mbps_SFC/2, type="virtual_link", switch=VIRTUAL_SWITCH_INT.prefix)     # npop2 -> ps4 -> e8
     
    
    print_topology(q1, end_points=3)
    print("#####")
    return q1



def generate_sfc_q2(number_physical_end_points):
    q2 = nx.Graph()
    end_points = 2
    nodes = {VIRTUAL_END_POINT:end_points, VIRTUAL_FIREWALL:1}
    nodes_dict = create_nodes(q2, nodes, number_physical_end_points)
    q2.add_edge(nodes_dict["A"], "fw", bw=BANDWIDTH_Mbps_SFC, type="virtual_link", switch=VIRTUAL_SWITCH_SCION.prefix)     # e1 -> ps1 -> npop1
    q2.add_edge("fw", nodes_dict["B"], bw=BANDWIDTH_Mbps_SFC, type="virtual_link", switch=VIRTUAL_SWITCH_SCION.prefix)     # npop1 -> ps1 -> e2
    print("##### TIPO Q2 - 2 endpoints")
    print_topology(q2, end_points=2)
    return q2 
 
 

def get_test_instance():
    bw = BANDWIDTH_Mbps_PHYSICAL
    physical_topology_test = nx.Graph()
    end_points = 8
    nodes = {END_POINT:end_points, PROGRAMMABLE_SWITCH:7, NPOP:2}
    create_nodes(physical_topology_test, nodes)

    physical_topology_test.add_edge("e1","ps1", bw=bw, type="link")
    physical_topology_test.add_edge("e2","ps1", bw=bw, type="link")
    physical_topology_test.add_edge("npop1","ps1", bw=bw, type="link")

    physical_topology_test.add_edge("e3","ps2", bw=bw, type="link")
    physical_topology_test.add_edge("e4","ps2", bw=bw, type="link")

    physical_topology_test.add_edge("e5","ps3", bw=bw, type="link")
    physical_topology_test.add_edge("e6","ps3", bw=bw, type="link")

    physical_topology_test.add_edge("e7","ps4", bw=bw, type="link")
    physical_topology_test.add_edge("e8","ps4", bw=bw, type="link")
    physical_topology_test.add_edge("npop2","ps4", bw=bw, type="link")

    physical_topology_test.add_edge("ps1","ps5", bw=bw, type="link")
    physical_topology_test.add_edge("ps2","ps5", bw=bw, type="link")

    physical_topology_test.add_edge("ps3","ps6", bw=bw, type="link")
    physical_topology_test.add_edge("ps4","ps6", bw=bw, type="link")

    physical_topology_test.add_edge("ps5","ps7", bw=bw, type="link")
    physical_topology_test.add_edge("ps6","ps7", bw=bw, type="link")

    q1 = generate_sfc_q1(end_points)
    q2 = generate_sfc_q2(end_points)
    # Print all topologies
    return physical_topology_test, [q1, q2] 
    
  
 

class EvaluationCampaign:
    def __init__(self, output_dir: str, verbosity: int, dryrun: bool = False, test: bool = False, n=None, m=None, sfcs=None, seed=None):
        self.output_dir = output_dir
        self.verbosity = verbosity
        self.dryrun = dryrun
        self.test = test 
        self.n = n 
        self.m = m 
        self.sfcs = sfcs
        self.seed = seed 
         
    def generate_brite_topology(self, n: int, m: int) -> str:
        """
        Generate network topology using BRITE
        
        Args:
            n: Number of nodes
        Returns:
            Path to generated topology file
        """
        logging.info(f"Generating BRITE topology with {n} nodes and {m} links per node")
        output_file = os.path.join(self.output_dir, "tmp", f'topology_seed{self.seed}_n{n}_m{m}.brite')
        config_file_name = os.path.join(self.output_dir, "tmp", f'topology_seed{self.seed}_n{n}_m{m}.conf')
        seed_file_name = os.path.join(self.output_dir, "tmp", f'briteseed_seed{self.seed}_n{n}_m{m}.seed')
        try:
            
            config_file_text = f'''
#This config file was generated by the evaluation campaign. 

BriteConfig

BeginModel
	Name =  2		    #Router Barabasi=2, AS Barabasi = 4
	N = {n}		        #Number of nodes in graph
	HS = 1000		    #Size of main plane (number of squares)
	LS = 100		    #Size of inner planes (number of squares)
	NodePlacement = 1	#Random = 1, Heavy Tailed = 2
	m = {m}			    #Number of neighboring node each new node connects to.
	BWDist = 1		    #Constant = 1, Uniform = 2, HeavyTailed = 3, Exponential = 4
	BWMin = {BANDWIDTH_Mbps_PHYSICAL}
	BWMax = {BANDWIDTH_Mbps_PHYSICAL}
EndModel

BeginOutput
	BRITE = 1 	 #1=output in BRITE format, 0=do not output in BRITE format
	OTTER = 0 	 #1=Enable visualization in otter, 0=no visualization
EndOutput
'''     
            with open(config_file_name, 'w') as config_file:
                config_file.write(config_file_text)
                        
            #cmd = f"cp {config_file_name} BRITE/evaluation_campaign.conf "
            #run_cmd(cmd)

            seed_places = random.getrandbits(64)
            seed_connect = random.getrandbits(64)
            seed_edge_conn = random.getrandbits(64)
            seed_grouping = random.getrandbits(64)
            seed_assignment = random.getrandbits(64)
            seed_bandwidth = random.getrandbits(64)
            seed_file_text = f'''
PLACES {seed_places}	# used when placing nodes on the plane
CONNECT {seed_connect}	# used when interconnecting nodes
EDGE_CONN {seed_edge_conn}	# used in the edge connection method of top down hier
GROUPING {seed_grouping}	# used when deciding which routers to group into an AS in bottom up hier
ASSIGNMENT {seed_assignment}	# used when deciding how many routers to group into an AS in bottom up hier
BANDWIDTH {seed_bandwidth}	# used when assigning bandwidths
'''     
            with open(seed_file_name, 'w') as seed_file:
                seed_file.write(seed_file_text)
                        
            #cmd = f"cp {seed_file_name} BRITE/Java/seed_file "
            #run_cmd(cmd)
             
            # Change to BRITE directory and run the Java command
            current_dir = os.getcwd()
            try:
                configfilepath = os.path.join(os.getcwd(), config_file_name)
                seedfilepath = os.path.join(os.getcwd(), seed_file_name)
                os.chdir('BRITE')
                classpath = os.path.join(os.getcwd(), 'Java')
                
                cmd = f"java -Xmx256M -classpath {classpath} Main.Brite {configfilepath}  network {seedfilepath}"
                run_cmd(cmd, self.dryrun)

                if not self.dryrun:
                    cmd = f"cp network.brite {os.path.join(current_dir,output_file)}"
                    run_cmd(cmd)
                    
            finally:
                os.chdir(current_dir)

           
            
            try:
                G, number_end_points = self.parse_brite_file(output_file, m)
                 
                # Convert node types to strings before saving to GEXF
                G_save = G.copy()
                for node, data in G_save.nodes(data=True):
                    if 'type' in data:
                        data['type'] = str(data['type'])
                gexf_path = os.path.join(self.output_dir, "tmp", f'topology_seed{self.seed}_n{n}_m{m}.gexf')
                nx.write_gexf(G_save, gexf_path)
                logging.info(f"Saved parsed topology to {gexf_path} (nodes={G.number_of_nodes()} edges={G.number_of_edges()})")
            except Exception as e:
                logging.error(f"Failed to parse BRITE output into graph: {e}")
                sys.exit()

            return G, number_end_points
        
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to generate BRITE topology: {e}")
            raise
      
    def assert_distribution_within_tolerance(self, G, tolerance=TOLERANCE_TEST_NODE_TYPES):
        """
        Assert that the actual distribution of network types is within the specified tolerance
        of the expected probabilities.
        
        Args:
            G: NetworkX graph with node types
            NETWORK_TYPES: List of network type objects with probability attributes
            tolerance: Allowed deviation from expected probability (default: 0.02 for 2%)
        
        Raises:
            AssertionError: If any network type distribution is outside the tolerance
        """
        # Count occurrences of each network type
        type_counts = {}
        total_nodes = len(G.nodes())
        
        if total_nodes == 0:
            raise ValueError("Graph has no nodes")
        
        for node_id, node_data in G.nodes(data=True):
            node_type = node_data['type']
            type_name = node_type.__class__.__name__
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        # Check each type against expected distribution
        violations = []
        
        for nt in NETWORK_TYPES:
            expected_prob = nt.probability
            type_name = nt.__class__.__name__
            actual_count = type_counts.get(type_name, 0)
            actual_prob = actual_count / total_nodes
            
            difference = abs(actual_prob - expected_prob)
            
            if difference > tolerance:
                violations.append({
                    'type': type_name,
                    'expected': expected_prob,
                    'actual': actual_prob,
                    'difference': difference
                })
        
        # Build assertion message if there are violations
        if violations:
            error_msg = f"Network type distribution outside {tolerance*100}% tolerance:\n"
            for violation in violations:
                error_msg += (
                    f"  {violation['type']}: Expected {violation['expected']:.3f}, "
                    f"Got {violation['actual']:.3f} "
                    f"(Difference: {violation['difference']:.3f})\n"
                )
            raise AssertionError(error_msg)
        
        # Optional: Print success message
        print(f"✓ All network types within {tolerance*100}% tolerance")
        for nt in NETWORK_TYPES:
            type_name = nt.__class__.__name__
            actual_prob = type_counts.get(type_name, 0) / total_nodes
            print(f"  {type_name}: Expected {nt.probability:.3f}, Got {actual_prob:.3f}")

  

    def parse_brite_file(self, brite_file: str, links_per_node) -> nx.Graph:
        """
        Robust, tolerant parser for BRITE files.

        This parser handles multiple BRITE variants by:
        - parsing NODE and EDGE lines (case-insensitive)
        - treating lines starting with two integers as edges
        - extracting key=value attributes when present (e.g., BW=10)
        - keeping any remaining tokens in a generic attributes dict

        Returns a networkx.Graph with node attributes and edge attributes (may include 'bw').
        """
        PhysicalSwitch().reset_id()
        EndPoint().reset_id()
        NPop().reset_id() 

        logging.info(f"Parsing BRITE file (strict format) into graph: {brite_file}")
        G = nx.Graph()

        if not os.path.isfile(brite_file):
            raise FileNotFoundError(f"BRITE file not found: {brite_file}")

        with open(brite_file, 'r') as fh:
            lines = [l.rstrip('\n') for l in fh]

        # helper to get next non-empty line starting from index
        def next_non_empty(start_idx):
            i = start_idx
            while i < len(lines) and lines[i].strip() == '':
                i += 1
            return i

        idx = next_non_empty(0)
        if idx >= len(lines):
            logging.error("BRITE file empty or malformed")
            return G

        topology_info = lines[idx].strip()
        print(f"Topology info: {topology_info}")
        n_nodes = topology_info.split(" ")[2]
        n_edges = topology_info.split(" ")[4]
        print(f"Topology info nodes: {n_nodes} edges: {n_edges}")
        idx += 1

        # second line: model info
        idx = next_non_empty(idx)
        if idx >= len(lines):
            logging.error("BRITE file missing model line")
            return G
        model_info = lines[idx].strip()
        idx += 1

        # find Nodes: line
        idx = next_non_empty(idx)
        if idx >= len(lines) or not lines[idx].strip().lower().startswith('nodes:'):
            logging.error("BRITE file missing 'Nodes:' section")
            return G

        # parse number of nodes
        try:
            line =  lines[idx] 
            print(f"Nodes line: {line}")
            nodes_count = line.split(" ")[2]
             
            print(f"Nodes count: {nodes_count}")
            assert(nodes_count==n_nodes)
            #nodes_count = int(re.split(r"\s*:\s*", lines[idx].strip(), maxsplit=1)[1])
        except Exception:
            logging.error("Cannot parse number of nodes from 'Nodes:' line")
            return G
        idx += 1

        number_end_points = 0
        nodes_count = int(nodes_count)
        nodes_translate = {}

        # Prepara a lista circular com distribuição balanceada
        total_nodes = nodes_count
        node_counts = {}

        for nt in NETWORK_TYPES:
            expected_count = int(nt.probability * total_nodes)
            node_counts[nt] = expected_count

        # Ajusta para total exato (pode haver arredondamento)
        total_allocated = sum(node_counts.values())
        if total_allocated < total_nodes:
            # Distribui os nós restantes pelos tipos com maior probabilidade
            remaining = total_nodes - total_allocated
            sorted_types = sorted(NETWORK_TYPES, key=lambda x: x.probability, reverse=True)
            for i in range(remaining):
                node_counts[sorted_types[i]] += 1

        # Cria lista circular com a distribuição exata
        circular_list = []
        for nt, count in node_counts.items():
            circular_list.extend([nt] * count)

        random.shuffle(circular_list)  # Embaralha para não ser previsível
        node_type_iterator = iter(circular_list)  # Cria um iterador

        
        nodes_count = int(nodes_count)
        nodes_translate = {}
        # parse nodes_count node lines
        brite_nodes = {}
        switches = []
        endpoints = []
        npops = [] 

        for n in range(int(nodes_count)):
            #https://open.bu.edu/server/api/core/bitstreams/84ade824-a0af-40a4-8674-b9107b1dbe40/content
            # Field Meaning
            # NodeId Unique id for each node
            # xpos x-axis coordinate in the plane
            # ypos y-axis coordinate in the plane
            # indegree Indegree of the node
            # outdegree Outdegree of the node
            # ASid id of the AS this node belongs to (if hierarchical)
            # type Type assigned to the node (e.g. router, AS)

            print(f"\t Node {n}")
            idx = next_non_empty(idx)
            if idx >= len(lines):
                logging.warning(f"Expected {nodes_count} node lines but file ended at {n}")
                break
            line = lines[idx].strip()
            print(f"\t line-node: {line}")
            idx += 1

            toks = re.split(r'\s+', line)
            if len(toks) < 6:
                logging.error(f"Unexpected node line format: '{line}'")
                sys.exit()

            # parse: id, x, y, v1, v2, v3, [type]
            try:
                bid = int(toks[0])
            except Exception:
                logging.warning(f"Node id not integer in line: '{line}'")
                continue

            try:
                x = float(toks[1])
                y = float(toks[2])
            except Exception:
                x = None
                y = None

            extra_vals = []
            for j in range(3, 6):
                if j < len(toks):
                    try:
                        v = float(toks[j])
                    except Exception:
                        v = toks[j]
                    extra_vals.append(v)
                else:
                    extra_vals.append(None)

            node_type_str = toks[6] if len(toks) >= 7 else ''

            brite_nodes[bid] = {
                'brite_id': bid,
                'x': x,
                'y': y,
                'extra': extra_vals,
                'brite_type': node_type_str,
            }

            # # assign our network node type from circular list
            # try:
            #     proto = next(node_type_iterator)
            #     our_nt = proto.__class__()
            # except StopIteration:
            #     # Fallback caso algo dê errado com a lista circular
            #     weights = [nt.probability for nt in NETWORK_TYPES]
            #     proto = random.choices(NETWORK_TYPES, weights=weights, k=1)[0]
            #     our_nt = proto.__class__()
            
            our_nt = PhysicalSwitch().__class__()

            node_id = our_nt.get_id()
            nodes_translate[bid] = node_id
            print(f"\t id:{node_id} bid: {bid} type:{our_nt}")
            
            switches.append(node_id)
            
            G.add_node(node_id, brite=brite_nodes[bid], type=our_nt)

        #self.assert_distribution_within_tolerance(G)
         
        # after nodes, expect possibly blank lines then Edges:
        idx = next_non_empty(idx)
        if idx >= len(lines) or not lines[idx].strip().lower().startswith('edges:'):
            logging.warning("BRITE file missing 'Edges:' section or it is after nodes block")
            return G

        try:
            line = lines[idx]
            print(f"\n\nheader-edge {line}\n")
            edges_count = line.split(" ")[2]
            assert(edges_count== n_edges)
            edges_count = int(edges_count)
            #edges_count = int(re.split(r"\s*:\s*", lines[idx].strip(), maxsplit=1)[1])
        except Exception:
            logging.error("Cannot parse number of edges from 'Edges:' line")
            return G
        idx += 1
        
        added_edges = 0
        skipped_edges = 0

         
        for e in range(edges_count):
            #print(f"\t edge: {e}")
            idx = next_non_empty(idx)
            if idx >= len(lines):
                logging.warning(f"Expected {edges_count} edge lines but file ended at {e}")
                break
            line = lines[idx].strip()
            print(f"\t edge: {e} line: {line}")
            idx += 1

            toks = re.split(r'\s+', line)
            if len(toks) < 3:
                logging.warning(f"Unexpected edge line format: '{line}'")
                continue
                    
            #https://open.bu.edu/server/api/core/bitstreams/84ade824-a0af-40a4-8674-b9107b1dbe40/content
            #Field Meaning
            # EdgeId Unique id for each edge
            # from node id of source
            # to node id of destination
            # length Euclidean length
            # delay propagation delay
            # bandwidth bandwidth (assigned by AssignBW method)
            # ASfrom if hierarchical topology, AS id of source node
            # ASto if hierarchical topology, AS id of destination node
            # type Type assigned to the edge by classification routine
            try:
                eid = int(toks[0])
                src = int(toks[1])
                
                src = nodes_translate[src]
                dst = int(toks[2])
                dst = nodes_translate[dst]
                 
                brite_bw = toks[5]

            except Exception:
                logging.warning(f"Edge ids not integers in line: '{line}'")
                continue

            brite_edge_vals = toks[3:]

            # ensure nodes exist (they should)
            if not G.has_node(src):
                # # create node with assigned type
                # weights = [nt.probability for nt in NETWORK_TYPES]
                # proto = random.choices(NETWORK_TYPES, weights=weights, k=1)[0]
                # G.add_node(src, brite={'brite_id': src}, type=proto.__class__())
                logging.error(f"nodo src not found: {src}")
                sys.exit()

            if not G.has_node(dst):
                # weights = [nt.probability for nt in NETWORK_TYPES]
                # proto = random.choices(NETWORK_TYPES, weights=weights, k=1)[0]
                # G.add_node(dst, brite={'brite_id': dst}, type=proto.__class__())
                logging.error(f"nodo dst not found: {dst}")
                sys.exit()
            print()
            print(f"edge from {src} to {dst}" )
            type_source = G.nodes[src].get('type')
            type_destiny = G.nodes[dst].get('type')

            print(f"\t\ttype source:{type_source}")
            print(f"\t\ttype destiny:{type_destiny}")
            # enforce rules
           
            # add edge with brite raw values and our standardized attrs
            attrs = {
                'brite': {'eid': eid, 'values': brite_edge_vals},
                'bw': brite_bw,
                'type': 'link'
            }
            G.add_edge(src, dst, **attrs)
            added_edges += 1

        number_end_points = 0 
        print(f"adding npops (one per switch)...")
        npop_type = NPop().__class__()
        for switch in switches:
            print(f"\t switch: {switch}")
            npop_id = npop_type.get_id()
            G.add_node(npop_id, brite={}, type=npop_type)
            G.add_edge(switch, npop_id, **attrs)
            added_edges += 1
            print(f"\t\t node npop: {npop_id}")

        target_endpoints = int(len(switches) * PROBABILITY_ENDPOINT)
        available_switches = switches.copy()
        print(f"adding endpoints... taget:{target_endpoints} considering {len(switches)} switches and {PROBABILITY_ENDPOINT} probability.")

        endpoint_type = EndPoint().__class__()
        while number_end_points < target_endpoints and available_switches:
                switch = random.choice(available_switches)
                endpoint_id = endpoint_type.get_id()
                G.add_node(endpoint_id, brite={}, type=endpoint_type)
                G.add_edge(switch, endpoint_id, **attrs)
                
                added_edges += 1
                number_end_points += 1 
                print(f"\t\t node endpoint: {endpoint_id}")
       

        print(f"Parsed BRITE strict: nodes={G.number_of_nodes()} edges_added={added_edges} ")
        # also attach file-level metadata
        G.graph['brite_topology_info'] = topology_info
        G.graph['brite_model_info'] = model_info
        G.graph['brite_nodes_declared'] = nodes_count
        G.graph['brite_edges_declared'] = edges_count

        return G, number_end_points

    def _chain_to_names(self, chain_item) -> Any:
        """Convert chain item (which may be NetworkType, VNFType or nested lists) to a serializable form (names)."""
        if isinstance(chain_item, list):
            return [self._chain_to_names(x) for x in chain_item]
        return getattr(chain_item, 'name', str(chain_item))

    def generate_sfc_instances(self, count: int, number_end_points) -> List:
        """Generate `count` concrete SFC instances sampled from SFC_TYPES.

        Returns a list of SFCInstance objects. Saves JSON file to output directory when not dryrun.
        """
        logging.info(f"Generating {count} SFC instances from templates")
        instances: List = []
        # Lista com todos os métodos
        methods = [generate_sfc_q1]#, generate_sfc_q2]

        for i in range(count):
            method_chosen = random.choice(methods)
            instances.append(method_chosen(number_end_points))
        
        return instances

    def solve_naive(self, topology_file: str, m: int) -> Dict:
        """
        Create solution using naive approach
        
        Args:
            topology_file: Path to BRITE topology file
            m: Number of machines
        Returns:
            Results dictionary with performance metrics
        """
        logging.info(f"Running naive solution with {m} machines")
        # TODO: Implement naive solution algorithm
        return {
            'algorithm': 'naive',
            'runtime': 0,
            'objective_value': 0,
            'solution': {}
        }

    def solve_cplex(self, topology_file: str, m: int) -> Dict:
        """
        Create solution using CPLEX optimization
        
        Args:
            topology_file: Path to BRITE topology file
            m: Number of machines
        Returns:
            Results dictionary with performance metrics
        """
        logging.info(f"Running CPLEX optimization with {m} machines")
        # TODO: Implement CPLEX optimization
        return {
            'algorithm': 'cplex',
            'runtime': 0,
            'objective_value': 0,
            'solution': {}
        }

    def solve_ml(self, topology_file: str, m: int) -> Dict:
        """
        Create solution using machine learning approach
        
        Args:
            topology_file: Path to BRITE topology file
            m: Number of machines
        Returns:
            Results dictionary with performance metrics
        """
        logging.info(f"Running ML solution with {m} machines")
        # TODO: Implement machine learning solution
        return {
            'algorithm': 'ml',
            'runtime': 0,
            'objective_value': 0,
            'solution': {}
        }

    def update_plots(self, results: List[Dict]):
        """
        Update performance metric plots
        
        Args:
            results: List of result dictionaries from different solutions
        """
        logging.info("Updating performance plots")
        # TODO: Implement plotting functionality
        pass

    # def run_single_evaluation(self, n: int, m: int, sfcs: int) -> List[Dict]:
    #     """
    #     Run a single evaluation with given parameters
        
    #     Args:
    #         n: Number of nodes (-1 indicates test mode)
    #         m: Number of machines
    #     Returns:
    #         List of results from different solution approaches
    #     """
         
    #     if n == -1: 
    #         topology, sfc_instances = get_test_instance() 
    #         topology_file = os.path.join(self.output_dir, f'topology_test_seed{self.seed}.txt')
    #     else: 
    #         topology, number_end_points = self.generate_brite_topology(n, m)
    #         # generate SFC instances to map onto the substrate
             
    #         sfc_instances = self.generate_sfc_instances(sfcs, number_end_points)
    #         logging.info(f"Generated {len(sfc_instances)} SFC instances for this evaluation")
    #         sfcs = len(sfc_instances)
    #         topology_file = os.path.join(self.output_dir, f'topology_seed{self.seed}_nodes{n:05d}_links{m:02d}_sfcs{sfcs:05d}.txt')

    #     print_topology(topology, "Topology", topology_file, append=False)
    #     for count, sfc in enumerate(sfc_instances, start=1):
    #         print_topology(sfc, f"SFC Q{count}", topology_file, append=True)

    #     results = []
        
    #     # Run each solution approach
    #     solution_methods = [
    #         # ('naive', self.solve_naive),
    #         # ('cplex', self.solve_cplex),
    #         # ('ml', self.solve_ml)
    #     ]
        
    #     for name, method in solution_methods:
    #         try:
    #             start_time = datetime.datetime.now()
    #             result = method(topology, m)
    #             end_time = datetime.datetime.now()
    #             result['duration'] = (end_time - start_time).total_seconds()
    #             results.append(result)
    #             logging.info(f"{name} solution completed in {result['duration']}s")
    #         except Exception as e:
    #             logging.error(f"Error in {name} solution: {e}")
                
    #     return results

    def run_campaign(self):
        """Run the complete evaluation campaign"""
        campaign_start = datetime.datetime.now()
        logging.info("Starting evaluation campaign")
       
        if self.test: 
            topology, sfc_instances = get_test_instance() 
            topology_file = os.path.join(self.output_dir, f'topology_test.txt')
            print_topology(topology, "Topology", topology_file, append=False)
            for count, sfc in enumerate(sfc_instances, start=1):
                print_topology(sfc, f"SFC Q{count}", topology_file, append=True)
        else:
            ns = self.n 
            ms = self.m 
            

            # Iterate through all parameter combinations
            for n in ns:
                for m in ms:
                    print(f"generating network with {n} nodes and {m} edges")
                    
                    sfcs = [x if x != -1 else n for x in self.sfcs]
                    topology, number_end_points = self.generate_brite_topology(n, m)
                    # generate SFC instances to map onto the substrate  
                    
                    sfc_instances = self.generate_sfc_instances(max(sfcs), number_end_points)
                    print(f"Generated {len(sfc_instances)} SFC instances for this evaluation")
                    
                    for number_sfc in sfcs:
                        if number_sfc == -1:
                            number_sfc = n 
                         
                        topology_file = os.path.join(self.output_dir, f'topology_seed{self.seed}_nodes{n:05d}_links{m:02d}_sfcs{number_sfc:05d}.txt')
                        
                        print_topology(topology, "Topology", topology_file, append=False)
                        for count, instance in enumerate(sfc_instances, start=1):
                            if count > number_sfc:
                                break
                            else:
                                print_topology(instance, f"SFC Q{count}", topology_file, append=True)
                            
        
         

def run_cmd(cmd: str, dryrun: bool = False) -> None:
    """
    Execute a shell command with dryrun support
    
    Args:
        cmd: Command to execute
        dryrun: If True, only print the command without executing
    """
    print(f"Command: {cmd}")
    if not dryrun:
        subprocess.run(shlex.split(cmd), check=True)
    else:
        print("[DRYRUN] Command would be executed here")

def main():
    parser = argparse.ArgumentParser(
        description='NOMS\'26 Network Optimization Evaluation Campaign'
    )
    
    parser.add_argument(
        "--verbosity",
        "-v",
        help="Logging verbosity level",
        default=DEFAULT_VERBOSITY_LEVEL,
        type=int
    )
    
    parser.add_argument(
        "--output",
        "-o",
        help="Output directory",
        #default=f'outputs/campaign_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
        default=f'outputs/'
    )
    
    parser.add_argument(
        "--dryrun",
        "-d",
        help="Show commands without executing them",
        action='store_true',
        default=False
    )

    parser.add_argument(
        "--nodes",
        "-n",
        help="BRITE's number of nodes (N). Can provide multiple values.",
        type=int,
        nargs='+',
        default=BRITE_NUMBER_OF_NODES
    )

    parser.add_argument(
        "--edges",
        "-m",
        help="BRITE's number of edges per link (m). Can provide multiple values.",
        type=int,
        nargs='+',
        default=BRITE_LINKS_PER_NODE
    )

    parser.add_argument(
        "--sfcs",
        "-s",
        help="Number of sfcs. Can provide multiple values. -1 means = nodes",
        type=int,
        nargs='+',
        default=NUMBER_SFC
    )

    parser.add_argument(
        "--test",
        "-t",
        help="Run the test case.",
        action='store_true',
        default=False
    )

    parser.add_argument(
        "--seed",
        help="Random seed for reproducible results",
        type=int,
        default=None
    )
   

    args = parser.parse_args()
    
    if args.seed != None:
        random.seed(args.seed)
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output,"tmp"), exist_ok=True)
    
    # Run campaign
    campaign = EvaluationCampaign(args.output, args.verbosity, args.dryrun, args.test, args.nodes, args.edges, args.sfcs, args.seed)
    campaign.run_campaign()

if __name__ == '__main__':
    sys.exit(main())