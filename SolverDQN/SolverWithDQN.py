import argparse
import copy
import logging
import random
import sys
import time
import traceback
from collections import deque
from typing import Dict

import numpy
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv

from Components.Model.Agent.SFCDQN import SFCDQN
from Components.Model.Agent.StateEncoder import StateEncoder
from Components.Model.Environment.Environment import SFCEnvironment
from Components.Network.NetworkParser import NetworkParser

# Logging configuration
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

# Default hyperparameters
BATCH_SIZE = 64
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.9975
LEARNING_RATE = 0.0003
MEMORY_SIZE = 50000
TARGET_UPDATE = 5
NUM_EPISODES = 600
TRAIN_FREQ = 4
GAMMA = 0.90

def format_time(seconds):
    """Convert seconds to readable format"""
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
        return f"{hours} hora(s), {minutes} minuto(s) e {secs:.2f} segundos"


def create_arg_parser():
    arg_parser = argparse.ArgumentParser(
        description="SFC DQN SOLVER - Deep Q-Learning Network for Service Function Chain Placement",
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

    # Required input file
    arg_parser.add_argument('--input', '-i', required=True, help='Input network topology file')

    # Optional output file
    arg_parser.add_argument(
        '--output', '-o',
        nargs='?',
        default='dqn_results.json',
        help='Output JSON file (default: dqn_results.json)'
    )

    # DQN Training parameters
    arg_parser.add_argument(
        '--episodes', '-e',
        type=int,
        default=NUM_EPISODES,
        help=f'Number of training episodes (default: {NUM_EPISODES})'
    )

    arg_parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=BATCH_SIZE,
        help=f'Batch size for training (default: {BATCH_SIZE})'
    )

    arg_parser.add_argument(
        '--learning-rate', '-lr',
        type=float,
        default=LEARNING_RATE,
        help=f'Learning rate (default: {LEARNING_RATE})'
    )

    arg_parser.add_argument(
        '--gamma', '-g',
        type=float,
        default=GAMMA,
        help=f'Discount factor (default: {GAMMA})'
    )

    arg_parser.add_argument(
        '--epsilon-start',
        type=float,
        default=EPSILON_START,
        help=f'Initial exploration rate (default: {EPSILON_START})'
    )

    arg_parser.add_argument(
        '--epsilon-end',
        type=float,
        default=EPSILON_END,
        help=f'Final exploration rate (default: {EPSILON_END})'
    )

    arg_parser.add_argument(
        '--epsilon-decay',
        type=float,
        default=EPSILON_DECAY,
        help=f'Exploration decay rate (default: {EPSILON_DECAY})'
    )

    arg_parser.add_argument(
        '--memory-size',
        type=int,
        default=MEMORY_SIZE,
        help=f'Replay memory size (default: {MEMORY_SIZE})'
    )

    arg_parser.add_argument(
        '--target-update',
        type=int,
        default=TARGET_UPDATE,
        help=f'Target network update frequency (default: {TARGET_UPDATE})'
    )

    arg_parser.add_argument(
        '--train-freq',
        type=int,
        default=TRAIN_FREQ,
        help=f'Training frequency in steps (default: {TRAIN_FREQ})'
    )

    # Device selection
    arg_parser.add_argument(
        '--device',
        choices=['cpu', 'cuda', 'auto'],
        default='auto',
        help='Device to use for training (default: auto)'
    )

    # Random seed
    arg_parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )

    try:
        args = arg_parser.parse_args()
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)

    return args


def print_all_settings(arguments):
    """Logs the full configuration of the current experiment based on parsed arguments."""
    print(f"Command:\n\t{' '.join(sys.argv)}\n")
    print(f"Settings:")
    lengths = [len(x) for x in vars(arguments).keys()]
    max_length = max(lengths)

    for key_item, values in sorted(vars(arguments).items()):
        message = "\t"
        message += key_item.ljust(max_length, " ")
        message += f" : {values}"
        print(message)

    print("")


def set_random_seed(seed):
    """Set random seed for reproducibility"""
    if seed is not None:
        random.seed(seed)
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        logger.info(f"Random seed set to: {seed}")


class DQNAgent:

    def __init__(self, infra_feat_dim: int, sfc_feat_dim: int, num_npops: int, args):
        self.num_npops = num_npops
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.train_freq = args.train_freq

        state_encoder = StateEncoder(infra_feat_dim, sfc_feat_dim)
        self.policy_net = SFCDQN(state_encoder).to(DEVICE)
        self.target_net = SFCDQN(StateEncoder(infra_feat_dim, sfc_feat_dim)).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=args.learning_rate, weight_decay=1e-5)
        self.memory = deque(maxlen=args.memory_size)

        self.epsilon = args.epsilon_start
        self.epsilon_end = args.epsilon_end
        self.epsilon_decay = args.epsilon_decay
        self.steps = 0

    def select_action(self, state, valid_actions):
        if random.random() < self.epsilon:
            valid_indices = torch.where(valid_actions)[0]
            if len(valid_indices) == 0:
                return 0
            return random.choice(valid_indices.tolist())
        else:
            with torch.no_grad():
                q_values = self.policy_net(
                    state['infra_data'],
                    state['sfc_data'],
                    state['vnf_idx'],
                    state['num_npops'],
                    valid_actions
                )
                return q_values.argmax().item()

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return 0.0

        batch = random.sample(self.memory, self.batch_size)
        losses = []

        for state, action, reward, next_state, done in batch:
            q_values = self.policy_net(
                state['infra_data'],
                state['sfc_data'],
                state['vnf_idx'],
                state['num_npops']
            )
            q_value = q_values[action]

            if done or next_state is None:
                target_q = reward
            else:
                with torch.no_grad():
                    next_q_policy = self.policy_net(
                        next_state['infra_data'],
                        next_state['sfc_data'],
                        next_state['vnf_idx'],
                        next_state['num_npops']
                    )
                    best_action = next_q_policy.argmax()

                    next_q_target = self.target_net(
                        next_state['infra_data'],
                        next_state['sfc_data'],
                        next_state['vnf_idx'],
                        next_state['num_npops']
                    )
                    target_q = reward + self.gamma * next_q_target[best_action]

            loss = F.smooth_l1_loss(q_value, torch.tensor(target_q, device=DEVICE, dtype=torch.float32))
            losses.append(loss)

        if losses:
            total_loss = torch.stack(losses).mean()
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
            self.optimizer.step()
            return total_loss.item()

        return 0.0

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


class SFCDQNSolver:
    """Main solver class"""

    def __init__(self, args):
        self.topology = None
        self.agent = None
        self.parser = None
        self.args = args

    def load_topology_from_file(self, filename: str):
        """Load topology from file"""
        logger.info(f"\n{'=' * 80}")
        logger.info("LOADING TOPOLOGY")
        logger.info(f"{'=' * 80}")

        self.parser = NetworkParser()
        self.parser.parse_file(filename)

        logger.info(f"✓ Endpoints: {len(self.parser.endpoints)}")
        logger.info(f"✓ Physical Switches: {len(self.parser.physical_switches)}")
        logger.info(f"✓ Network PoPs: {len(self.parser.npops)}")
        logger.info(f"✓ Physical Links: {len(self.parser.links)}")
        logger.info(f"✓ SFC Requests: {len(self.parser.sfc_requests)}")
        logger.info(
            f"✓ Total Graph Nodes: {len(self.parser.npops) + len(self.parser.physical_switches) + len(self.parser.endpoints)}")

        self._build_topology_dict()
        return self

    def _build_topology_dict(self):
        """Build topology dictionary from parser"""
        self.topology = {
            'endpoints': self.parser.endpoints,
            'npops': {},
            'physical_switches': {},
            'physical_links': {},
            'sfc_requests': []
        }

        for name, cpu, mem in self.parser.npops:
            self.topology['npops'][name] = {
                'name': name,
                'cpu_capacity': cpu,
                'mem_capacity': mem
            }

        for name, capacity in self.parser.physical_switches:
            self.topology['physical_switches'][name] = {
                'name': name,
                'stage_capacity': capacity
            }

        for link_id, src, tgt, bw in self.parser.links:
            self.topology['physical_links'][link_id] = {
                'id': link_id,
                'source': src,
                'target': tgt,
                'bandwidth': bw
            }

        for sfc in self.parser.sfc_requests:
            sfc_dict = {
                'name': sfc.name,
                'endpoints': sfc.endpoints,
                'vnfs': [
                    {
                        'name': vf.name,
                        'cpu_demand': vf.cpu_demand,
                        'mem_demand': vf.mem_demand,
                        'assigned_npop': vf.assigned_npop
                    }
                    for vf in sfc.vnfs
                ],
                'virtual_switches': [
                    {
                        'name': vs.name,
                        'stage_demand': vs.stage_demand
                    }
                    for vs in sfc.virtual_switches
                ],
                'virtual_links': [
                    {
                        'source': vl.source,
                        'target': vl.target,
                        'bandwidth_demand': vl.bandwidth_demand,
                        'vs_type': vl.vs_type
                    }
                    for vl in sfc.virtual_links
                ]
            }
            self.topology['sfc_requests'].append(sfc_dict)

    def train(self, num_episodes: int = None):
        """Train DQN agent"""
        if num_episodes is None:
            num_episodes = self.args.episodes

        train_start_time = time.time()

        logger.info(f"\n{'=' * 80}")
        logger.info(f"✅ ENHANCED STATE REPRESENTATION")
        logger.info(f"   - Infrastructure: Global + Current SFC resource usage")
        logger.info(f"   - SFC: VNF placement history + location tracking")
        logger.info(f"   - VS: shared (consumed once)")
        logger.info(f"   - VNF: accumulated (each SFC adds demand)")
        logger.info(f"   - VL: accumulated (each SFC adds bandwidth)")
        logger.info(f"{'=' * 80}")
        logger.info(f"Device: {DEVICE}")
        logger.info(f"Episodes: {num_episodes}")

        infra_feat_dim = 8
        sfc_feat_dim = 8
        num_npops = len(self.topology['npops'])

        self.agent = DQNAgent(infra_feat_dim, sfc_feat_dim, num_npops, self.args)

        episode_rewards = []
        episode_accepted = []
        episode_npops = []
        episode_losses = []
        episode_vnf_instances = []
        episode_vs_instances = []

        best_acceptance = 0
        best_model = None
        best_consolidation = float('inf')

        reward_ma = deque(maxlen=50)
        acceptance_ma = deque(maxlen=50)
        npops_ma = deque(maxlen=50)

        logger.info(f"\n{'=' * 80}")
        logger.info("TRAINING PROGRESS")
        logger.info(f"{'=' * 80}\n")

        for episode in range(num_episodes):
            env = SFCEnvironment(self.topology)
            state = env.reset()

            episode_reward = 0
            episode_loss = 0
            steps = 0

            while state is not None:
                valid_actions = env.get_valid_actions()
                if valid_actions is None or valid_actions.sum() == 0:
                    break

                action = self.agent.select_action(state, valid_actions)
                next_state, reward, done = env.step(action)

                self.agent.store_experience(state, action, reward, next_state, done)

                episode_reward += reward
                steps += 1

                if steps % self.agent.train_freq == 0:
                    loss = self.agent.train_step()
                    if loss > 0:
                        episode_loss += loss

                state = next_state

                if done:
                    break

            results = env.get_results()
            episode_rewards.append(episode_reward)
            episode_accepted.append(results['accepted_sfcs'])
            episode_npops.append(results['npops_used'])
            episode_losses.append(episode_loss / max(1, steps // self.agent.train_freq))
            episode_vnf_instances.append(results['vnf_instances'])
            episode_vs_instances.append(results['vs_instances'])

            reward_ma.append(episode_reward)
            acceptance_rate = results['accepted_sfcs'] / results['total_sfcs']
            acceptance_ma.append(acceptance_rate)
            npops_ma.append(results['npops_used'])

            self.agent.decay_epsilon()

            if (episode + 1) % self.args.target_update == 0:
                self.agent.update_target_network()

            current_acceptance = numpy.mean(list(acceptance_ma))
            current_npops = numpy.mean(list(npops_ma))

            if current_acceptance > best_acceptance or \
                    (current_acceptance == best_acceptance and current_npops < best_consolidation):
                best_acceptance = current_acceptance
                best_consolidation = current_npops
                best_model = copy.deepcopy(self.agent.policy_net.state_dict())

            if (episode + 1) % max(1, num_episodes // 20) == 0:
                avg_reward = numpy.mean(list(reward_ma))
                avg_acceptance = numpy.mean(list(acceptance_ma))
                avg_npops = numpy.mean(list(npops_ma))
                avg_loss = numpy.mean(episode_losses[-50:]) if episode_losses else 0
                avg_vnf_inst = numpy.mean(episode_vnf_instances[-50:]) if episode_vnf_instances else 0
                avg_vs_inst = numpy.mean(episode_vs_instances[-50:]) if episode_vs_instances else 0

                logger.info(f"Episode {episode + 1}/{num_episodes}")
                logger.info(f"  Reward MA(50): {avg_reward:.2f}")
                logger.info(f"  Acceptance MA(50): {avg_acceptance * 100:.1f}%")
                logger.info(f"  NPoPs MA(50): {avg_npops:.1f}")
                logger.info(f"  VNF Instances MA(50): {avg_vnf_inst:.1f}")
                logger.info(f"  VS Instances MA(50): {avg_vs_inst:.1f}")
                logger.info(f"  Loss MA(50): {avg_loss:.4f}")
                logger.info(f"  Epsilon: {self.agent.epsilon:.4f}")
                logger.info(f"  Best: {best_acceptance * 100:.1f}% acceptance, {best_consolidation:.1f} NPoPs")
                logger.info("")

        if best_model is not None:
            self.agent.policy_net.load_state_dict(best_model)
            logger.info(f"✓ Loaded best model: {best_acceptance * 100:.1f}% acceptance")

        train_duration = time.time() - train_start_time

        logger.info(f"\n{'=' * 80}")
        logger.info("TRAINING COMPLETED")
        logger.info(f"{'=' * 80}")
        logger.info(f"Time: {format_time(train_duration)}")
        logger.info(f"Best acceptance: {best_acceptance * 100:.1f}%")
        logger.info(f"Best consolidation: {best_consolidation:.1f} NPoPs")

        return {
            'episode_rewards': episode_rewards,
            'episode_accepted': episode_accepted,
            'episode_losses': episode_losses,
            'episode_npops_used': episode_npops,
            'episode_vnf_instances': episode_vnf_instances,
            'episode_vs_instances': episode_vs_instances,
            'best_acceptance': best_acceptance,
            'best_consolidation': best_consolidation,
            'training_time': train_duration
        }

    def solve(self):
        """Solve using trained DQN"""
        solve_start_time = time.time()

        logger.info(f"\n{'=' * 80}")
        logger.info("SOLVING WITH TRAINED DQN")
        logger.info(f"{'=' * 80}")

        if self.agent is None:
            raise ValueError("Agent not trained. Call train() first.")

        self.agent.epsilon = 0.0

        env = SFCEnvironment(self.topology)
        state = env.reset()

        step_count = 0
        current_sfc_name = None

        while state is not None:
            if state['sfc_name'] != current_sfc_name:
                current_sfc_name = state['sfc_name']
                logger.info(f"\n{'=' * 60}")
                logger.info(f"Processing SFC: {current_sfc_name}")
                logger.info(f"{'=' * 60}")

            valid_actions = env.get_valid_actions()
            if valid_actions is None or valid_actions.sum() == 0:
                break

            vnf_idx = state['vnf_idx']
            current_vnf = env.current_sfc['vnfs'][vnf_idx]
            npop_names = list(env.topology['npops'].keys())

            logger.info(f"\n  VNF {vnf_idx + 1}/{len(env.current_sfc['vnfs'])}: {current_vnf['name']}")
            logger.info(f"    CPU demand: {current_vnf['cpu_demand']}, MEM demand: {current_vnf['mem_demand']}")
            logger.info(f"    Valid actions: {valid_actions.sum().item()}/{len(npop_names)}")

            with torch.no_grad():
                q_values = self.agent.policy_net(
                    state['infra_data'],
                    state['sfc_data'],
                    state['vnf_idx'],
                    state['num_npops'],
                    valid_actions
                )
                action = q_values.argmax().item()

            selected_npop = npop_names[action]
            logger.info(f"    ➜ Selected NPoP: {selected_npop}")

            cpu_avail_before = env.topology['npops'][selected_npop]['cpu_capacity'] - env.npop_cpu_used[selected_npop]
            mem_avail_before = env.topology['npops'][selected_npop]['mem_capacity'] - env.npop_mem_used[selected_npop]
            logger.info(f"    Available resources: CPU={cpu_avail_before:.1f}, MEM={mem_avail_before:.1f}")

            next_state, reward, done = env.step(action)

            cpu_avail_after = env.topology['npops'][selected_npop]['cpu_capacity'] - env.npop_cpu_used[selected_npop]
            mem_avail_after = env.topology['npops'][selected_npop]['mem_capacity'] - env.npop_mem_used[selected_npop]
            logger.info(f"    After placement: CPU={cpu_avail_after:.1f}, MEM={mem_avail_after:.1f}")
            logger.info(f"    Reward: {reward:.2f}")

            state = next_state
            step_count += 1

            if done:
                break

        results = env.get_results()

        solve_end_time = time.time()
        solve_duration = solve_end_time - solve_start_time

        logger.info(f"\n{'=' * 80}")
        logger.info("RESULTS")
        logger.info(f"{'=' * 80}")
        logger.info(f"Solving time: {format_time(solve_duration)}")
        logger.info(f"\n✅ SFCs Accepted: {results['accepted_sfcs']}/{results['total_sfcs']}")
        logger.info(f"📊 Acceptance Rate: {results['accepted_sfcs'] / results['total_sfcs'] * 100:.1f}%")
        logger.info(f"🔥 NPoPs Used: {results['npops_used']}/{len(self.topology['npops'])}")
        logger.info(f"📦 Consolidation Rate: {(1 - results['npops_used'] / len(self.topology['npops'])) * 100:.1f}%")
        logger.info(f"🔧 VNF Instances: {results['vnf_instances']}")
        logger.info(f"⚙️ VS Instances: {results['vs_instances']}")

        logger.info("\n--- SFC Status ---")
        for sfc in self.topology['sfc_requests']:
            accepted = sfc['name'] in results['deployed_sfcs']
            logger.info(f"{sfc['name']}: {'✓ ACCEPTED' if accepted else '✗ REJECTED'}")

        logger.info("\n--- VNF Assignments ---")
        for sfc in self.topology['sfc_requests']:
            if sfc['name'] in results['deployed_sfcs']:
                logger.info(f"\n{sfc['name']}:")
                for vnf in sfc['vnfs']:
                    key = (sfc['name'], vnf['name'])
                    if key in results['vnf_placements']:
                        logger.info(f"  {vnf['name']} → {results['vnf_placements'][key]}")

        logger.info("\n--- VS Assignments ---")
        for sfc in self.topology['sfc_requests']:
            if sfc['name'] in results['deployed_sfcs']:
                logger.info(f"\n{sfc['name']}:")
                for vs in sfc['virtual_switches']:
                    key = f"{sfc['name']}_{vs['name']}"
                    if key in results['vs_placements']:
                        ps_list = results['vs_placements'][key]
                        logger.info(f"  {vs['name']} → {ps_list}")

        return {
            'status': 'Optimal',
            'accepted_sfcs': results['accepted_sfcs'],
            'total_sfcs': results['total_sfcs'],
            'npops_used': results['npops_used'],
            'vnf_instances': results['vnf_instances'],
            'vs_instances': results['vs_instances'],
            'sfc_acceptance': {sfc['name']: sfc['name'] in results['deployed_sfcs']
                               for sfc in self.topology['sfc_requests']},
            'vnf_assignments': results['vnf_placements'],
            'vs_assignments': results['vs_placements'],
            'vlink_routes': results['vlink_routes'],
            'solving_time': solve_duration
        }

    def calculate_residual_capacity(self, results: Dict) -> Dict:
        """Calculate residual capacity for all resources"""
        residual = {
            'npops': {},
            'physical_switches': {},
            'physical_links': {}
        }

        env = SFCEnvironment(self.topology)
        env.reset()

        # Replicate VNF deployments
        for sfc in self.topology['sfc_requests']:
            if results['sfc_acceptance'].get(sfc['name'], False):
                for vnf in sfc['vnfs']:
                    key = (sfc['name'], vnf['name'])
                    if key in results['vnf_assignments']:
                        npop = results['vnf_assignments'][key]
                        env.resource_manager.add_vnf_deployment(
                            vnf['name'],
                            npop,
                            sfc['name'],
                            vnf['cpu_demand'],
                            vnf['mem_demand']
                        )

        # Replicate VS deployments
        for sfc in self.topology['sfc_requests']:
            if results['sfc_acceptance'].get(sfc['name'], False):
                for vs in sfc['virtual_switches']:
                    key = f"{sfc['name']}_{vs['name']}"
                    if key in results['vs_assignments']:
                        ps_list = results['vs_assignments'][key]
                        stage_demand = vs['stage_demand']
                        for ps_name in ps_list:
                            env.resource_manager.add_vs_user(
                                vs['name'], ps_name, sfc['name'], stage_demand
                            )

        # Replicate VL deployments
        for sfc in self.topology['sfc_requests']:
            if results['sfc_acceptance'].get(sfc['name'], False):
                for vlink_idx, vl in enumerate(sfc['virtual_links']):
                    route_key = (sfc['name'], vlink_idx)
                    if route_key in results['vlink_routes']:
                        _, link_path = results['vlink_routes'][route_key]
                        for link_id in link_path:
                            env.resource_manager.add_link_usage(
                                link_id, sfc['name'], vl['bandwidth_demand']
                            )

        # Calculate residuals for NPoPs
        for npop_name, npop in self.topology['npops'].items():
            cpu_used = 0.0
            mem_used = 0.0
            for (vf_type, npop_key), deployment in env.resource_manager.vnf_deployments.items():
                if npop_key == npop_name:
                    cpu_used += deployment['total_cpu']
                    mem_used += deployment['total_mem']
            residual['npops'][npop_name] = {
                'cpu_capacity': npop['cpu_capacity'],
                'cpu_used': cpu_used,
                'cpu_residual': npop['cpu_capacity'] - cpu_used,
                'cpu_utilization_percent': (cpu_used / npop['cpu_capacity'] * 100) if npop['cpu_capacity'] > 0 else 0,
                'mem_capacity': npop['mem_capacity'],
                'mem_used': mem_used,
                'mem_residual': npop['mem_capacity'] - mem_used,
                'mem_utilization_percent': (mem_used / npop['mem_capacity'] * 100) if npop['mem_capacity'] > 0 else 0
            }

        # Calculate residuals for Physical Switches
        for ps_name, ps in self.topology['physical_switches'].items():
            stages_used = 0
            for (vs_type, ps_key, stage_demand), instance in env.resource_manager.vs_instances.items():
                if ps_key == ps_name:
                    stages_used += instance['stages_used']
            residual['physical_switches'][ps_name] = {
                'stage_capacity': ps['stage_capacity'],
                'stages_used': stages_used,
                'stages_residual': ps['stage_capacity'] - stages_used,
                'utilization_percent': (stages_used / ps['stage_capacity'] * 100) if ps['stage_capacity'] > 0 else 0
            }

        # Calculate residuals for Physical Links
        for link_id, link in self.topology['physical_links'].items():
            bw_used = env.resource_manager.get_link_usage(link_id)
            residual['physical_links'][link_id] = {
                'bandwidth_capacity': link['bandwidth'],
                'bandwidth_used': bw_used,
                'bandwidth_residual': link['bandwidth'] - bw_used,
                'utilization_percent': (bw_used / link['bandwidth'] * 100) if link['bandwidth'] > 0 else 0
            }

        return residual

    @staticmethod
    def display_residual_capacity(residual: Dict):
        """Display residual capacity information"""
        logger.info(f"\n{'=' * 80}")
        logger.info("RESIDUAL CAPACITY")
        logger.info(f"{'=' * 80}")

        logger.info("\n--- Network PoPs ---")
        for npop_name, data in residual['npops'].items():
            logger.info(f"\n{npop_name}:")
            logger.info(f"  CPU: {data['cpu_used']:.2f}/{data['cpu_capacity']:.2f} " +
                        f"(Residual: {data['cpu_residual']:.2f}, {data['cpu_utilization_percent']:.1f}% used)")
            logger.info(f"  MEM: {data['mem_used']:.2f}/{data['mem_capacity']:.2f} " +
                        f"(Residual: {data['mem_residual']:.2f}, {data['mem_utilization_percent']:.1f}% used)")

        logger.info("\n--- Physical Switches ---")
        for ps_name, data in residual['physical_switches'].items():
            logger.info(f"{ps_name}: {data['stages_used']}/{data['stage_capacity']} stages " +
                        f"(Residual: {data['stages_residual']}, {data['utilization_percent']:.1f}% used)")

        logger.info("\n--- Physical Links ---")
        for link_id, data in residual['physical_links'].items():
            logger.info(f"Link {link_id}: " +
                        f"{data['bandwidth_used']:.2f}/{data['bandwidth_capacity']:.2f} BW " +
                        f"(Residual: {data['bandwidth_residual']:.2f}, {data['utilization_percent']:.1f}% used)")

    def export_topology_to_json(self, results: Dict, residual: Dict, output_file: str = "dqn_results.json"):
        """Export complete topology information and results to JSON file"""
        import json
        from datetime import datetime

        # Build topology data
        topology_data = {
            "metadata": {
                "solver": "SFC DQN Solver",
                "status": results.get('status', 'Optimal'),
                "accepted_sfcs": results.get('accepted_sfcs'),
                "total_sfcs": results.get('total_sfcs'),
                "npops_used": results.get('npops_used'),
                "vnf_instances": results.get('vnf_instances'),
                "vs_instances": results.get('vs_instances'),
                "solving_time": results.get('solving_time'),
                "timestamp": datetime.now().isoformat(),
                "hyperparameters": {
                    "episodes": self.args.episodes,
                    "batch_size": self.args.batch_size,
                    "learning_rate": self.args.learning_rate,
                    "gamma": self.args.gamma,
                    "epsilon_start": self.args.epsilon_start,
                    "epsilon_end": self.args.epsilon_end,
                    "epsilon_decay": self.args.epsilon_decay,
                    "memory_size": self.args.memory_size,
                    "target_update": self.args.target_update,
                    "train_freq": self.args.train_freq,
                    "device": str(DEVICE)
                }
            },
            "physical_topology": {
                "endpoints": self.topology['endpoints'],
                "physical_switches": [
                    {
                        "name": ps_data['name'],
                        "stage_capacity": ps_data['stage_capacity']
                    }
                    for ps_name, ps_data in self.topology['physical_switches'].items()
                ],
                "network_pops": [
                    {
                        "name": npop_data['name'],
                        "cpu_capacity": npop_data['cpu_capacity'],
                        "mem_capacity": npop_data['mem_capacity']
                    }
                    for npop_name, npop_data in self.topology['npops'].items()
                ],
                "physical_links": [
                    {
                        "id": link_data['id'],
                        "source": link_data['source'],
                        "target": link_data['target'],
                        "bandwidth": link_data['bandwidth']
                    }
                    for link_id, link_data in self.topology['physical_links'].items()
                ]
            },
            "sfc_requests": [
                {
                    "name": sfc['name'],
                    "endpoints": sfc['endpoints'],
                    "vnfs": [
                        {
                            "name": vf['name'],
                            "cpu_demand": vf['cpu_demand'],
                            "mem_demand": vf['mem_demand'],
                            "assigned_npop": vf.get('assigned_npop')
                        }
                        for vf in sfc['vnfs']
                    ],
                    "virtual_switches": [
                        {
                            "name": vs['name'],
                            "stage_demand": vs['stage_demand']
                        }
                        for vs in sfc['virtual_switches']
                    ],
                    "virtual_links": [
                        {
                            "source": vl['source'],
                            "target": vl['target'],
                            "bandwidth_demand": vl['bandwidth_demand'],
                            "vs_type": vl.get('vs_type')
                        }
                        for vl in sfc['virtual_links']
                    ]
                }
                for sfc in self.topology['sfc_requests']
            ],
            "solution": {
                "accepted_sfcs": results.get('accepted_sfcs', 0),
                "total_sfcs": results.get('total_sfcs', 0),
                "sfc_acceptance": results.get('sfc_acceptance', {}),
                "vnf_assignments": {
                    f"{k[0]}_{k[1]}": v
                    for k, v in results.get('vnf_assignments', {}).items()
                },
                "vs_assignments": results.get('vs_assignments', {}),
                "vlink_routes": {
                    f"{k[0]}_vl{k[1]}": {
                        "node_path": v[0],
                        "link_path": v[1]
                    }
                    for k, v in results.get('vlink_routes', {}).items()
                }
            },
            "residual_capacity": residual
        }

        # Save to JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(topology_data, f, indent=2, ensure_ascii=False)

        logger.info(f"\n✓ Topology data exported to: {output_file}")

        return topology_data


def main():
    """Main function with argparser"""

    arg_parser = create_arg_parser()
    print_all_settings(arg_parser)

    # Set random seed if specified
    if arg_parser.seed is not None:
        set_random_seed(arg_parser.seed)

    # Configure device
    global DEVICE
    if arg_parser.device == 'cpu':
        DEVICE = torch.device('cpu')
    elif arg_parser.device == 'cuda':
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Using CPU instead.")
            DEVICE = torch.device('cpu')
        else:
            DEVICE = torch.device('cuda')
    else:  # auto
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info(f"Using device: {DEVICE}")

    filename = arg_parser.input
    output_json = arg_parser.output

    logger.info(f"{'=' * 80}")
    logger.info("SFC DQN SOLVER")
    logger.info(f"{'=' * 80}")
    logger.info(f"\nInput file: {filename}")
    logger.info(f"Output file: {output_json}")
    logger.info(f"Training episodes: {arg_parser.episodes}")

    try:
        # Load topology
        solver = SFCDQNSolver(arg_parser)
        solver.load_topology_from_file(filename)

        # Train
        start_time = time.time()
        training_stats = solver.train()
        training_time = time.time() - start_time

        # Solve
        results = solver.solve()

        # Calculate residual capacity
        residual = solver.calculate_residual_capacity(results)

        # Display residual capacity
        solver.display_residual_capacity(residual)

        # Export to JSON with residual capacity
        topology_data = solver.export_topology_to_json(results, residual, output_json)

        # Summary
        logger.info(f"\n{'=' * 80}")
        logger.info("SUMMARY")
        logger.info(f"{'=' * 80}")
        logger.info(f"\n✓ Device: {DEVICE}")
        logger.info(f"✓ Training time: {format_time(training_time)}")
        logger.info(f"✓ Episodes: {arg_parser.episodes}")
        logger.info(f"✓ Best acceptance: {training_stats['best_acceptance'] * 100:.1f}%")
        logger.info(f"✓ Best consolidation: {training_stats['best_consolidation']:.1f} NPoPs")
        logger.info(f"✓ Final: {results['accepted_sfcs']}/{results['total_sfcs']} SFCs")
        logger.info(f"✓ NPoPs Used: {results['npops_used']}")
        logger.info(f"✓ VNF Instances: {results['vnf_instances']}")
        logger.info(f"✓ VS Instances: {results['vs_instances']}")
        logger.info(f"✓ Results saved to: {output_json}")

        logger.info(f"\n{'=' * 80}")
        logger.info("✅ EXECUTION COMPLETED")
        logger.info(f"{'=' * 80}")

        return results

    except FileNotFoundError:
        logger.error(f"\n✗ ERROR: File '{filename}' not found")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n✗ ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
