# Standard library imports
import numpy as np
import os
import time
import logging # For structured logging
from typing import Tuple, Dict, Any, List # For type hinting, Tuple is crucial for Python < 3.9

# PyTorch related imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from battlesnake_gym.snake_gym import BattlesnakeGym

# Environment import (Gymnasium is the successor to Gym)
# import gymnasium as gym # Consider updating if BattlesnakeGym supports/requires Gymnasium
import gym # Sticking to gym as per original for now
import random
# import torch.nn.functional as F # Not explicitly used, can be removed if not needed elsewhere
# Optional: Weights & Biases for experiment tracking
# import wandb

# --- Basic Logging Configuration ---
# Configures the logging module to output messages with a timestamp, level, and message.
# This helps in debugging and tracking the training process.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

team_ids = [1, 0, 3, 2] # Team IDs for the two teams in the game

def get_agent_observed_state(agent_id: int, observation: np.ndarray, team_id: int) -> np.ndarray:
    """
    Transforms the raw environment observation for a specific agent.
    Input observation is typically (H, W, C_raw), where C_raw includes food and all snakes.
    Output is a (4, H, W) grid: [agent_self, agent_teammate, enemies, food].

    Args:
        agent_id: The ID of the agent for whom the state is being processed (0-3).
        observation: The raw observation from the environment.
        team_id: The ID of the teammate for the current agent.
                 (Note: In a 2v2, agent_id 0's teammate is 1, and agent_id 1's teammate is 0.
                  Similarly for agents 2 and 3. The provided team_ids array handles this mapping.)

    Returns:
        A NumPy array of shape (4, H, W) representing the agent's processed view.
    """
    food_spaces = observation[:, :, 0]
    agent_positions = observation[:, :, agent_id + 1] # +1 because channel 0 is food

    # Ensure team_id is valid and corresponds to an actual snake channel
    # The team_ids array maps agent_id to its teammate's agent_id.
    # So, observation[:, :, team_ids[agent_id] + 1] gives the teammate's layer.
    teammate_actual_id = team_ids[agent_id]
    team_agent_positions = observation[:, :, teammate_actual_id + 1]

    # Summing all snake layers (excluding food layer)
    all_snake_layers = observation[:, :, 1:]
    # Subtract self and teammate to get enemy positions
    # Ensure broadcasting or direct indexing is correct if layers are binary
    enemy_positions = np.sum(all_snake_layers, axis=2) - agent_positions - team_agent_positions
    # Clamp negative values that might arise if agent/teammate positions overlap or sum is imperfect
    enemy_positions = np.clip(enemy_positions, 0, None)


    stacked = np.stack([agent_positions, team_agent_positions, enemy_positions, food_spaces], axis=0)
    return np.array(stacked, dtype=np.float32)


class RolloutBuffer:
    """
    Stores trajectories (sequences of states, actions, rewards, etc.) collected during environment interaction.
    This data is used by the PPO agent to update its policy.
    """
    def __init__(self):
        """Initializes empty lists to store trajectory data."""
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.logprobs: List[float] = []
        self.rewards: List[float] = []
        self.state_values: List[float] = []
        self.is_terminals: List[bool] = []

    def store_transition(self, state: np.ndarray, action: int, logprob: float, reward: float, state_value: float, is_terminal: bool):
        """Appends a single transition to the buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.state_values.append(state_value)
        self.is_terminals.append(is_terminal)

    def clear(self):
        """Clears all stored transitions from the buffer."""
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

    def __len__(self) -> int:
        """Returns the current number of transitions stored in the buffer."""
        return len(self.states)

def initial_random_policy(observation: np.ndarray, action_space: gym.spaces.Discrete) -> int:
    """
    A simple heuristic/random policy.
    Args:
        observation: Processed observation for the agent (e.g., (4, H, W) tensor).
                     Assumes observation[0] is agent_positions, observation[1] is team_agent_positions.
        action_space: The agent's action space.
    Returns:
        A randomly chosen valid action.
    """
    # observation is expected to be the (4, H, W) processed state
    agent_layer = observation[0] # Agent's own position
    # Assuming '5' is not a standard value in the processed observation.
    # The agent's body parts are usually represented by 1s on its layer.
    # Head position finding might need to be environment-specific if not just 'any part of agent'.
    # For simplicity, we'll find any '1' on the agent layer as a reference.
    # This part may need refinement based on how head vs body is represented.
    agent_parts_indices = np.argwhere(agent_layer > 0)
    if agent_parts_indices.shape[0] == 0:
        return action_space.sample() # No agent found, random action

    # Using the first found part as a proxy for head (needs better head tracking if available)
    # For Battlesnake, the head is usually distinct or identifiable.
    # This simple policy focuses on immediate non-colliding moves.
    # A true head position would be (y,x) from the environment observation if available.
    # Assuming observation is (C, H, W), agent_parts_indices are (idx, row, col)
    # Let's assume the observation passed here is agent_state for one agent.
    # For the initial policy, let's simplify and make it move randomly if it can,
    # or take a valid move. The original logic seems to rely on a specific value '5'.
    # If the input 'observation' here is the agent's layer directly (11x11):
    agent_head_coords = np.unravel_index(np.argmax(agent_layer), agent_layer.shape) # Simplistic head finding

    # This policy is very basic. The original had specific logic for value '5'.
    # Sticking to a more general valid move selection:
    # For a proper heuristic, one would need to know what values represent walls/hazards/own body.
    # Assuming 0 is empty, 1 is obstacle/body.
    # The example from the original code:
    # enemy_states = observation[1] # This assumes observation[1] is enemy state.
    # In our get_agent_observed_state:
    # observation[0] = self, observation[1]=teammate, observation[2]=enemies
    
    # Replicating original intent if '5' was head on agent_layer (now observation[0])
    if np.sum(observation[0] == 5) > 0: # Check if '5' (head marker) exists
        agent_head_index = (np.where(observation[0] == 5)[0][0], np.where(observation[0] == 5)[1][0])
    else: # Fallback if no '5' found, maybe just take any part of the snake
        agent_body_parts = np.argwhere(observation[0] > 0)
        if len(agent_body_parts) > 0:
            agent_head_index = tuple(agent_body_parts[0]) # Take first body part
        else:
            return action_space.sample() # No snake body found

    enemy_and_team_positions = observation[1] + observation[2] # Teammate + Enemies

    new_head_indices = {
        (agent_head_index[0] - 1, agent_head_index[1]): 0,  # Up
        (agent_head_index[0] + 1, agent_head_index[1]): 1,  # Down
        (agent_head_index[0], agent_head_index[1] - 1): 2,  # Left
        (agent_head_index[0], agent_head_index[1] + 1): 3   # Right
    }

    valid_moves = []
    for (r, c), move_idx in new_head_indices.items():
        if 0 <= r < observation.shape[1] and 0 <= c < observation.shape[2]: # Check bounds (H, W)
            # Check if target cell is not part of own body (on agent_layer, assuming head moves)
            # and not part of enemy/teammate body
            # Assuming 0 is empty in all these layers. >0 means occupied.
            if observation[0][r, c] == 0 and enemy_and_team_positions[r,c] == 0:
                valid_moves.append(move_idx)
    
    if valid_moves:
        return np.random.choice(valid_moves)
    else:
        # Fallback to any action if no "safe" move is found (might lead to collision)
        # A better heuristic would try to survive longer, e.g., by not hitting itself immediately.
        # The original code did action_space.sample(), which could be any of the 4 actions.
        # This is a tricky part for a "simple" policy.
        # For now, let's stick to the original fallback.
        return action_space.sample()
    
def random_policy(observation: np.ndarray, action_space: gym.spaces.Discrete) -> int:
    return action_space.sample()


class PolicyValueNetwork(nn.Module):
    """
    Actor-Critic Network for PPO.
    Input: (N, 4, H, W) tensor (self, teammate, enemies, food)
    Outputs action probabilities and state value.
    """
    def __init__(self, n_actions: int, hidden_dim: int = 64, device: str = 'cpu'):
        super(PolicyValueNetwork, self).__init__()
        self.device = device

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1),  # e.g. (4,11,11) -> (16,11,11)
            nn.Tanh(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # (16,11,11) -> (32,11,11)
            nn.Tanh(),
            nn.Flatten(start_dim=1),  # (32,H,W) -> (32*H*W)
            # Adjust Linear input size based on H, W (e.g., 11x11)
            nn.Linear(32 * 11 * 11, hidden_dim), # Assuming 11x11 map
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        ).to(device)

        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, n_actions),
            nn.Softmax(dim=-1)
        ).to(device)

        self.critic_head = nn.Linear(hidden_dim, 1).to(device)

    def forward(self, x: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass.
        Args:
            x: Input state observation(s). Can be NumPy array or Tensor.
               Shape (C,H,W) for single, or (N,C,H,W) for batch.
        """
        if not isinstance(x, torch.Tensor):
            # Ensure x is a NumPy array before converting if it's not already tensor
            x_np = np.asarray(x, dtype=np.float32)
            x = torch.tensor(x_np, dtype=torch.float32, device=self.device)
        elif x.device != self.device:
            x = x.to(self.device)

        # Handle single observation (C,H,W) by adding batch dim -> (1,C,H,W)
        if x.ndim == 3:  # (C, H, W)
            x = x.unsqueeze(0)
        # Note: original code had x.ndim == 1 for flat obs, not applicable for this CNN.

        features = self.feature_extractor(x)
        action_probs = self.actor_head(features)
        state_value = self.critic_head(features)
        return action_probs, state_value

    @torch.no_grad()
    def select_action(self, obs: np.ndarray) -> Tuple[int, float, float]:
        """Selects an action for a given observation during interaction."""
        # Forward pass handles tensor conversion, device placement, and batching if needed
        action_probs, state_value_tensor = self.forward(obs)

        dist = Categorical(probs=action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.item(), action_logprob.item(), state_value_tensor.item()

    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluates actions for PPO update."""
        # states are expected to be batched tensors (N,C,H,W)
        action_probs, state_values_tensor = self.forward(states)

        dist = Categorical(action_probs)
        # Ensure actions tensor is correctly shaped for log_prob
        action_logprobs = dist.log_prob(actions.squeeze(-1) if actions.ndim > 1 else actions)
        dist_entropy = dist.entropy()

        return state_values_tensor.squeeze(-1), action_logprobs, dist_entropy


class PPOAgent:
    def __init__(
            self,
            action_dim: int, # Number of possible actions
            lr: float = 3e-4,
            num_epochs: int = 10,
            eps_clip: float = 0.2,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            entropy_coef: float = 0.01,
            value_loss_coef: float = 0.5,
            hidden_dim: int = 64, # Hidden dim for PolicyValueNetwork
            minibatch_size: int = 64,
            device: str = 'cpu',
            load_model_path: str = None,
            max_grad_norm: float = 0.5
        ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.num_epochs = num_epochs
        self.minibatch_size = minibatch_size
        self.eps_clip = eps_clip
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.device = device
        self.max_grad_norm = max_grad_norm

        # Store for potential use elsewhere (e.g. creating fixed opponent policies)
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.policy = PolicyValueNetwork(
            n_actions=action_dim,
            hidden_dim=hidden_dim,
            device=device
        )
        self.optimizer = optim.AdamW(self.policy.parameters(), lr=lr, eps=1e-5)
        self.buffer = RolloutBuffer()
        self.mse_loss = nn.MSELoss()
        self.total_steps_trained = 0

        self.load_model_path = load_model_path
        if load_model_path:
            self.load_model(load_model_path)

    def load_model(self, path: str):
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.total_steps_trained = checkpoint.get('total_steps_trained', 0)
            self.policy.eval()
            logging.info(f"Loaded model from {path}, resuming from {self.total_steps_trained} steps.")
        except FileNotFoundError:
            logging.warning(f"Model file not found at {path}. Starting from scratch.")
        except Exception as e:
            logging.error(f"Error loading model from {path}: {e}. Starting from scratch.")
            self.total_steps_trained = 0

    def save_model(self, path: str):
        try:
            # Ensure directory exists
            save_dir = os.path.dirname(path)
            if save_dir and not os.path.exists(save_dir): # Check if save_dir is not empty string
                 os.makedirs(save_dir, exist_ok=True)

            torch.save({
                'policy_state_dict': self.policy.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'total_steps_trained': self.total_steps_trained
            }, path)
            logging.info(f"Saved model to {path} at {self.total_steps_trained} steps.")
        except Exception as e:
            logging.error(f"Error saving model to {path}: {e}")

    def _compute_gae_and_returns(self, next_value_bootstrap: float, next_is_terminal_bootstrap: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        num_steps = len(self.buffer.rewards)
        advantages = torch.zeros(num_steps, device=self.device)
        
        rewards = torch.tensor(self.buffer.rewards, dtype=torch.float32, device=self.device)
        state_values = torch.tensor(self.buffer.state_values, dtype=torch.float32, device=self.device)
        is_terminals_tensor = torch.tensor(self.buffer.is_terminals, dtype=torch.float32, device=self.device)

        last_gae_lam = 0.0
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_non_terminal = 1.0 - float(next_is_terminal_bootstrap)
                next_val = next_value_bootstrap
            else:
                next_non_terminal = 1.0 - is_terminals_tensor[t + 1]
                next_val = state_values[t + 1]
            
            delta = rewards[t] + self.gamma * next_val * next_non_terminal - state_values[t]
            advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
        
        returns_to_go = advantages + state_values
        return advantages, returns_to_go

    def KL_divergence(self, p_probs: torch.Tensor, q_probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Computes KL divergence D_KL(p || q) for batches of probability distributions."""
        # Add epsilon for numerical stability
        log_p = torch.log(p_probs + eps)
        log_q = torch.log(q_probs + eps)
        # Sum over the action dimension (last dimension)
        kl_div = torch.sum(p_probs * (log_p - log_q), dim=-1)
        return kl_div.mean() # Mean over the batch

    def update_policy(self, next_obs_for_bootstrap: np.ndarray, next_done_for_bootstrap: bool, second_agent: 'PPOAgent' = None) -> Tuple[float, float, float]:
        # Estimate value of the state after the rollout buffer ends (for GAE)
        with torch.no_grad():
            # self.policy.forward handles tensor conversion, device placement, and batching
            _, next_value_tensor = self.policy.forward(next_obs_for_bootstrap) # obs shape (C,H,W)
            next_value_bootstrap = next_value_tensor.item()

        advantages, rewards_to_go = self._compute_gae_and_returns(next_value_bootstrap, next_done_for_bootstrap)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Prepare data from buffer
        # States are expected to be (C,H,W) from get_agent_observed_state
        old_states_np = np.array(self.buffer.states, dtype=np.float32) # Should be (num_samples, C, H, W)
        old_states = torch.tensor(old_states_np, dtype=torch.float32, device=self.device)
        
        old_actions = torch.tensor(self.buffer.actions, dtype=torch.long, device=self.device) # No need to unsqueeze if evaluate_actions handles it
        old_logprobs = torch.tensor(self.buffer.logprobs, dtype=torch.float32, device=self.device)

        num_samples_in_buffer = len(self.buffer)
        total_actor_loss, total_critic_loss, total_entropy, total_kl_div = 0.0, 0.0, 0.0, 0.0

        self.policy.train()
        for _ in range(self.num_epochs):
            indices = torch.randperm(num_samples_in_buffer, device=self.device)
            for start_idx in range(0, num_samples_in_buffer, self.minibatch_size):
                end_idx = min(start_idx + self.minibatch_size, num_samples_in_buffer)
                batch_indices = indices[start_idx:end_idx]

                batch_states = old_states[batch_indices]
                batch_actions = old_actions[batch_indices]
                batch_old_logprobs = old_logprobs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_rewards_to_go = rewards_to_go[batch_indices]
                
                current_state_values, current_logprobs, dist_entropy = self.policy.evaluate_actions(batch_states, batch_actions)

                ratios = torch.exp(current_logprobs - batch_old_logprobs) # batch_old_logprobs is 1D

                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                critic_loss = self.mse_loss(current_state_values, batch_rewards_to_go)
                
                kl_div_value = torch.tensor(0.0, device=self.device)
                lam_kl = 0.02

                if second_agent is not None and hasattr(second_agent, 'policy'):
                    with torch.no_grad(): # Do not train the second agent
                        current_policy_action_probs, _ = self.policy(batch_states)
                        second_agent_action_probs, _ = second_agent.policy(batch_states)
                    
                    kl_div_value = self.KL_divergence(current_policy_action_probs, second_agent_action_probs)
                    lam_kl = 0.005 # Hyperparameter for KL divergence penalty/bonus
                    total_kl_div += kl_div_value.item()
                
                loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * dist_entropy.mean() + lam_kl * kl_div_value
                
                self.optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += dist_entropy.mean().item()
        
        self.policy.eval()
        self.buffer.clear()
        
        num_update_steps = self.num_epochs * (num_samples_in_buffer // self.minibatch_size + (1 if num_samples_in_buffer % self.minibatch_size != 0 else 0))
        avg_actor_loss = total_actor_loss / num_update_steps
        avg_critic_loss = total_critic_loss / num_update_steps
        avg_entropy = total_entropy / num_update_steps
        # avg_kl_div = total_kl_div / num_update_steps if second_agent is not None else 0.0 # Optional: log KL

        return avg_actor_loss, avg_critic_loss, avg_entropy


class PPOTrainer:
    def __init__(self,
                 env: BattlesnakeGym,
                 agents: List[PPOAgent], # Expecting a list of PPOAgent instances
                 rollout_steps: int = 2048,
                 log_interval_episodes: int = 10,
                 save_interval_steps: int = 50000,
                 ckpt_dir: str = "models/PPO",
                 render_mode: bool = False, # Changed from str to bool
                 seed: int = 42,
                 max_ep_len: int = 500,
                 render_freq: float = None, # Should be float for time.sleep
                 dummy_agent: PPOAgent = None): # Used for config like action_dim

        self.env = env
        # self.env.reset(seed=seed) # Seed the environment if its reset method supports it
        self.agents = agents # List of PPOAgent instances
        self.num_learning_agents = len(agents) # Typically 2 for this setup
        self.rollout_steps = rollout_steps
        self.log_interval_episodes = log_interval_episodes
        self.save_interval_steps = save_interval_steps
        self.ckpt_dir = ckpt_dir
        self.max_ep_len = max_ep_len
        self.seed = seed # Store seed, set it globally if needed (torch, numpy, random)
        self.render = render_mode
        self.render_freq = render_freq
        self.dummy_agent = dummy_agent # Should be a PPOAgent instance to get network params

        self.existing_policies: List[Any] = [random_policy, initial_random_policy] # Can hold PolicyValueNetwork or functions
        if self.ckpt_dir:
            os.makedirs(self.ckpt_dir, exist_ok=True)
            self._update_existing_policies()


        self.episode_rewards_buffer: List[float] = []
        self.episode_lengths_buffer: List[int] = []
        self.total_episodes_completed = 0

        # Initialize last_save_step_counts based on the number of agents
        self.last_save_step_counts = [(agent.total_steps_trained // save_interval_steps) * save_interval_steps for agent in self.agents]


        # Set seeds for reproducibility
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.agents[0].device == "cuda": # Assuming all agents on same device
            torch.cuda.manual_seed_all(self.seed)


    def _update_existing_policies(self):
        """Loads existing policies (PolicyValueNetwork instances) from the checkpoint directory."""
        self.existing_policies = [initial_random_policy, random_policy] # Start with the random policy
        if not self.dummy_agent:
            logging.warning("No dummy_agent provided to PPOTrainer, cannot load existing policies.")
            return
        if not os.path.exists(self.ckpt_dir):
            logging.warning(f"Checkpoint directory {self.ckpt_dir} not found. No policies loaded.")
            return

        try:
            files_in_load_dir = sorted(os.listdir(self.ckpt_dir), key=lambda x: int(x.split('_')[-1][:-4]))
            print(f"Files in checkpoint directory: {files_in_load_dir}")
        except FileNotFoundError:
            logging.warning(f"Checkpoint directory {self.ckpt_dir} does not exist. Cannot load policies.")
            return
            
        for file_name in files_in_load_dir:
            if file_name.endswith(".pth"):
                file_path = os.path.join(self.ckpt_dir, file_name)
                try:
                    loaded_policy_net = PolicyValueNetwork(
                        n_actions=self.dummy_agent.action_dim,
                        hidden_dim=self.dummy_agent.hidden_dim,
                        device=self.dummy_agent.device
                    )
                    checkpoint = torch.load(file_path, map_location=self.dummy_agent.device)
                    if 'policy_state_dict' in checkpoint:
                        loaded_policy_net.load_state_dict(checkpoint['policy_state_dict'])
                        loaded_policy_net.eval()
                        self.existing_policies.append(loaded_policy_net)
                        logging.info(f"Loaded existing policy from {file_path}")
                    else:
                        logging.warning(f"Checkpoint {file_path} does not contain 'policy_state_dict'. Skipping.")
                except Exception as e:
                    logging.error(f"Error loading policy from {file_path}: {e}")
        logging.info(f"Current existing policies count: {len(self.existing_policies)}")

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))  # for numerical stability
        return e_x / e_x.sum()

    def _set_policies_for_agents(self) -> Tuple[List[Any], List[int]]:
        """
        Assigns policies for all 4 snakes in the game.
        The first two are the learning agents.
        The next two are opponents chosen from existing_policies.
        """
        # Policies for the learning agents
        learning_agent_policies = [agent.policy for agent in self.agents[:self.num_learning_agents]]
        
        # Indices to track original policy type (0 for learning, >0 for existing_policies index)
        policy_source_indices = list(range(self.num_learning_agents)) # Placeholder, actual meaning differs

        opponent_policies = []
        opponent_source_indices = []

        num_opponents = 4 - self.num_learning_agents
        if len(self.existing_policies) > 0:
            for _ in range(num_opponents):
                # Choose randomly from all existing policies (including initial_random_policy at index 0)
                # chosen_idx = np.random.randint(0, len(self.existing_policies))
                
                chosen_idx = np.random.choice(
                    len(self.existing_policies), 
                    p=self.softmax(np.array([(i // 2 + 1) for i in range(len(self.existing_policies))]))
                )
                opponent_policies.append(self.existing_policies[chosen_idx])
                opponent_source_indices.append(chosen_idx) # Store the index from existing_policies
        else: # Fallback if no policies loaded (should not happen if initial_random_policy is always there)
            for _ in range(num_opponents):
                opponent_policies.append(initial_random_policy)
                opponent_policies.append(random_policy)
                opponent_source_indices.append(0) # Assuming initial_random_policy is effectively index 0

        # The game expects 4 policies.
        # Example: 2 learning agents, 2 opponents
        # combined_policies needs to be ordered by snake ID (0, 1, 2, 3)
        # This part depends on how agents are mapped to snake IDs in the environment.
        # Assuming self.agents[0] is snake 0, self.agents[1] is snake 1.
        # And opponents are snakes 2 and 3.
        
        # Simpler: always have learning agents as first N, then pick opponents.
        # The mapping to specific snake IDs (0..3) needs to be clear.
        # The original code had a shuffle for team_ids for learning agents.
        # Let's assume for now:
        # Snake 0: self.agents[0].policy
        # Snake 1: self.agents[1].policy
        # Snake 2: opponent_policies[0]
        # Snake 3: opponent_policies[1]
        
        # The original code had `policy_indices = [0, 1]; random.shuffle(policy_indices)`
        # This implied agent 0 and 1 could be mapped to different snake slots controlled by learning policies.
        # For simplicity, let's map directly for now.
        # If team_ids [1,0,3,2] means snake 0's teammate is snake 1, snake 1's is 0 etc.
        # Agent 0 controls snake 0, Agent 1 controls snake 1.
        
        all_game_policies = []
        all_game_policy_source_indices = [] # To know if it's a network or initial_random_policy

        # Learning agents (e.g., snakes 0 and 1)
        for i in range(self.num_learning_agents):
            all_game_policies.append(self.agents[i].policy)
            all_game_policy_source_indices.append(-1) # Special marker for learning agent's policy

        # Opponent agents (e.g., snakes 2 and 3)
        for i in range(num_opponents):
            all_game_policies.append(opponent_policies[i])
            all_game_policy_source_indices.append(opponent_source_indices[i])


        # This returns a list of 4 policy objects and corresponding source indices.
        # The order should match the snake IDs in the environment (0, 1, 2, 3).
        return all_game_policies, all_game_policy_source_indices


    def train(self, total_training_timesteps: int):
        logging.info(f"Starting training for {total_training_timesteps} timesteps.")
        start_time = time.monotonic()

        # Initial reset of the environment
        # Pass seed here if env supports it, or rely on global random.seed
        obs_dict_all_snakes, _, _, _ = self.env.reset() # Assuming reset gives obs for all snakes

        current_episode_rewards = [0.0] * self.num_learning_agents # Track rewards for each learning agent
        current_episode_length = 0

        # Global step counter: use the one from the first agent, assuming they train somewhat lockstep
        # or maintain a separate global step counter if agents can have varying total_steps_trained.
        # For simplicity, use agent 0's steps as the reference for the loop.
        # Start from the current number of steps agent 0 has been trained for.
        
        # Start from the maximum number of steps any agent has been trained for, to ensure all agents reach `total_training_timesteps`
        # This assumes total_training_timesteps is an additional amount of training.
        max_initial_steps = 0
        if self.agents:
            max_initial_steps = max(agent.total_steps_trained for agent in self.agents)

        # The loop should run for `total_training_timesteps` additional steps
        # The t_step_global should reflect the actual number of environment interactions.
        # Each agent's total_steps_trained will be updated internally by the PPOAgent.
        # This loop is for global environment steps.
        win_count = 0
        termination = [False, False]
        for t_step_global in range(1, total_training_timesteps + 1):
            # Update policies for the game (can change per episode or less frequently)
            # For now, let's set them once per episode start (handled in the 'if is_episode_done' block)
            if current_episode_length == 0: # Start of a new episode
                game_policies, game_policy_source_indices = self._set_policies_for_agents()

            if self.render:
                self.env.render(mode='human') # Original was 'human'
                if self.render_freq is not None and self.render_freq > 0:
                    time.sleep(self.render_freq)

            # Get processed observations for all 4 snakes
            # obs_dict_all_snakes is the raw multi-snake observation from env.step or env.reset
            agents_processed_obs = [get_agent_observed_state(snake_idx, obs_dict_all_snakes, team_ids[snake_idx]) for snake_idx in range(4)]

            actions_for_env = [0] * 4 # List to hold actions for all 4 snakes
            
            # Temporary storage for learning agents' transition data
            current_actions_learning = [None] * self.num_learning_agents
            current_logprobs_learning = [None] * self.num_learning_agents
            current_statevals_learning = [None] * self.num_learning_agents

            # Collect actions from all policies
            for snake_idx in range(4):
                policy_object = game_policies[snake_idx]
                processed_obs_for_snake = agents_processed_obs[snake_idx] # Shape (4,H,W)
                
                if isinstance(policy_object, PolicyValueNetwork): # Learning agent or loaded network
                    # select_action expects a single obs (C,H,W)
                    action, log_prob, state_val = policy_object.select_action(processed_obs_for_snake)
                    actions_for_env[snake_idx] = action
                    if snake_idx < self.num_learning_agents: # If this is a learning agent
                        current_actions_learning[snake_idx] = action
                        current_logprobs_learning[snake_idx] = log_prob
                        current_statevals_learning[snake_idx] = state_val
                elif callable(policy_object): # e.g. initial_random_policy
                    # initial_random_policy expects processed_obs and action_space
                    actions_for_env[snake_idx] = policy_object(processed_obs_for_snake, self.env.action_space[snake_idx])
                else:
                    logging.error(f"Unknown policy type for snake {snake_idx}: {type(policy_object)}")
                    actions_for_env[snake_idx] = self.env.action_space[snake_idx].sample() # Fallback


            next_obs_dict_all_snakes, rewards_all_snakes, terminated_all_snakes, info_all_snakes = self.env.step(actions_for_env)
            # done_by_env = (terminated_all_snakes[i] for i in range(self.num_learning_agents))
            terminated_all_snakes_list = list(terminated_all_snakes.values())
            done_by_env = (terminated_all_snakes_list[2] and terminated_all_snakes_list[3]) and (terminated_all_snakes_list[0] or terminated_all_snakes_list[1]) # snakes 2 and 3 are out, and one of the learning agents is out
            done_by_opponents = np.sum(terminated_all_snakes_list) >= 3 # If 3 or more snakes are out, the game is done

            # Accumulate rewards and length
            for i in range(self.num_learning_agents):
                current_episode_rewards[i] += rewards_all_snakes[i]
            current_episode_length += 1

            # Store transitions for learning agents
            for i in range(self.num_learning_agents):
                # if done_by_env:
                    # If the learning agents are done, we penalize them for being done
                #     rewards_all_snakes[i] -= -1.0
                
                    # If opponents are done, we reward the learning agents if they aren't done/rewarded
                    # print(f"Rewards for all snakes: {rewards_all_snakes}")
                    # if not terminated_all_snakes[i]:
                    #     rewards_all_snakes[i] += 1.0
                # print("Dones and Rewards: ", terminated_all_snakes, rewards_all_snakes)
                # print("Episode Termination: ", done_by_env, done_by_opponents)
                if termination[i]:
                    continue  # Already handled the terminal state, don't store further

                if terminated_all_snakes[i]:
                    termination[i] = True

                self.agents[i].buffer.store_transition(
                    agents_processed_obs[i], # The obs for agent i
                    current_actions_learning[i],
                    current_logprobs_learning[i],
                    rewards_all_snakes[i],    # Reward for agent i
                    current_statevals_learning[i],
                    terminated_all_snakes[i]  # Termination status for agent i
                )
            
            obs_dict_all_snakes = next_obs_dict_all_snakes

            # Check for episode termination
            # An episode is done if all learning agents are terminated OR max_ep_len is reached.
            # Or, if the environment signals a global 'done' if available.
            # The original code checked `terminated[0] and terminated[1]` for 2 agents.
            # Let's make it more general: if any learning agent is done, or max length.
            # Battlesnake typically ends when snakes die or a certain condition is met.
            # The `terminated_all_snakes` is a list/dict. Let's assume it's a list for now.
            
            # Episode ends if any of the learning agents is 'done' or max length reached
            # Or if the game itself declares an end (e.g. only one team left)
            # For simplicity with multi-agent: episode is done if *all* learning agents are done,
            # or if max_ep_len is reached. This might need adjustment based on game rules.
            # A common approach is if any agent terminates, or a global termination condition.
            # Using BattlesnakeGym's 'done' from info if available, or individual terminations.
            
            # Using the original logic: done if all agents in first team are done.
            # Assuming self.num_learning_agents = 2 for the first team.
            done_by_maxlen = current_episode_length >= self.max_ep_len
            is_episode_done = done_by_opponents or done_by_maxlen
            if done_by_env and is_episode_done:
                win_count += 1

            # Policy Updates & Saving (for each learning agent)
            for i in range(self.num_learning_agents):
                agent = self.agents[i]
                agent.total_steps_trained += 1 # Increment step count for this agent

                # Update policy if buffer is full
                if len(agent.buffer) >= self.rollout_steps:
                    # Bootstrap obs for this agent
                    next_obs_bootstrap_agent = get_agent_observed_state(i, next_obs_dict_all_snakes, team_ids[i])
                    # Who is the second_agent for comparison? The other learning agent.
                    # This assumes num_learning_agents is 2.
                    other_agent_idx = (i + 1) % self.num_learning_agents if self.num_learning_agents > 1 else None
                    second_ppo_agent = self.agents[other_agent_idx] if other_agent_idx is not None else None

                    # Termination status for this agent for bootstrap
                    agent_terminated_for_bootstrap = terminated_all_snakes[i]

                    actor_loss, critic_loss, entropy = agent.update_policy(
                        next_obs_bootstrap_agent,
                        agent_terminated_for_bootstrap, # Use this agent's termination for its own bootstrap
                        second_ppo_agent
                    )
                    # Optional: Log these losses with wandb or print
                    # logging.debug(f"Agent {i} Update: ActorL={actor_loss:.3f}, CriticL={critic_loss:.3f}, Entropy={entropy:.3f}")


                # Save model periodically
                # agent.total_steps_trained is the global step for this agent.
                if (agent.total_steps_trained - self.last_save_step_counts[i]) >= self.save_interval_steps:
                    save_path = os.path.join(self.ckpt_dir, f"agent_{i}_steps_{agent.total_steps_trained}.pth")
                    agent.save_model(save_path)
                    self.last_save_step_counts[i] = (agent.total_steps_trained // self.save_interval_steps) * self.save_interval_steps
                    # Potentially update existing policies pool if desired mid-training
                    self._update_existing_policies()


            if is_episode_done:
                self.total_episodes_completed += 1
                termination = [False, False]
                # Aggregate rewards for logging (e.g., average reward of learning agents)
                avg_episode_reward_learning_agents = np.mean([current_episode_rewards[i] for i in range(self.num_learning_agents)])
                self.episode_rewards_buffer.append(avg_episode_reward_learning_agents)
                self.episode_lengths_buffer.append(current_episode_length)

                if self.total_episodes_completed % self.log_interval_episodes == 0 and len(self.episode_rewards_buffer) > 0:
                    avg_reward = np.mean(self.episode_rewards_buffer[-self.log_interval_episodes:])
                    avg_length = np.mean(self.episode_lengths_buffer[-self.log_interval_episodes:])
                    elapsed_time = time.monotonic() - start_time
                    logging.info(
                        f"Eps: {self.total_episodes_completed} | Global Steps (Agent 0): {self.agents[0].total_steps_trained} | "
                        f"Avg Reward (last {self.log_interval_episodes}): {avg_reward:.2f} | Win %: {win_count / self.total_episodes_completed * 100} | Avg Len: {avg_length:.2f} | Time: {elapsed_time:.2f}s"
                    )
                
                # Reset for next episode
                obs_dict_all_snakes, _, _, _ = self.env.reset() #seed=self.seed + self.total_episodes_completed
                current_episode_rewards = [0.0] * self.num_learning_agents
                current_episode_length = 0
                # Policies are reset at the start of the loop if current_episode_length == 0

        logging.info(f"Training finished. Total time: {(time.monotonic() - start_time)/60:.2f} minutes.")
        for i in range(self.num_learning_agents):
            agent = self.agents[i]
            save_path = os.path.join(self.ckpt_dir, f"agent_{i}_steps_{agent.total_steps_trained}_final.pth")
            agent.save_model(save_path)
        self.env.close()


def main():
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    # MPS check from original code (can be enabled if desired and PyTorch supports it well for the models)
    # if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    #     device_str = "mps"
    #     logging.info("Using MPS device (Apple Silicon).")
    logging.info(f"Using device: {device_str}")

    config: Dict[str, Any] = {
        "env_id": 'BattleSnake',
        "total_training_timesteps": 2_000_000, # This is per-agent effectively, or total env steps
        "rollout_steps": 2048,
        "hidden_dim": 64,
        "lr": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "num_epochs": 10,
        "minibatch_size": 256, # Was 64, if GPU memory allows, 128 or 256 can be faster
        "eps_clip": 0.2,
        "entropy_coef": 0.01,
        "value_loss_coef": 0.5,
        "max_grad_norm": 0.5,
        "seed": 42,
        "max_ep_len": 500,
        "log_interval_episodes": 100,
        "save_interval_steps": 50_000, # Per agent
        # everything had -0.2 for dying, +1 and -1 for win/loss
        # "ckpt_dir": "./models/PPO_BattleSnake_NEW_WIN_LOSS_PENALTIES_FOR_DEATH_AND_REWARDS_FOR_KILLS",
        "ckpt_dir": "./models/PPO_BattleSnakes_go_offense_no_contact_hopefully_less_overfit",
        # Example: "./models/PPOBattlesnake_Corrected/agent_0_steps_50000.pth"
        "load_model_path_agent0": "", # Path for agent 0
        "load_model_path_agent1": "", # Path for agent 1
        "render_mode": False, # Set to True to watch
        "render_freq": 0.5, # Time in seconds between rendered frames, e.g., 0.1 for 10 FPS
    }
    """
        "another_turn": 0.000,
        "ate_food": 1.,
        "won": 10,
        "died": -10,
        "ate_another_snake": 3.0,
        "hit_wall": -0.0,
        "hit_other_snake": 0,
        "hit_self": -1.0,
        "was_eaten": -1.0,
        "other_snake_hit_body": -1,
        "forbidden_move": -1.0,
        "starved": -5.0
        """

    map_size = (11, 11)
    n_snakes = 4 # Standard Battlesnake usually has 2 or 4. This code assumes 4.
    num_learning_agents = 2 # Let's assume we are training the first 2 snakes

    # Create dummy env to get action_dim (obs_dim handled by network structure)
    # IMPORTANT: Use the same observation settings as the training env
    dummy_env = BattlesnakeGym(map_size=map_size, number_of_snakes=n_snakes)
    _ = dummy_env.reset() # obs_shape from this not directly used for PolicyValueNetwork init
    action_dim = dummy_env.action_space[0].n # Assuming all snakes have same action space
    dummy_env.close()
    logging.info(f"Action dimension: {action_dim}")


    # --- Agent Initialization ---
    # Create two separate PPOAgent instances for two learning agents
    agent_args = {
        "action_dim": action_dim,
        "lr": config["lr"],
        "num_epochs": config["num_epochs"],
        "eps_clip": config["eps_clip"],
        "gamma": config["gamma"],
        "gae_lambda": config["gae_lambda"],
        "entropy_coef": config["entropy_coef"],
        "value_loss_coef": config["value_loss_coef"],
        "hidden_dim": config["hidden_dim"],
        "minibatch_size": config["minibatch_size"],
        "device": device_str,
        "max_grad_norm": config["max_grad_norm"]
    }
    
    ppo_agent1 = PPOAgent(**agent_args, load_model_path=config["load_model_path_agent0"])
    ppo_agent2 = PPOAgent(**agent_args, load_model_path=config["load_model_path_agent1"])
    
    learning_agents = [ppo_agent1, ppo_agent2]

    # --- Trainer Initialization ---
    # IMPORTANT: Ensure BattlesnakeGym uses the default grid observation, not "flat-num"
    # as get_agent_observed_state and the CNN expect grid-like input.
    env_for_training = BattlesnakeGym(map_size=map_size, number_of_snakes=n_snakes)

    # dummy_agent for PPOTrainer is used to get network parameters for loading fixed policies
    # It can be one of the learning agents.
    trainer = PPOTrainer(
        env=env_for_training,
        agents=learning_agents,
        rollout_steps=config["rollout_steps"],
        log_interval_episodes=config["log_interval_episodes"],
        save_interval_steps=config["save_interval_steps"],
        ckpt_dir=config["ckpt_dir"],
        render_mode=config["render_mode"],
        seed=config["seed"],
        max_ep_len=config["max_ep_len"],
        render_freq=config["render_freq"],
        dummy_agent=ppo_agent1 # Used for params when loading opponent policies
    )

    logging.info(f"Initial existing policies: {[type(p) for p in trainer.existing_policies]}")
    # policies, indices = trainer._set_policies_for_agents()
    # logging.info(f"Sample game policies setup: {[type(p) for p in policies]}, Indices: {indices}")


    try:
        trainer.train(total_training_timesteps=config["total_training_timesteps"])
    except KeyboardInterrupt:
        logging.info("Training interrupted by user (KeyboardInterrupt).")
    except Exception as e:
        logging.error(f"An unexpected error occurred during training: {e}", exc_info=True)
    finally:
        logging.info("Training run concluded.")
        if env_for_training: # Ensure env is closed if it was created
            env_for_training.close()

if __name__ == "__main__":
    main()