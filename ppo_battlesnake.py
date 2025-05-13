# Standard library imports
import numpy as np
import os
import time
import logging # For structured logging
from typing import Tuple, Dict, Any # For type hinting, Tuple is crucial for Python < 3.9

# PyTorch related imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from battlesnake_gym.snake_gym import BattlesnakeGym

# Environment import (Gymnasium is the successor to Gym)
import gym as gym

# Optional: Weights & Biases for experiment tracking
# import wandb

# --- Basic Logging Configuration ---
# Configures the logging module to output messages with a timestamp, level, and message.
# This helps in debugging and tracking the training process.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_agent_observed_state(agent_id, observation):
    """
    Agent state is input as 11x11 grids on n+1 dimensions. First dimension is food positions, rest of them are positions of each agent.
    This is a way to modify that to get a 3x11x11 grid with agent position, ennemy positions and food positions

    Agent ID is the current agent's id, and we want to get an output wrt this agent.
    """
    food_spaces = observation[:, :, 0]
    agent_positions = observation[:, :, agent_id + 1]

    remaining_agent_positions = observation[:, :, 1:]
    remaining_agent_positions = np.sum(remaining_agent_positions, axis = 2) - agent_positions

    stacked = np.stack([agent_positions, remaining_agent_positions, food_spaces])
    stacked = np.array(stacked, dtype = np.float32)

    return stacked

class RolloutBuffer:
    """
    Stores trajectories (sequences of states, actions, rewards, etc.) collected during environment interaction.
    This data is used by the PPO agent to update its policy.
    """
    def __init__(self):
        """Initializes empty lists to store trajectory data."""
        self.states = []         # List of states encountered
        self.actions = []        # List of actions taken
        self.logprobs = []       # List of log probabilities of actions taken
        self.rewards = []        # List of rewards received
        self.state_values = []   # List of state values estimated by the critic
        self.is_terminals = []   # List of booleans indicating if a state was terminal (episode ended naturally)
                                 # Truncated episodes (ended by max_ep_len) are handled by GAE bootstrapping.

    def store_transition(self, state: np.ndarray, action: int, logprob: float, reward: float, state_value: float, is_terminal: bool):
        """
        Appends a single transition (one step of interaction) to the buffer.
        
        Args:
            state: The state observed from the environment.
            action: The action taken by the agent.
            logprob: The log probability of the chosen action under the current policy.
            reward: The reward received from the environment.
            state_value: The value of the current state, estimated by the critic.
            is_terminal: Boolean indicating if the episode terminated at this step.
        """
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.state_values.append(state_value)
        self.is_terminals.append(is_terminal)
    
    def clear(self):
        """Clears all stored transitions from the buffer. Called after a policy update."""
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

    def __len__(self) -> int:
        """Returns the current number of transitions stored in the buffer."""
        return len(self.states)

# Assuming action indices: 0=up, 1=down, 2=left, 3=right
def initial_random_policy(observation, action_space):
    agent_state = observation[0]
    if (np.sum(agent_state == 5) == 0):
        # No agent head found, return a random action
        return action_space.sample()
    agent_head_index = (np.where(agent_state == 5)[0][0], np.where(agent_state == 5)[1][0])
    enemy_states = observation[1]
    new_head_indices = {
        (agent_head_index[0] - 1, agent_head_index[1]): 0, # Up
        (agent_head_index[0] + 1, agent_head_index[1]): 1, # Down
        (agent_head_index[0], agent_head_index[1] - 1): 2, # Left
        (agent_head_index[0], agent_head_index[1] + 1): 3  # Right
    }

    # Filter out invalid moves (out of bounds or into walls)
    valid_moves = []
    for i, ((x, y), move) in enumerate(new_head_indices.items()):
        if 0 <= x < agent_state.shape[0] and 0 <= y < agent_state.shape[1]:
            if agent_state[x, y] != 1 and enemy_states[x, y] == 0: # Assuming 1 is a wall
                valid_moves.append(move)

    # print(f"Valid moves: {valid_moves}")
    if len(valid_moves) > 0:
        action = np.random.choice(valid_moves)
    else:
        action = action_space.sample() # Fallback to random action if no valid moves
    return action
# Optional: JIT compilation can sometimes provide speedups for neural network forward passes.
# To use it, uncomment the line below. Test performance with and without it.
# @torch.jit.script
class PolicyValueNetwork(nn.Module):
    """
    Actor-Critic Network for PPO.
    It has a shared feature extractor, an actor head (outputs action probabilities),
    and a critic head (outputs state value).
    """
    def __init__(self, n_obs: int, n_actions: int, hidden_dim: int = 64, device: str = 'cpu'):
        """
        Initializes the network layers.
        
        Args:
            n_obs: Dimensionality of the observation space.
            n_actions: Number of possible discrete actions.
            hidden_dim: Number of units in the hidden layers.
            device: The device (e.g., 'cpu', 'cuda') to run the network on.
        """
        super(PolicyValueNetwork, self).__init__()
        self.device = device

        # Shared feature extractor (MLP)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # (3,11,11) -> (16,11,11)
            nn.Tanh(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # (16,11,11) -> (32,11,11)
            nn.Tanh(),
            nn.Flatten(start_dim=1),  # (32,11,11) -> (32*11*11,)
            nn.Linear(32 * 11 * 11, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        ).to(device)
 # Move layers to the specified device

        # Actor head: outputs probabilities for each discrete action
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, n_actions),
            nn.Softmax(dim=-1) # Softmax to get action probabilities
        ).to(device)
        
        # Critic head: outputs a single value representing the estimated state value
        self.critic_head = nn.Linear(hidden_dim, 1).to(device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass through the network.
        
        Args:
            x: Input tensor representing the state observation(s).
               Can be a single observation or a batch.
               
        Returns:
            A tuple containing:
                - action_probs: Tensor of action probabilities.
                - state_value: Tensor of estimated state value(s).
        """
        # Ensure input is a tensor and on the correct device
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        elif x.device != self.device: # If tensor is on wrong device, move it
            x = x.to(self.device)
        
        # If input is a single observation (1D), add a batch dimension (becomes 2D)
        if x.ndim == 1:
             x = x.unsqueeze(0)

        # Pass through the shared feature extractor
        features = self.feature_extractor(x)
        # print(f"Features shape: {features.shape}")
        # Get action probabilities from the actor head
        action_probs = self.actor_head(features)
        # print(f"Action probabilities shape: {action_probs.shape}")
        # Get state value from the critic head
        state_value = self.critic_head(features)
        # print(f"State value shape: {state_value.shape}")
        return action_probs, state_value
    
    @torch.no_grad() # Disables gradient calculations for action selection (inference mode)
    def select_action(self, obs: np.ndarray) -> Tuple[int, float, float]:
        """
        Selects an action based on the current policy for a given observation.
        Used during environment interaction.
        
        Args:
            obs: A NumPy array representing the current environment observation.
            
        Returns:
            A tuple containing:
                - action: The selected action (integer).
                - action_logprob: The log probability of the selected action.
                - state_value: The critic's estimate of the value of the current state.
        """
        # The forward method handles tensor conversion and device placement
        action_probs, state_value_tensor = self.forward(obs) 
        
        # Create a categorical distribution from action probabilities
        dist = Categorical(probs=action_probs)
        # Sample an action from the distribution
        action = dist.sample()
        # Calculate the log probability of the sampled action
        action_logprob = dist.log_prob(action)
        
        # Return results as Python native types for environment interaction and buffer storage
        return action.item(), action_logprob.item(), state_value_tensor.item()

    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluates actions taken previously, using the current policy.
        Used during PPO updates to calculate ratios and entropy.
        
        Args:
            states: A batch of states (tensor).
            actions: A batch of actions taken in those states (tensor).
            
        Returns:
            A tuple containing:
                - state_values: Current critic's estimates for the input states.
                - action_logprobs: Log probabilities of the input actions under the current policy.
                - dist_entropy: Entropy of the action distribution for the input states.
        """
        # states and actions are expected to be tensors from the buffer
        action_probs, state_values_tensor = self.forward(states)
        
        dist = Categorical(action_probs)
        # Calculate log probabilities of the given actions under the current policy
        # Actions might have an extra dimension (e.g., [batch_size, 1]), squeeze if needed.
        action_logprobs = dist.log_prob(actions.squeeze(-1))
        # Calculate the entropy of the action distribution (encourages exploration)
        dist_entropy = dist.entropy()
        
        # Squeeze state_values to match dimensions for loss calculation (e.g. [batch_size])
        return state_values_tensor.squeeze(-1), action_logprobs, dist_entropy


class PPOAgent:
    """
    Implements the Proximal Policy Optimization (PPO) agent.
    This class manages the policy network, optimizer, and the PPO update logic.
    """
    def __init__(
            self, 
            n_obs_dim: int,
            action_dim: int, 
            lr: float = 3e-4,           # Learning rate for the optimizer
            num_epochs: int = 10,       # Number of epochs to train on the collected data per update
            eps_clip: float = 0.2,      # Clipping parameter for PPO's surrogate objective
            gamma: float = 0.99,        # Discount factor for future rewards
            gae_lambda: float = 0.95,   # Lambda factor for Generalized Advantage Estimation (GAE)
            entropy_coef: float = 0.01, # Coefficient for the entropy bonus in the loss
            value_loss_coef: float = 0.5,# Coefficient for the critic's value loss
            hidden_dim: int = 64,       # Hidden dimension for the PolicyValueNetwork
            minibatch_size: int = 64,   # Size of minibatches for policy updates
            device: str = 'cpu',        # Device to run computations on ('cpu' or 'cuda')
            load_model_path: str = None,# Path to a pre-trained model to load
            max_grad_norm: float = 0.5  # Maximum norm for gradient clipping (0 to disable)
        ):
        # Store hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.num_epochs = num_epochs
        self.minibatch_size = minibatch_size
        self.eps_clip = eps_clip
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.device = device
        self.max_grad_norm = max_grad_norm

        # Initialize the policy and value network
        self.policy = PolicyValueNetwork(
            n_obs_dim, 
            action_dim,
            hidden_dim=hidden_dim,
            device=device
        )
        # Initialize the optimizer (AdamW is often a good choice)
        self.optimizer = optim.AdamW(self.policy.parameters(), lr=lr, eps=1e-5) # eps for numerical stability
        # Initialize the rollout buffer to store experiences
        self.buffer = RolloutBuffer()
        # Mean Squared Error loss for the critic
        self.mse_loss = nn.MSELoss()
        # Track total steps trained for model loading/saving continuity
        self.total_steps_trained = 0

        # Load a pre-trained model if a path is provided
        if load_model_path:
            self.load_model(load_model_path)

    def load_model(self, path: str):
        """Loads a pre-trained model's state dictionary for policy and optimizer."""
        try:
            checkpoint = torch.load(path, map_location=self.device) # Load to the specified device
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.total_steps_trained = checkpoint.get('total_steps_trained', 0) # Resume step count
            self.policy.eval() # Set policy to evaluation mode after loading
            logging.info(f"Loaded model from {path}, resuming from {self.total_steps_trained} steps.")
        except FileNotFoundError:
            logging.warning(f"Model file not found at {path}. Starting from scratch.")
        except Exception as e:
            logging.error(f"Error loading model from {path}: {e}. Starting from scratch.")
            self.total_steps_trained = 0 # Reset step count if loading fails
            
    def save_model(self, path: str):
        """Saves the current model's state dictionary (policy and optimizer) and total steps trained."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save({
                'policy_state_dict': self.policy.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'total_steps_trained': self.total_steps_trained
            }, path)
            logging.info(f"Saved model to {path} at {self.total_steps_trained} steps.")
        except Exception as e:
            logging.error(f"Error saving model to {path}: {e}")

    def _compute_gae_and_returns(self, next_value_bootstrap: float, next_is_terminal_bootstrap: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes Generalized Advantage Estimation (GAE) and returns (discounted rewards-to-go).
        GAE provides a balance between high-variance Monte Carlo estimates and biased TD estimates.

        Args:
            next_value_bootstrap: The critic's value estimate of the state *after* the last state in the buffer.
                                  Used for bootstrapping if the rollout didn't end with a terminal state.
            next_is_terminal_bootstrap: Boolean indicating if the state *after* the last state in the buffer is terminal.
        
        Returns:
            A tuple of tensors:
                - advantages: Calculated GAE for each step in the buffer.
                - returns: Calculated returns (targets for the value function) for each step.
        """
        num_steps = len(self.buffer.rewards)
        advantages = torch.zeros(num_steps, device=self.device) # Initialize advantages tensor
        # returns = torch.zeros(num_steps, device=self.device) # Not strictly needed if calculated from advantages
        
        # Convert buffer data (lists of Python scalars/NumPy arrays) to PyTorch tensors on the correct device
        rewards = torch.tensor(self.buffer.rewards, dtype=torch.float32, device=self.device)
        state_values = torch.tensor(self.buffer.state_values, dtype=torch.float32, device=self.device)
        is_terminals = torch.tensor(self.buffer.is_terminals, dtype=torch.float32, device=self.device) # 0.0 for non-terminal, 1.0 for terminal

        last_gae_lam = 0.0 # Stores GAE at t+1, used for recursive calculation
        # Iterate backwards through the trajectory
        for t in reversed(range(num_steps)):
            if t == num_steps - 1: # If this is the last step in the buffer
                # Use the bootstrapped values for the state *after* this last step
                next_non_terminal = 1.0 - float(next_is_terminal_bootstrap) # 1.0 if next state is not terminal
                next_val = next_value_bootstrap                             # Value of the state after this last step
            else: # For all other steps
                next_non_terminal = 1.0 - is_terminals[t + 1] # 1.0 if s_{t+1} is not terminal
                next_val = state_values[t + 1]                # Critic's value of s_{t+1}
            
            # Calculate the TD error (delta) for the current step t
            delta = rewards[t] + self.gamma * next_val * next_non_terminal - state_values[t]
            # Calculate GAE for step t using TD error and GAE from step t+1
            advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
        
        # Calculate returns (targets for value function) as GAE + V(s_t)
        returns_to_go = advantages + state_values 
        return advantages, returns_to_go
    
    def update_policy(self, next_obs_for_bootstrap: np.ndarray, next_done_for_bootstrap: bool) -> Tuple[float, float, float]:
        """
        Performs the PPO policy update using data collected in the rollout buffer.
        
        Args:
            next_obs_for_bootstrap: The observation *after* the last step in the buffer. Used for GAE bootstrapping.
            next_done_for_bootstrap: Boolean indicating if the episode terminated *after* the last step in the buffer.
        
        Returns:
            A tuple of floats for logging:
                - avg_actor_loss: Average actor loss over the update epochs.
                - avg_critic_loss: Average critic loss over the update epochs.
                - avg_entropy: Average policy entropy over the update epochs.
        """
        # Estimate the value of the state that comes after the last state in the buffer.
        # This is needed for GAE calculation (bootstrapping).
        with torch.no_grad(): # No gradient needed for this value estimation
            # self.policy.forward handles tensor conversion and device placement
            _, next_value_tensor = self.policy.forward(next_obs_for_bootstrap.reshape(1, 3, 11, 11)) # Reshape to match input dimensions
            next_value_bootstrap = next_value_tensor.item()

        # Compute GAE and returns (targets for the value function)
        advantages, rewards_to_go = self._compute_gae_and_returns(next_value_bootstrap, next_done_for_bootstrap)
        
        # Normalize advantages for more stable updates (subtract mean, divide by std dev)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) # Add epsilon for numerical stability

        # Prepare data from the buffer for batch processing.
        # Convert lists of NumPy arrays/scalars to stacked PyTorch tensors.
        old_states_np = np.array(self.buffer.states, dtype=np.float32) # Assuming states are stored as np.ndarray
        old_states = torch.tensor(old_states_np, dtype=torch.float32, device=self.device)
        
        # Actions are discrete integers, convert to long tensor, add a dimension for batching.
        old_actions = torch.tensor(np.array(self.buffer.actions), dtype=torch.long, device=self.device).unsqueeze(1)
        # Log probabilities are floats, convert to float tensor, add a dimension.
        old_logprobs = torch.tensor(self.buffer.logprobs, dtype=torch.float32, device=self.device).unsqueeze(1)

        num_samples_in_buffer = len(self.buffer)
        
        # Accumulators for average losses and entropy for logging
        total_actor_loss, total_critic_loss, total_entropy = 0.0, 0.0, 0.0

        self.policy.train() # Set the policy network to training mode (enables dropout, batchnorm if used)
        # Perform multiple epochs of updates on the collected batch of data
        for epoch in range(self.num_epochs):
            # Shuffle indices for minibatch creation. `torch.randperm` is efficient on the device.
            indices = torch.randperm(num_samples_in_buffer, device=self.device)

            # Iterate over the data in minibatches
            for start_idx in range(0, num_samples_in_buffer, self.minibatch_size):
                end_idx = min(start_idx + self.minibatch_size, num_samples_in_buffer) # Ensure end_idx doesn't exceed buffer size
                batch_indices = indices[start_idx:end_idx]

                # Slice the data to create minibatches
                batch_states = old_states[batch_indices]
                batch_actions = old_actions[batch_indices]
                batch_old_logprobs = old_logprobs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_rewards_to_go = rewards_to_go[batch_indices]
                
                # Evaluate actions using the current policy to get new logprobs, values, and entropy
                current_state_values, current_logprobs, dist_entropy = self.policy.evaluate_actions(batch_states, batch_actions)
                
                # Calculate the probability ratio (pi_new / pi_old)
                # Ensure batch_old_logprobs is squeezed if it has an unnecessary dimension from unsqueeze(1)
                ratios = torch.exp(current_logprobs - batch_old_logprobs.squeeze(-1))

                # PPO's clipped surrogate objective for the actor
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean() # Negative because we want to maximize this objective
                
                # Critic loss: Mean Squared Error between estimated values and actual returns
                critic_loss = self.mse_loss(current_state_values, batch_rewards_to_go)
                
                # Total loss: Combination of actor loss, critic loss, and entropy bonus
                # Entropy bonus encourages exploration by penalizing overly deterministic policies.
                loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * dist_entropy.mean()
                
                # Perform backpropagation and optimization step
                self.optimizer.zero_grad() # Clear old gradients
                loss.backward()           # Calculate new gradients
                # Optional: Gradient clipping to prevent exploding gradients
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()      # Update network parameters

                # Accumulate losses for logging
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += dist_entropy.mean().item()
        
        self.policy.eval() # Set the policy network back to evaluation mode
        self.buffer.clear() # Clear the rollout buffer after updates
        
        # Calculate average losses over all minibatches and epochs
        num_update_steps = self.num_epochs * (num_samples_in_buffer // self.minibatch_size + (1 if num_samples_in_buffer % self.minibatch_size != 0 else 0) )
        avg_actor_loss = total_actor_loss / num_update_steps
        avg_critic_loss = total_critic_loss / num_update_steps
        avg_entropy = total_entropy / num_update_steps
        
        return avg_actor_loss, avg_critic_loss, avg_entropy


class PPOTrainer:
    """
    Manages the PPO training loop, including environment interaction,
    data collection, policy updates, logging, and model saving.
    """
    def __init__(self, 
                 env,                # ID of the Gymnasium environment (e.g., 'CartPole-v1')
                 agent: PPOAgent,            # The PPOAgent instance
                 rollout_steps: int = 2048,  # Number of steps to collect in each rollout before updating policy
                 log_interval_episodes: int = 10, # How often to log average episode rewards (in episodes)
                 save_interval_steps: int = 50000, # How often to save the model (in global timesteps)
                 ckpt_dir: str = "models/PPO",  # Directory to save model checkpoints
                 render_mode: str = False,    # Environment render mode ('human', None, etc.)
                 seed: int = 42,             # Random seed for reproducibility
                 max_ep_len: int = 500,       # Maximum length of an episode
                 render_freq = None
                ):
        
        # Initialize the environment
        self.env = env
        self.env.reset() # Seed the environment for reproducibility
        # self.env.action_space.seed() # Seed the environment's action space
        
        self.agent = agent
        self.rollout_steps = rollout_steps
        self.log_interval_episodes = log_interval_episodes
        self.save_interval_steps = save_interval_steps
        self.ckpt_dir = ckpt_dir
        self.max_ep_len = max_ep_len
        self.seed = seed
        self.render = render_mode
        self.render_freq = render_freq

        # Create checkpoint directory if it doesn't exist
        if self.ckpt_dir:
            os.makedirs(self.ckpt_dir, exist_ok=True)

        # Buffers for logging episode statistics
        self.episode_rewards_buffer = [] # Stores rewards of recent episodes for averaging
        self.episode_lengths_buffer = [] # Stores lengths of recent episodes for averaging
        self.total_episodes_completed = 0 # Counter for total episodes finished
        
        # Track when the last save occurred to manage save intervals
        # Initialize based on agent's loaded step count to avoid immediate re-save if close to an interval
        self.last_save_step_count = (agent.total_steps_trained // save_interval_steps) * save_interval_steps

    def train(self, total_training_timesteps: int):
        """
        Runs the main PPO training loop.
        
        Args:
            total_training_timesteps: The total number of environment steps to train for across all runs.
        """
        logging.info(f"Starting training for {total_training_timesteps} timesteps on device {self.agent.device}.")
        start_time = time.monotonic() # Use monotonic time for measuring duration
        
        # Reset environment to get initial observation. Seed is applied once in __init__ or here.
        obs, _, _, _ = self.env.reset() 
        
        current_episode_reward = 0.0 # Accumulates reward for the current episode
        current_episode_length = 0   # Tracks length of the current episode
        
        # The main loop iterates for a total number of timesteps.
        # `self.agent.total_steps_trained` is cumulative across training runs if a model is loaded.
        # Loop from the agent's current trained step count up to the desired total.
        for t_step_global in range(self.agent.total_steps_trained + 1, self.agent.total_steps_trained + total_training_timesteps + 1):

            if self.render:
                self.env.render(mode='human')
                if self.render_freq:
                    time.sleep(self.render_freq)
            
            # --- Step 1: Collect one step of experience ---
            # print("Obs Shape", obs.shape)
            # print("Observation")
            agents_obs = [get_agent_observed_state(i, obs) for i in range(4)]
            
            action, log_prob, state_val = self.agent.policy.select_action(agents_obs[0].reshape(1, 3, 11, 11))
            action_rest_agents = [action] + [initial_random_policy(agents_obs[i], self.env.action_space[i]) for i in range(1, len(agents_obs))]
            next_obs, reward, terminated, info = self.env.step(action_rest_agents)
            # print("Next Obs Shape", next_obs.shape)
            
            current_episode_reward += reward[0]
            current_episode_length += 1
            
            # Store the transition in the agent's buffer.
            # `terminated` is True if the episode ended due to environment-defined conditions (e.g., goal reached, failed).
            # `truncated` is True if the episode ended due to a time limit (e.g., `max_ep_len`).
            self.agent.buffer.store_transition(agents_obs[0], action, log_prob, reward[0], state_val, terminated[0])
            obs = next_obs # Update current observation to the next observation

            # --- Step 2: Check if episode is done ---
            # An episode is 'done' if it's either terminated or truncated, or reaches max_ep_len.
            done_by_env = terminated[0]
            done_by_maxlen = current_episode_length >= self.max_ep_len
            is_episode_done = done_by_env or done_by_maxlen
            
            if is_episode_done:
                self.total_episodes_completed += 1
                self.episode_rewards_buffer.append(current_episode_reward)
                self.episode_lengths_buffer.append(current_episode_length)
                
                # Log episode statistics periodically
                if self.total_episodes_completed % self.log_interval_episodes == 0 and len(self.episode_rewards_buffer) > 0:
                    # Calculate average reward and length over the last `log_interval_episodes`
                    avg_reward = np.mean(self.episode_rewards_buffer[-self.log_interval_episodes:])
                    avg_length = np.mean(self.episode_lengths_buffer[-self.log_interval_episodes:])
                    elapsed_time = time.monotonic() - start_time
                    logging.info(
                        f"Eps: {self.total_episodes_completed} | Steps: {t_step_global}/{self.agent.total_steps_trained + total_training_timesteps} | "
                        f"Avg Reward (last {self.log_interval_episodes}): {avg_reward:.2f} | "
                        f"Avg EpLength: {avg_length:.2f} | Time: {elapsed_time:.2f}s"
                    )
                
                # Reset the environment for the next episode
                obs, _, _, _ = self.env.reset()
                current_episode_reward = 0.0
                current_episode_length = 0

            # --- Step 3: Update policy if rollout buffer is full ---
            if len(self.agent.buffer) >= self.rollout_steps:
                # `done_by_env` is passed for GAE bootstrapping. If the *rollout* ends mid-episode,
                # `done_by_env` will be False, and GAE will bootstrap using V(next_obs).
                # If the *rollout* ends exactly when an episode ends, `done_by_env` will be True, and GAE
                # will use 0 as the value for the terminal state.
                actor_loss, critic_loss, entropy = self.agent.update_policy(np.array(get_agent_observed_state(0, next_obs)), done_by_env)
                # Optional: log update losses
                # logging.debug(f"Update at step {t_step_global}: ActorL={actor_loss:.3f}, CriticL={critic_loss:.3f}, Entropy={entropy:.3f}")

            # --- Step 4: Save model periodically ---
            self.agent.total_steps_trained = t_step_global # Update agent's internal global step counter
            if self.ckpt_dir and (t_step_global - self.last_save_step_count) >= self.save_interval_steps and t_step_global > self.last_save_step_count :
                save_path = os.path.join(self.ckpt_dir, f"ppo_steps_{t_step_global}.pth")
                self.agent.save_model(save_path)
                # Update last_save_step_count to the multiple of save_interval_steps at or before current step
                self.last_save_step_count = (t_step_global // self.save_interval_steps) * self.save_interval_steps

        # --- End of Training Loop ---
        logging.info(f"Training finished. Total time: {(time.monotonic() - start_time)/60:.2f} minutes.")
        # Save the final model
        if self.ckpt_dir:
            save_path = os.path.join(self.ckpt_dir, f"ppo_steps_{self.agent.total_steps_trained}_final.pth")
            self.agent.save_model(save_path)
        self.env.close() # Close the environment


def main():
    """Main function to configure and run the PPO training."""
    # Determine compute device (CUDA if available, else CPU)
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built(): # Check for MPS (Apple Silicon)
         pass # device_str = "mps" # Uncomment if you want to try MPS, but be aware of potential PyTorch compatibility issues.
    
    # --- Configuration Dictionary ---
    # Centralized configuration for all hyperparameters and settings.
    config: Dict[str, Any] = {
        "env_id": 'BattleSnake',     # Environment ID from Gymnasium
        "total_training_timesteps": 200_000, # Total environment steps for training
        "rollout_steps": 2048,       # N: Number of steps to collect per PPO update cycle
        "hidden_dim": 64,            # Number of units in hidden layers of the PolicyValueNetwork
        "lr": 3e-4,                  # Learning rate for the AdamW optimizer
        "gamma": 0.99,               # Discount factor for future rewards
        "gae_lambda": 0.95,          # Lambda parameter for Generalized Advantage Estimation
        "num_epochs": 10,            # K: Number of optimization epochs per PPO update cycle
        "minibatch_size": 64,        # Size of minibatches used during optimization epochs
        "eps_clip": 0.2,             # PPO clipping parameter (epsilon)
        "entropy_coef": 0.01,        # Coefficient for the entropy bonus in the loss function
        "value_loss_coef": 0.5,      # Coefficient for the critic's value loss in the total loss
        "max_grad_norm": 0.5,        # Maximum norm for gradient clipping (0 to disable)
        "seed": 42,                  # Random seed for reproducibility
        "max_ep_len": 500,           # Maximum number of steps per episode (specific to env, e.g., CartPole-v1 is 500)
        "log_interval_episodes": 20, # Log average performance every N episodes
        "save_interval_steps": 50_000, # Save model checkpoint every N global timesteps
        "ckpt_dir": "./models/PPO_BattleSnake", # Relative directory to save model checkpoints
        "load_model_path": None, # Path to a pre-trained model to load (e.g., "./models/PPO_CartPole/ppo_steps_100000.pth")
        "render_mode": False,          # Environment render bool - True or False
        "render_freq": 0.5,         # Time interval (in seconds) to render the environment 
    }

    # --- Environment Setup (to get observation and action dimensions) ---
    # Create a temporary dummy environment to inspect its properties.
    # try:
    map_size = (11, 11)
    n_snakes = 4

    dummy_env = BattlesnakeGym(map_size=map_size, number_of_snakes=n_snakes)
    obs, _, _, _ = dummy_env.reset()
    n_obs_dim = dummy_env.observation_space.shape # Assuming 1D observation space
    # n_obs_dim = (3, map_size[0], map_size[1]) # Assuming 3 channels for RGB-like observation
    action_dim = dummy_env.action_space[0].n # Assuming discrete action space
    print(f"Observation space shape: {n_obs_dim}")
    print(f"Action space shape: {action_dim}")

    modified_obs = get_agent_observed_state(0, obs)
    initial_random_policy(modified_obs, dummy_env.action_space[0])
    

    dummy_env.close()



    # --- Agent Initialization ---
    ppo_agent = PPOAgent(
        n_obs_dim=n_obs_dim,
        action_dim=action_dim,
        lr=config["lr"],
        num_epochs=config["num_epochs"],
        eps_clip=config["eps_clip"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        entropy_coef=config["entropy_coef"],
        value_loss_coef=config["value_loss_coef"],
        hidden_dim=config["hidden_dim"],
        minibatch_size=config["minibatch_size"],
        device=device_str,
        load_model_path=config["load_model_path"],
        max_grad_norm=config["max_grad_norm"]
    )

    # --- Trainer Initialization ---
    env = BattlesnakeGym(map_size=map_size, number_of_snakes=n_snakes)
    trainer = PPOTrainer(
        env=env,
        agent=ppo_agent,
        rollout_steps=config["rollout_steps"],
        log_interval_episodes=config["log_interval_episodes"],
        save_interval_steps=config["save_interval_steps"],
        ckpt_dir=config["ckpt_dir"],
        render_mode=config["render_mode"],
        seed=config["seed"],
        max_ep_len=config["max_ep_len"],
        render_freq=config["render_freq"]
    )
    
    # --- Start Training ---
    try:
        trainer.train(total_training_timesteps=config["total_training_timesteps"])
    except KeyboardInterrupt: # Allow graceful interruption of training
        logging.info("Training interrupted by user (KeyboardInterrupt).")
    except Exception as e: # Catch any other unexpected errors during training
        logging.error(f"An unexpected error occurred during training: {e}", exc_info=True)
    finally:
        logging.info("Training run concluded.")
        # The trainer's `train` method handles final saving and closing the environment.

if __name__ == "__main__":
    main()