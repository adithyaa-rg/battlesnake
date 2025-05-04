
# @misc{ppo_pytorch,
#     author = {Azim, Saqib},
#     title = {Proximal Policy Optimization using PyTorch},
#     year = {2024},
#     publisher = {GitHub},
#     journal = {GitHub repository},
#     howpublished = {\url{https://github.com/saqib1707/RL-PPO-PyTorch}},
# }
# Based on the above code


import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.distributions import Categorical
import wandb

from battlesnake_gym.snake_gym import BattlesnakeGym


class RolloutBuffer:
    def __init__(self):
        self.states = {1: [], 2: [], 3: [], 4: []}
        self.actions = {1: [], 2: [], 3: [], 4: []}
        self.logprobs = {1: [], 2: [], 3: [], 4: []}
        self.rewards = {1: [], 2: [], 3: [], 4: []}
        self.state_values = {1: [], 2: [], 3: [], 4: []}
        self.dones = {1: [], 2: [], 3: [], 4: []}

    def store_transition(self, state, action, logprob, reward, done, state_value, key):
        self.states[key].append(state)
        self.actions[key].append(action)
        self.logprobs[key].append(logprob)
        self.rewards[key].append(reward)
        self.state_values[key].append(state_value)
        self.dones[key].append(done)
    
    def clear(self, key):
        self.states[key].clear()
        self.actions[key].clear()
        self.logprobs[key].clear()
        self.rewards[key].clear()
        self.state_values[key].clear()
        self.dones[key].clear()

class PolicyValueNetwork(nn.Module):
    def __init__(self, n_actions, device = 'cpu'):
        super(PolicyValueNetwork, self).__init__()

        self.n_actions = n_actions
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 15, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(start_dim = 1),
            nn.Linear(15 * 4 * 4, 128),
            nn.ReLU()
        ).to(device)

        self.actor_head = nn.Sequential(
                nn.Linear(128, n_actions, dtype=torch.float32),
                nn.Softmax(dim=-1)
            ).to(device)
        
        self.critic_head = nn.Linear(128, 1).to(device)

        self.device = device

    def forward(self, x):

        features = self.feature_extractor(x)

        policy_probs = self.actor_head(features)
        state_value = self.critic_head(features)
        return policy_probs, state_value
    
    def select_action(self, obs):
        with torch.no_grad():
            action_out, value = self.forward(obs)
            # print('stage-0:', action_out.shape, value, obs.shape)

            dist = torch.distributions.Categorical(probs=action_out)
            actions = dist.sample()
            action_logprobs = dist.log_prob(actions)

            # print(actions, action_logprobs, value)
        return actions.cpu().numpy(), action_logprobs.cpu().numpy(), value.cpu().numpy()
    
    def evaluate_actions(self, states, actions):
        action_out, values = self.forward(states)

        dist = Categorical(action_out)
        action_logprobs = dist.log_prob(actions.squeeze(-1).long())
        dist_entropy = dist.entropy()

        return values.squeeze(), action_logprobs, dist_entropy


class PPOAgent:
    def __init__(
            self, 
            action_dim, 
            lr_actor, 
            lr_critic, 
            num_epochs=10, 
            eps_clip=0.2, 
            gamma=0.99,
            entropy_coef=0.01,
            value_loss_coef=0.5,
            batch_size=64,
            device='cpu'
        ):
        self.gamma = gamma
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.eps_clip = eps_clip
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.action_dim = action_dim
        self.device = device

        self.policy = PolicyValueNetwork(
            action_dim, 
            device=device,
        )

        self.optimizer = torch.optim.Adam([
            {'params': self.policy.feature_extractor.parameters()},
            {'params': self.policy.actor_head.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic_head.parameters(), 'lr': lr_critic}
        ])

        self.buffer = RolloutBuffer()
        self.mse_loss = nn.MSELoss()

        self.keys = [1, 2, 3, 4]

    def compute_returns(self):
        overall_returns = {
            1: [],
            2: [],
            3: [],
            4: []
        }

        for key in self.keys:
            returns = overall_returns[key]
            discounted_reward = 0
            for reward, done in zip(reversed(self.buffer.rewards[key]), reversed(self.buffer.dones[key])):
                if done:
                    discounted_reward = 0
                discounted_reward = reward + self.gamma * discounted_reward
                returns.insert(0, discounted_reward)

            returns = np.array(returns, dtype=np.float32)
            returns = torch.flatten(torch.from_numpy(returns).float()).to(self.device)
            overall_returns[key] = returns
        return overall_returns
    
    def update_policy(self):
        rewards_to_go = self.compute_returns()

        for key in self.keys:
            states = torch.from_numpy(np.array([i.detach().numpy() for i in self.buffer.states[key]])).float().to(self.device)
            actions = torch.from_numpy(np.array(self.buffer.actions[key])).float().to(self.device)
            old_logprobs = torch.from_numpy(np.array(self.buffer.logprobs[key])).float().to(self.device)
            state_vals = torch.from_numpy(np.array(self.buffer.state_values[key])).float().to(self.device)

            advantages = rewards_to_go[key] - state_vals
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)


            for _ in range(self.num_epochs):
                indices = np.random.permutation(len(self.buffer.states[key]))

                for start_idx in range(0, len(states), self.batch_size):
                    end_idx = start_idx + self.batch_size
                    batch_indices = indices[start_idx:end_idx]

                    batch_states = states[batch_indices]
                    batch_actions = actions[batch_indices]
                    batch_old_logprobs = old_logprobs[batch_indices]
                    batch_advantages = advantages[batch_indices]
                    batch_rewards_to_go = rewards_to_go[key][batch_indices]
                    
                    # evaluate old actions and values
                    state_values, logprobs, dist_entropy = self.policy.evaluate_actions(batch_states, batch_actions)
                    # print(logprobs.shape, batch_old_logprobs.shape)

                    # Finding the ratio (pi_theta / pi_theta_old)
                    ratios = torch.exp(logprobs - batch_old_logprobs.squeeze(-1))

                    # Finding Surrogate Loss
                    # print(ratios.shape, batch_advantages.shape)
                    surr1 = ratios * batch_advantages
                    surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * batch_advantages

                    # final loss of clipped objective PPO
                    actor_loss = -torch.min(surr1, surr2).mean()
                    # print(state_values.dtype, batch_rewards_to_go.dtype)
                    critic_loss = 0.5 * self.mse_loss(state_values.squeeze(), batch_rewards_to_go)
                    loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * dist_entropy.mean()
                    # print("Final loss:", actor_loss, critic_loss, dist_entropy, loss)

                    # calculate gradients and backpropagate for actor network
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
        
            self.buffer.clear(key=key)

class PPOTraining:
    def __init__(self, env, ppo: PPOAgent, n_observations, n_channels, map_size, update_interval, log_interval, save_interval, ckpt_directory):
        self.env = env
        self.buffer = RolloutBuffer()
        self.PPO = ppo
        self.n_observations = n_observations
        self.n_channels = n_channels
        self.map_size = map_size
        self.keys = [1, 2, 3, 4]
        self.update_interval = update_interval
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.ckpt_dir = ckpt_directory

    def get_agent_observed_state(self, agent_id, observation):
        food_spaces = observation[:, :, 0]
        agent_positions = observation[:, :, agent_id + 1]

        remaining_agent_positions = observation[:, :, 1:]
        remaining_agent_positions = np.sum(remaining_agent_positions, axis = 2) - agent_positions

        stacked = np.stack([agent_positions, remaining_agent_positions, food_spaces])
        stacked = np.array(stacked, dtype = np.float32)

        return stacked


    def _collect_trajectory(self, max_eps_steps, num_train_steps, wandb = None, logpath = None):
        observation_agents = np.empty((self.n_observations[-1] - 1, self.n_channels, self.map_size[0], self.map_size[1]))
        state_agents = np.empty((self.n_observations[-1] - 1, self.n_channels, self.map_size[0], self.map_size[1]))
        print(f"Observation Space Shape: {observation_agents.shape}")

        running_eps_reward = 0
        running_eps_length = 0
        running_num_eps = 0

        t_so_far = 0
        eps_so_far = 0

        metrics = {
        'eps_rewards': [],
        'eps_lengths': [],
        'mean_reward': 0, 
        'mean_eps_length': 0,
        'num_episodes': 0,
        }
        
        while t_so_far < num_train_steps:

            state, _, _, info = self.env.reset()
            for j in range(self.n_observations[-1] - 1):
                state_space = self.get_agent_observed_state(j, state)
                state_agents[j] = torch.tensor(state_space)
            eps_reward = 0
            eps_length = 0

            for _ in range(1, max_eps_steps + 1):
                
                state_agents = torch.tensor(state_agents, device = self.PPO.device, dtype=torch.float32)
                actions, logprobs, values = self.PPO.policy.select_action(state_agents)

                next_obs, rewards, terminated, info = self.env.step(actions)

                # print(rewards)

                eps_reward += np.sum(list(rewards.values()))/len(list(rewards.values()))
                t_so_far += 1
                eps_length += 1

                termination = np.array(list(terminated.values()), dtype = np.int32)
                done = np.count_nonzero(termination) == termination.shape[0] - 1 or np.count_nonzero(termination) == termination.shape[0]

                for i, key in enumerate(self.keys):
                    self.PPO.buffer.store_transition(state_agents[i], actions[i], logprobs[i], rewards[i], terminated[i], values[i], key=key)


                if t_so_far % self.update_interval == 0:
                    self.PPO.update_policy()

                if t_so_far % self.log_interval == 0:
                    running_eps_reward /= running_num_eps
                    running_eps_length /= running_num_eps

                    print(f'episode: {eps_so_far} | step: {t_so_far} | reward: {running_eps_reward:.4f} | episode length: {running_eps_length}')

                    if logpath is not None:
                        with open(logpath, 'a') as f:
                            f.write(f'episode: {eps_so_far} | step: {t_so_far} | reward: {running_eps_reward:.4f} | episode length: {running_eps_length}\n')

                    if wandb is not None:
                        wandb.log({
                            "mean_episode_reward": running_eps_reward,
                            "mean_episode_length": running_eps_length,
                            "episode": eps_so_far,
                            "total_steps": t_so_far,
                        }, step=t_so_far)

                    running_eps_reward = 0
                    running_eps_length = 0
                    running_num_eps = 0

                if t_so_far % self.save_interval == 0:
                    checkpoint_path = os.path.join(self.ckpt_dir, f"battlesnake_step_{t_so_far}.pt")
                    torch.save(self.PPO.policy.state_dict(), checkpoint_path)

                state = next_obs
                if done:
                    break
            metrics['eps_rewards'].append(eps_reward)
            metrics['eps_lengths'].append(eps_length)

            running_eps_reward += eps_reward
            running_eps_length += eps_length
            running_num_eps += 1
            eps_so_far += 1
        
        if wandb:
            wandb.finish()    # close wandb logging
        # print(f"Training time: {(time.time()-start_time) / 60.0:.2f} mins")


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Device: {device}")

    ppo = PPOAgent(
        action_dim=4,
        lr_actor=1e-4,
        lr_critic=1e-3
    )

    # Game Settings
    map_size = (11, 11)
    n_snakes = 4

    env = BattlesnakeGym(map_size=map_size, number_of_snakes=n_snakes)
    env.seed(42)

    n_actions = env.action_space[0].n
    print(f"Number of Actions: {n_actions}")

    observation, reward, terminated, info = env.reset()
    n_observations = observation.shape
    print(f"Number of Observations: {n_observations}")

    ## Number of Agents
    n_agents = 4
    n_channels = 3


    training_ppo = PPOTraining(
        env = env,
        ppo=ppo,
        n_observations=n_observations,
        n_channels=n_channels,
        map_size=map_size,
        update_interval=50,
        log_interval=30,
        save_interval=100,
        ckpt_directory="/home/adithyaa/KTH/battlesnake/models/PPO"
    )

    run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="mas4",
    # Set the wandb project where this run will be logged.
    project="mas-ppo-battlesnake",
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": 1e-4,
        "architecture": "CNN+PPO",
        "dataset": "BattleSnake"
    },

)

    training_ppo._collect_trajectory(50, 20000, wandb=run, logpath= None)

if __name__ == "__main__":
    main()