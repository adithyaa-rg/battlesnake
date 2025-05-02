import gym
import numpy as np
import time
import os
from collections import deque

from replay_buffer import *
from dqn import *

from battlesnake_gym.snake_gym import BattlesnakeGym
from battlesnake_gym.snake import Snake

import matplotlib.pyplot as plt

import wandb

# Start a new wandb run to track this script.
run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="mas4",
    # Set the wandb project where this run will be logged.
    project="mas-dqn-battlesnake",
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": 1e-4,
        "architecture": "CNN+DQN",
        "dataset": "BattleSnake"
    },

)

actions = {
    "up": 0,
    "down": 1,
    "left": 2,
    "right": 3,
}
# is_move_safe = {"up": True, "down": True, "left": True, "right": True}

BATCH_SIZE = 128
GAMMA = 0.99
TEMP_START = 100
EPS_END = 0.1
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# Game Settings
map_size = (11, 11)
n_snakes = 4

env = BattlesnakeGym(map_size=map_size, number_of_snakes=n_snakes)
env.seed(42)

## Number of Agents
n_agents = 4


device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
print(f"Device: {device}")

# Check gpu

# Maybe modify the direction of snake head to reduce symmetry
weights_dir = "models/DQN"
os.makedirs(weights_dir, exist_ok=True)

def find_latest_checkpoint(weights_dir, name):
    """
    Find the latest checkpoint number from the saved checkpoint files.
    This includes both 2000-episode files and the temporary 100-episode files.

    Returns:
        latest_checkpoint_num (int): The latest checkpoint episode number, or None if no checkpoints found.
    """
    latest_checkpoint_num = None
    
    # List all files in the weights directory
    for filename in os.listdir(weights_dir):
        # Look for both 2000-episode files and the temporary 100-episode files
        if filename.startswith(f"{name}_dqn_weights_ep") and filename.endswith(".pth"):
            try:
                # Extract the episode number from the filename
                episode_num = int(filename.split("_ep")[-1].split(".")[0])
                
                # Update the latest checkpoint number
                if latest_checkpoint_num is None or episode_num > latest_checkpoint_num:
                    latest_checkpoint_num = episode_num
            except ValueError:
                continue  # Skip files that don't match the expected format
    
    if latest_checkpoint_num:
        print(f"Latest checkpoint number: {latest_checkpoint_num}")
    else:
        print(f"No checkpoints found for {name}.")
    
    return latest_checkpoint_num

def load_model(model, name):
    """
    Load the model weights based on the name.
    If an episode is provided, load the corresponding weights based on the episode logic.
    If no episode is provided, it loads the latest available 2000-episode checkpoint,
    or the latest temporary file (overwritten every 100 episodes).
    """
    # Check for the latest 2000-episode file (most recent)
    latest_2000_file = None
    for filename in os.listdir(weights_dir):
        if filename.startswith(f"{name}_dqn_weights_ep") and filename.endswith(".pth"):
            episode_num = int(filename.split("_ep")[-1].split(".")[0])  # Extract episode number
            if episode_num % 2000 == 0:
                if latest_2000_file is None or episode_num > int(latest_2000_file.split("_ep")[-1].split(".")[0]):
                    latest_2000_file = filename

    # Check for the latest temporary overwrite file (dqn_weights_temp.pth)
    latest_temp_file = f"{name}_dqn_weights_temp.pth" if os.path.exists(os.path.join(weights_dir, f"{name}_dqn_weights_temp.pth")) else None

    # Load the latest available file
    if latest_2000_file:
        checkpoint_path = os.path.join(weights_dir, latest_2000_file)
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"Loaded model weights from {checkpoint_path}")
    elif latest_temp_file:
        checkpoint_path = os.path.join(weights_dir, latest_temp_file)
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"Loaded model weights from {checkpoint_path}")
    else:
        print(f"No model weights found for {name}.")
    
    return model

def save_model(model, episode, name):
    if episode % 2000 == 0:
        # Save as a new file every 2000 episodes
        save_path = os.path.join(weights_dir, f"{name}_dqn_weights_ep{episode}.pth")
    elif episode % 100 == 0:
        # Overwrite the same file every 100 episodes
        save_path = os.path.join(weights_dir, "{name}_dqn_weights_temp.pth")
    else:
        return  # Don't save if not a 100-episode interval

    torch.save(model.state_dict(), save_path)
    print(f"Saved model at episode {episode} to {save_path}")


def get_agent_observed_state(agent_id, observation):
    food_spaces = observation[:, :, 0]
    agent_positions = observation[:, :, agent_id + 1]

    remaining_agent_positions = observation[:, :, 1:]
    remaining_agent_positions = np.sum(remaining_agent_positions, axis = 2) - agent_positions

    stacked = np.stack([agent_positions, remaining_agent_positions, food_spaces])
    stacked = np.array(stacked, dtype = np.float32)

    return stacked

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.stack([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.stack(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    # print("State Batch Size:",state_batch.shape)
    # print("Action Batch Size:",action_batch.shape)
    # print("Reward Batch Size:",reward_batch.shape)
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    # print("State Action Values Size:", state_action_values.shape)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

n_actions = env.action_space[0].n
print(f"Number of Actions: {n_actions}")

observation, reward, terminated, info = env.reset()
n_observations = observation.shape
print(f"Number of Observations: {n_observations}")

n_channels = 3
restart = True

policy_net = DQN(n_actions).to(device)
target_net = DQN(n_actions).to(device)
start_ep = 0
if not restart:
    policy_net = load_model(policy_net, "policynet")
    target_net = load_model(target_net, "targetnet")
    start_ep = find_latest_checkpoint(weights_dir, "policynet")
        
    

target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0

def softmax(x, temp):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(np.divide(x,temp)) / np.sum(np.exp(np.divide(x,temp)), axis=0)

def select_action(states):
    global steps_done

    actions = []
    for i in range(states.shape[0]):
        temperature = max((100 * (1 - steps_done/EPS_DECAY), 0.1))
        state = states[i].unsqueeze(0).to(device)
        # print(f"Shape: {state.shape}")
        actions_fn = policy_net(state).detach().numpy()
        # print(f"Action Space: {actions_fn.shape}")
        probab_dist = softmax(actions_fn[0], temperature)
        choice = np.random.choice(n_actions, p = probab_dist)
        actions.append(choice)

    return np.array(actions)

episode_durations = []
episodic_rewards = []

episode_durs = deque([], maxlen=100)
episodic_rews = deque([], maxlen=100)

def plot_rewards(show_result=False, window_size=100):
    plt.figure(2)
    rewards_t = torch.tensor(episodic_rewards, dtype=torch.float)
    colours = ['r', 'g', 'b', 'y']
    
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    # print(rewards_t.shape)

    def moving_average(data, window):
        if len(data) < window:
            window = len(data)
        return np.convolve(data, np.ones(window)/window, mode='valid')

    for i in range(rewards_t.shape[1]):
        reward_series = rewards_t[:, i].numpy()
        ma = moving_average(reward_series, window_size)
        x = range(len(reward_series) - len(ma), len(reward_series)) if len(ma) > 0 else []
        plt.plot(x, ma, color=colours[i])
    
    plt.pause(0.001)

def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated

if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = start_ep + 10000
else:
    num_episodes = start_ep + 10000



observation_agents = np.empty((n_observations[-1] - 1, n_channels, map_size[0], map_size[1]))
state_agents = np.empty((n_observations[-1] - 1, n_channels, map_size[0], map_size[1]))
print(f"Observation Space Shape: {observation_agents.shape}")

for i_episode in range(start_ep, num_episodes + 1):
    # Initialize the environment and get its state
    accumulated_rewards = np.zeros((4))
    if i_episode % 50 == 0:
        print(f"Episode {i_episode} started")
    state, _, _, info = env.reset()
    for j in range(n_observations[-1] - 1):
        state_space = get_agent_observed_state(j, state)
        state_agents[j] = torch.tensor(state_space)

    # state = torch.tensor(state, device=device).unsqueeze(0)
    state_agents = torch.tensor(state_agents, device = device, dtype=torch.float32)

    for t in count():
        actions = select_action(state_agents)
        observation, rewards, terminated, _ = env.step(actions)
        # env.render()

        termination = np.array(list(terminated.values()), dtype = np.int32)
        done = np.count_nonzero(termination) == termination.shape[0] - 1 or np.count_nonzero(termination) == termination.shape[0]
        # print(termination)

        for agent in range(n_observations[-1] - 1):
            if terminated[agent]:
                # Check for a better terminal representation
                observation_agents[agent] = get_agent_observed_state(agent, np.zeros(state.shape))
                next_state = torch.tensor(observation_agents[agent], dtype=torch.float32, device = device)
            else:
                observation_agents[agent] = get_agent_observed_state(agent, observation)
                next_state = torch.tensor(observation_agents[agent], dtype=torch.float32, device = device)

            # Store the transition in memory
            reward = torch.tensor([rewards[agent]], device=device)
            accumulated_rewards[agent] += rewards[agent]
            action = torch.tensor([[actions[agent]]], device=device)
            # print(state_agents[agent].shape, action.shape, next_state.shape, reward.shape,)
            memory.push(state_agents[agent], action, next_state, reward)

            # Move to the next state
            # print(state_agents[agent].shape, next_state.shape)
            state_agents[agent] = next_state
        # print(f"{t} steps")
        # Perform one step of the optimization (on the policy network)
        optimize_model()
        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        

        if done:
            episode_durations.append(t + 1)
            episode_durs.append(t + 1)

            episodic_rewards.append(accumulated_rewards)
            episodic_rews.append(np.mean(accumulated_rewards))

            run.log(
            {"acc_rew": np.mean(episodic_rews),
             "episode duration": np.mean(episode_durs),
            }, step = i_episode)
            
            save_model(policy_net, i_episode, 'policynet')
            save_model(target_net, i_episode, 'targetnet')

            # if  i_episode % 5 == 0:
            #     plot_durations()
            #     plot_rewards()
            break

print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()

# for i in range(1000):
#     # fig, ax = plt.subplots(4, 3, figsize = (15, 10))
#     action = 0 

#     # print(observation.shape)
#     # my_head = game_state["you"]["body"][0]
#     # my_neck = game_state["you"]["body"][1]

#     action = env.action_space.sample()
#     print(env.action_space)

#     observation, reward, terminated, info = env.step(action)

#     for j in range(n_observations[-1] - 1):
#         obs_space = get_agent_observed_state(j, observation)
#         observation_agents[j] = obs_space

#     obs_tensor = torch.tensor(observation_agents, dtype=torch.float32).to(device)
#     print(policy_net(obs_tensor))

#     # for x in range(4):
#     #     for y in range(3):
#     #         ax[x][y].imshow(observation_agents[x][y])

#     print("Snake Lengths Values", np.count_nonzero(observation[:, :, 1]), np.count_nonzero(observation[:, :, 2]))
#     env.render()
#     # plt.show()
#     time.sleep(1)



#     termination = np.array(list(terminated.values()), dtype = np.int32)

#     if np.count_nonzero(termination) == termination.shape[0] - 1:
#         # print(f"Simulated {i+1} steps")
#         obs, _, done, info = env.reset()
# env.close()
