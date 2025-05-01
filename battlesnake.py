import gym
import numpy as np
import time

from replay_buffer import *
from dqn import *

from battlesnake_gym.snake_gym import BattlesnakeGym
from battlesnake_gym.snake import Snake

import matplotlib.pyplot as plt

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

# Maybe modify the direction of snake head to reduce symmetry

def get_agent_observed_state(agent_id, observation):
    food_spaces = observation[:, :, 0]
    agent_positions = observation[:, :, agent_id + 1]

    remaining_agent_positions = observation[:, :, 1:]
    remaining_agent_positions = np.sum(remaining_agent_positions, axis = 2) - agent_positions

    return np.stack([agent_positions, remaining_agent_positions, food_spaces])

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
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

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


policy_net = DQN(n_actions).to(device)
target_net = DQN(n_actions).to(device)

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
        probab_dist = softmax(policy_net(states[i]).numpy(), temperature)
        choice = np.random.choice(n_actions, p = probab_dist)
        actions.append(choice)

    return np.array(actions)

episode_durations = []

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
    num_episodes = 600
else:
    num_episodes = 50

observation_agents = np.empty((n_observations[-1] - 1, n_channels, map_size[0], map_size[1]))
state_agents = np.empty((n_observations[-1] - 1, n_channels, map_size[0], map_size[1]))
print(f"Observation Space Shape: {observation_agents.shape}")

for i_episode in range(num_episodes):
    # Initialize the environment and get its state
    if i_episode % 10 == 0:
        print(f"Episode {i_episode} started")
    state, _, _, info = env.reset()
    for j in range(n_observations[-1] - 1):
        state_space = get_agent_observed_state(j, state)
        state_agents[j] = state_space

    # state = torch.tensor(state, device=device).unsqueeze(0)
    state_agents = torch.tensor(state_agents, device = device, dtype=torch.float32)

    for t in count():
        actions = select_action(state_agents)
        observation, rewards, terminated, _ = env.step(actions)
        done = np.count_nonzero(terminated) == terminated.shape[0] - 1

        for agent in range(n_observations[-1] - 1):
            if terminated[agent]:
                next_state = None
            else:
                observation_agents[agent] = get_agent_observed_state(agent, observation)
                next_state = torch.tensor(observation_agents[agent], dtype=torch.float32, device=device)

            # Store the transition in memory
            reward = torch.tensor([rewards[agent]], device=device)
            memory.push(state_agents[agent], actions[agent], next_state, reward)

            # Move to the next state
            state_agents[agent] = next_state

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
            plot_durations()
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
