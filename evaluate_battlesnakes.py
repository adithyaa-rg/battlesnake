import numpy as np
import pandas as pd
import torch
import time
from battlesnake_gym.snake_gym import BattlesnakeGym
from ppo_battlesnake import PolicyValueNetwork, get_agent_observed_state, initial_random_policy

actions = {
    0: "down",
    1: "up",
    2: "left",
    3: "right",
}

env = BattlesnakeGym(
    map_size = (11, 11),
    number_of_snakes = 4,
    # observation_type = "flat-num"
)

win_counter = 0
num_games = 0
obs, _, _, _ = env.reset()

for i in range(1000):
    obs_inputs = [get_agent_observed_state(i, obs) for i in range(4)]
    obs_input = obs_inputs[0]

    actions_agent = np.zeros(4, dtype=np.float32)
    model = PolicyValueNetwork(None, 4, hidden_dim=64)
    model_path = "./models/PPO_Acrobot/ppo_steps_50000.pth"
    checkpoint = torch.load(model_path)

    model.load_state_dict(checkpoint['policy_state_dict'])
    model.eval()

    with torch.no_grad():
        obs_input = torch.from_numpy(obs_input).unsqueeze(0)
        # print(obs_input.shape)
        action_id = model(obs_input)[0].argmax().item()
    actions_agent[0] = action_id
    for j in range(1, 4):
        actions_agent[j] = initial_random_policy(obs_inputs[j], env.action_space[j])

    obs, reward, done, info = env.step(actions_agent)
    # print(done)

    # env.render()
    # time.sleep(0.5)
    # Measure winning rate of agent index 0
    dones = np.array(list(done.values()))
    if dones.sum() == 3 and dones[0] != 1:
        win_counter += 1
    if dones[0] == 1 or dones.sum() >= 3:
        num_games += 1
        obs, _, _, _ = env.reset()
    if i % 100 == 0 and i != 0:
        print(f"Game {i}: {dones}")
        print(f"Win rate: {win_counter / num_games * 100:.2f}%")
        print(f"Games played: {num_games}")
        print(f"Win counter: {win_counter}")
