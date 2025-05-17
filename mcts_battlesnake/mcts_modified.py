import random
import math
import time
from graphvizz import visualize_mcts_tree  # Import is present, keep it.
import concurrent.futures
import copy
import numpy as np
from collections import deque

from state_modified import State
from battlesnake_gym.snake_gym import BattlesnakeGym

class Node:
    """Represents a node in the Monte Carlo Search Tree."""
    def __init__(self, state, parent=None, depth=0):
        self.state = state
        self.parent = parent
        self.children = []
        self.q_value = 0.0
        self.visits = 0
        self.exploration_constant = 2
        self.untried_moves = deque(state.get_possible_moves())
        random.shuffle(self.untried_moves)
        self.depth = depth
        self.last_action = None # Store the action that led to this node

    def is_fully_expanded(self):
        return not self.untried_moves

    def expand(self):
        if self.untried_moves:
            move = self.untried_moves.popleft()
            new_state = self.state.step(move)
            child_node = Node(new_state, parent=self, depth=self.depth + 1)
            child_node.last_action = move # Store the move
            self.children.append(child_node)
            return child_node
        return None

    def backpropagate(self, result):
        self.visits += 1
        self.q_value += (result - self.q_value) / self.visits
        if self.parent:
            self.parent.backpropagate(result)

    def select(self):
        if not self.children:
            return None
        parent_visits = self.parent.visits if self.parent else 1
        log_parent_visits = math.log(parent_visits + 1)
        return max(
            self.children,
            key=lambda child: (child.q_value / (child.visits + 1e-6)) +
                            self.exploration_constant * math.sqrt(log_parent_visits / (child.visits + 1e-6))
        )

class MCTS:
    """Monte Carlo Tree Search algorithm."""
    def __init__(self, state, time_limit=0.25, max_depth=None, simulation_depth=None, num_threads=4):
        self.root = Node(state, depth=0)
        self.agent_index = self.root.state.agent_id
        self.time_limit = time_limit
        self.max_depth = max_depth
        self.simulation_depth = simulation_depth
        self.num_threads = num_threads

    def run(self):
        start_time = time.time()
        deadline = start_time + self.time_limit
        iterations = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = []
            while time.time() < deadline:
                node = self._select_node()
                if node:
                    futures.append(executor.submit(self._simulate_and_backpropagate, node, deadline))
                    iterations += 1
                if len(futures) > self.num_threads * 2:
                    for future in concurrent.futures.as_completed(futures):
                        future.result()
                    futures = []

            for future in futures:
                future.result()

        best_action = self._get_best_move()
        return best_action

    def _select_node(self):
        node = self.root
        while not node.state.is_terminal() and node.is_fully_expanded() and \
              (self.max_depth is None or node.depth < self.max_depth):
            next_node = node.select()
            if not next_node:
                return None
            node = next_node
        if not node.is_fully_expanded() and (self.max_depth is None or node.depth < self.max_depth) and not node.state.is_terminal():
            return node.expand()
        return node

    def _simulate_and_backpropagate(self, node, deadline):
        result = self._simulate(node.state, deadline)
        node.backpropagate(result)

    def _simulate(self, start_state, deadline):
        temp_env = BattlesnakeGym(map_size=(11, 11), number_of_snakes=4, initial_game_state=start_state.json_state)
        _, _, env_dones, _ = temp_env.reset()
        rewards = np.zeros(temp_env.action_space.n)
        dones_array = np.array(list(env_dones.values()))

        sim_steps_limit = float('inf') if self.simulation_depth is None else self.simulation_depth
        depth = 0

        while not np.all(dones_array) and depth < sim_steps_limit and time.time() < deadline:
            depth += 1
            actions = [temp_env.action_space[0].sample()] + [
                temp_env.action_space[i].sample() for i in range(1, temp_env.action_space.n)
            ]
            _, step_rewards, step_dones, _ = temp_env.step(actions)
            for i, reward in step_rewards.items():
                rewards[i] += reward
            dones_array = np.array(list(step_dones.values()))
            if step_dones.get(start_state.agent_id, False):
                return rewards[start_state.agent_id]

        return rewards[start_state.agent_id]

    def _get_best_move(self):
        if not self.root.children:
            print("No children explored from the root.")
            return None
        best_child = max(self.root.children, key=lambda c: c.visits)
        return best_child.state.last_action

    def update_root(self, new_state):
        """
        Updates the root of the MCTS tree to a new state, reusing any relevant
        subtree if possible.
        """
        if self.root.children:
            for child in self.root.children:
                if child.state == new_state:
                    self.root = child
                    self.root.parent = None  # Detach from old parent
                    return  # Found matching child, reuse
        # If no matching child, create a new root
        self.root = Node(new_state, depth=0)
        
    def clear_tree(self):
        """Clears the entire tree, forcing a fresh MCTS search."""
        self.root = Node(self.root.state, depth=0) # keep the same state, but reset the tree

def play_battlesnake_mcts(num_games=25):
    env = BattlesnakeGym(map_size=(11, 11), number_of_snakes=4)
    games_done = 0
    unlost = 0

    while games_done < num_games:
        env.reset()
        json_state = env.get_json()
        observation = env._get_state()
        state1 = State(json_state=json_state, observation=observation)
        mcts1 = MCTS(
            state=state1,
            time_limit=0.25,
            max_depth=None,
            simulation_depth=50, # Added a simulation depth limit
            num_threads=4
        )

        game_steps = 0
        while True:
            game_steps += 1
            json_state = env.get_json()
            observation = env._get_state()
            state1 = State(json_state=json_state, observation=observation)

            # Update the MCTS root, reusing the tree if possible
            mcts1.update_root(state1)

            best_action1 = mcts1.run()

            actions = [best_action1, random.randint(0, 3), random.randint(0, 3), random.randint(0, 3)]
            env.render()
            _, _, done, _ = env.step(actions)

            if sum(done.values()) >= 3:
                unlost += 1 if not done[0] else 0
                env.reset()  # Reset the *environment*
                games_done += 1
                print(f"Game {games_done} done, win: {unlost}, Steps: {game_steps}")
                break

        print("Win Rate:", (unlost / games_done) * 100)

if __name__ == "__main__":
    play_battlesnake_mcts(num_games=25)