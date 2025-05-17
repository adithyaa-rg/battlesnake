from battlesnake_gym.snake_gym import BattlesnakeGym
import numpy as np

class State:
    def __init__(self,
                 json_state: dict,
                 observation,
                 last_action: int = None,
                 reward: float = 0.0,
                 done: bool = False,
                 agent_id: int = 0):
        """
        json_state : full snapshot of the BattlesnakeGym at this node
        observation: your snake’s obs (i.e. state[agent_id])
        reward, done, agent_id as before
        """
        self.json_state = json_state
        self.observation = observation
        self.reward = reward
        self.done = done
        self.agent_id = agent_id
        self.last_action = last_action

    def _build_env(self):
        """Make a fresh env from the stored JSON."""
        return BattlesnakeGym(map_size = (11, 11), number_of_snakes = 4, initial_game_state=self.json_state)

    def step(self, action):
        """
        Expand one child:
        1) Reconstruct the env at *this* node’s snapshot
        2) Apply the action + random opponents
        3) Grab the next JSON and return a new State
        """
        env = self._build_env()
        # --- reset the env to this node's JSON state ---
        env.reset()

        n = env.action_space.n
        full_actions = []
        for i in range(n):
            if i == self.agent_id:
                full_actions.append(action)
            else:
                full_actions.append(env.action_space[i].sample())

        # --- step and collect next-state data ---
        next_states, rewards, dones, infos = env.step(full_actions)

        # --- snapshot the new full JSON for the child ---
        next_json = env.get_json()

        return State(
            json_state=next_json,
            observation=next_states,
            last_action=action,
            reward=rewards[self.agent_id],
            done=dones[self.agent_id],
            agent_id=self.agent_id
        )

    def render(self):
        env = self._build_env()
        env.render()

    def is_terminal(self):
        return bool(self.done)

    def get_result(self):
        return self.reward

    def get_possible_moves(self):
        env = self._build_env()
        return list(range(env.action_space[self.agent_id].n))

    def get_valid_actions(self):
        """
        If your gym supports action masks:
        """
        env = self._build_env()
        try:
            return env.get_valid_actions(self.agent_id)
        except AttributeError:
            return list(range(env.action_space[self.agent_id].n))

    def __repr__(self):
        return (f"<State reward={self.reward:.3f} done={self.done} "
                f"agent={self.agent_id}>")