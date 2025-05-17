from mcts_modified import *
from battlesnake_gym.snake_gym import BattlesnakeGym
import copy

if __name__ == "__main__":
    env = BattlesnakeGym(map_size = (11, 11), number_of_snakes = 4)
    env.reset()

    games_done = 0
    unlost = 0
    while True:
        json_state = env.get_json()
        # print(f"JSON state {i}: {json_state}")
        observation = env._get_state()
        env_copy = copy.copy(env)
        state1 = State(json_state=json_state, observation=observation)
        state2 = State(json_state=json_state, observation=observation, agent_id=1)
        state3 = State(json_state=json_state, observation=observation, agent_id=2)
        state4 = State(json_state=json_state, observation=observation, agent_id=3)
        # state = State(env_copy ,json_state=json_state, observation=observation)
        mcts1 = MCTS(
            state=state1,
            time_limit=0.25,
            max_depth=None,
            simulation_depth=None
        )
        # mcts2 = MCTS(
        #     state=state2,
        #     time_limit=0.25,
        #     max_depth=None,
        #     simulation_depth=None
        # )
        # mcts3 = MCTS(
        #     state=state3,
        #     time_limit=0.25,
        #     max_depth=None,
        #     simulation_depth=None
        # )
        # mcts4 = MCTS(
        #     state=state4,
        #     time_limit=0.25,
        #     max_depth=None,
        #     simulation_depth=None
        # )
        

        # Run MCTS
        time_start = time.time()
        best_action1 = mcts1.run()
        # best_action2 = mcts2.run()
        # best_action3 = mcts3.run()
        # best_action4 = mcts4.run()

        # print(f"Best action: {best_action}")
        # print(f"Time taken: {time.time() - time_start} seconds")

        # actions = [best_action1, best_action2, best_action3, best_action4]
        actions = [best_action1, random.randint(0, 3), random.randint(0, 3), random.randint(0, 3)]
        # for action in range(env.action_space.n):
        #     if action == best_action:
        #         actions[state.agent_id] = action
        #     else:
        #         # Randomly sample actions for other agents
        #         actions[state.agent_id] = env.action_space.sample()
        env.render()

        _, _, done, _, = env.step(actions)

        # time.sleep(0.5)

        if sum(done.values()) >= 3:
            unlost += 1 if not done[0] else 0
            env.reset()
            games_done += 1
            print(f"Game {games_done} done, unlost: {unlost}")
            if games_done > 25:
                break
            # print("Resetting environment...")
            # print()


        if mcts1.root.children:
            print("  Root children stats (simulation depth limited):")
            # Sort children by visits for clarity
            sorted_children = sorted(mcts1.root.children, key=lambda c: c.visits, reverse=True)
            for child in sorted_children:
                avg_reward = child.q_value / child.visits if child.visits > 0 else 0
                print(f"    Move {child.state.last_action}: Visits={child.visits}, AvgReward={avg_reward:.2f}, NodeDepth={child.depth}")
        else:
            print("  No children were explored from the root node.")

        # Visualize the MCTS tree
        if mcts1.root:
            # visualize_mcts_tree(mcts1.root, "mcts_run_1_tree")
            pass
        else:
            print("Cannot visualize: MCTS root node is not available.")

    print("Non Lost rate: ", unlost/games_done * 100)