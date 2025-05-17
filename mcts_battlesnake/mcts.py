import random
import math
import time
from graphvizz import visualize_mcts_tree

from state import State

class Node:
    """
    Represents a node in the Monte Carlo Search Tree.
    Each node stores the game state, parent and children nodes,
    Q-value (estimated value of the state), number of visits,
    depth in the tree, and a list of untried moves from this state.
    """
    def __init__(self, state, parent=None, depth=0):
        """
        Initializes a Node.
        Args:
            state: The game state represented by this node.
            parent: The parent node in the MCTS tree. Defaults to None for the root.
            depth: The depth of this node in the MCTS tree. Root is at depth 0.
        """
        self.state = state
        self.parent = parent
        self.children = []
        self.q_value = 0  # Sum of simulation results from this node
        self.visits = 0   # Number of times this node has been visited
        self.exploration_constant = 1.4  # Typically sqrt(2), balances exploration/exploitation
        self.untried_moves = state.get_possible_moves() # All possible moves from this state
        random.shuffle(self.untried_moves) # Shuffle to ensure randomness in expansion
        self.depth = depth

    def is_fully_expanded(self):
        """
        Checks if all possible moves from this node have been tried (i.e., expanded into child nodes).
        Returns:
            True if all moves have been tried, False otherwise.
        """
        return len(self.untried_moves) == 0

    def select(self):
        """
        Selects the child node with the highest UCB1 (Upper Confidence Bound 1) score.
        UCB1 = Q_child / N_child + C * sqrt(log(N_parent) / N_child)
        This balances exploiting known good moves and exploring less-visited paths.
        Returns:
            The child node with the highest UCB1 score.
        """
        if not self.children:
            return None

        return max(
            self.children,
            key=lambda child: (child.q_value / (child.visits + 1e-5)) +
                              self.exploration_constant * math.sqrt(
                                  math.log(self.visits + 1) / (child.visits + 1e-5)
                              )
        )

    def expand(self):
        """
        Expands the current node by creating one new child node from an untried move.
        The new child's depth will be self.depth + 1.
        Returns:
            The newly created child node, or None if there are no untried moves.
        """
        if self.untried_moves:
            move = self.untried_moves.pop()
            new_state = self.state.step(move)
            # Create child node with incremented depth
            child_node = Node(new_state, parent=self, depth=self.depth + 1)
            self.children.append(child_node)
            return child_node
        return None

    def backpropagate(self, result):
        """
        Updates the Q-value and visit count of this node and recursively for its parent.
        Args:
            result: The outcome of the simulation (e.g., 1 for win, 0 for loss/draw).
        """
        self.visits += 1
        self.q_value += result

        if self.parent:
            self.parent.backpropagate(result)

class MCTS:
    """
    Monte Carlo Tree Search algorithm.
    """
    def __init__(self, state, time_limit=0.25, max_depth=None, simulation_depth=None):
        """
        Initializes the MCTS algorithm.
        Args:
            state: The initial game state.
            time_limit: The time (in seconds) allowed for the MCTS algorithm to run.
            max_depth: The maximum depth the explicit search tree is allowed to reach.
                       If None, no depth limit is imposed beyond game termination.
            simulation_depth: The maximum number of steps for each simulation/rollout.
                              If None, simulations run until a terminal state.
        """
        self.root = Node(state, depth=0) # Root node is at depth 0
        self.time_limit = time_limit
        self.max_depth = max_depth
        self.simulation_depth = simulation_depth

    def run(self):
        """
        Runs the MCTS algorithm for the specified time limit.
        Ensures that a result is returned by the time limit, even if a simulation
        needs to be terminated prematurely.
        Returns:
            The best move found from the root state.
        """
        start_time = time.time()
        deadline = start_time + self.time_limit
        iterations = 0

        while time.time() < deadline:
            node = self.root

            # 1. Selection Phase
            # print("Selection Phase")
            while not node.state.is_terminal() and \
                  node.is_fully_expanded() and \
                  (self.max_depth is None or node.depth < self.max_depth):
                selected_node = node.select()
                if selected_node is None:
                    break
                node = selected_node
                # Check time during long selections (though unlikely for typical MCTS uses)
                if time.time() >= deadline:
                    break
            if time.time() >= deadline: break # Exit outer loop if time up after selection

            # 2. Expansion Phase
            # print("Expansion Phase")
            if not node.state.is_terminal() and \
               (self.max_depth is None or node.depth < self.max_depth):
                expanded_node = node.expand()
                if expanded_node:
                    node = expanded_node # Move to the new child for simulation
            if time.time() >= deadline: break # Exit outer loop if time up after expansion


            # 3. Simulation Phase (Rollout)
            # print("Simulation Phase")
            # Pass the deadline to the simulation function
            result = self.simulate(node.state, deadline)

            # If time ran out *during* simulation, 'result' is from the interrupted state.
            # The simulation itself respected the deadline.

            # 4. Backpropagation Phase
            # print("Backpropagation Phase")
            node.backpropagate(result)
            iterations +=1

            # print(f"Iteration {iterations}: Node depth {node.depth}, Result: {result}, Time elapsed: {time.time() - start_time:.2f}s, Deadline: {deadline-start_time:.2f}s")

            # Check time one last time before attempting another iteration
            if time.time() >= deadline:
                print(f"Time limit reached after iteration {iterations}. Exiting.")
                break

        # print(f"MCTS ran for {iterations} iterations within {self.time_limit}s.")
        best_action = self.get_best_move()
        print(f"Best action: {best_action} selected after {iterations} iterations.")
        return best_action

    def simulate(self, state, deadline):
        """
        Simulates a random playout from the given state until a terminal state,
        simulation_depth, or the deadline is reached.
        Args:
            state: The state from which to start the simulation.
            deadline: The absolute time by which the simulation must conclude.
        Returns:
            The result of the game (e.g., 1 for win, 0 for loss/draw).
        """
        current_state = state

        sim_steps_limit = float('inf')
        if self.simulation_depth is not None:
            sim_steps_limit = self.simulation_depth

        depth = 0
        while not current_state.is_terminal() and \
              depth < sim_steps_limit and \
              time.time() < deadline: # Critical check for overall time limit
            depth += 1
            possible_moves = current_state.get_possible_moves()
            if not possible_moves: # Should ideally not happen if is_terminal is false
                return 0 # Default result for a stuck state
            move = random.choice(possible_moves)
            current_state = current_state.step(move)

        # If loop terminated due to deadline, current_state might not be terminal or at sim_steps_limit
        return current_state.get_result()

    def get_best_move(self):
        """
        Selects the best move from the root node's children, typically the one most visited.
        Returns:
            The best move (action) from the root state, or None if no moves are possible/explored.
        """
        print("Selecting best move from root node...")
        print(f"Root node visits: {self.root.visits}, Children count: {len(self.root.children)}")
        if not self.root.children:
            print("No children explored from the root. Cannot determine best move.")
            return None
        
        best_child = max(self.root.children, key=lambda c: c.q_value / (c.visits + 1e-5))
        print(f"Best child selected: {best_child.state}, Visits: {best_child.visits}, AvgReward: {best_child.q_value / best_child.visits if best_child.visits > 0 else 0:.2f}, Last Move: {best_child.state.last_action}")
        return best_child.state.last_action if best_child.state.last_action is not None else None

# Test the MCTS implementation on a simple game (commented out as you're using Battlesnake)
class SimpleGameState:
    """
    A very simple game state for testing MCTS.
    """
    def __init__(self, player_position, goal_position):
        self.player_position = player_position
        self.goal_position = goal_position
        self.last_action = None # To track the move that led to this state

    def get_possible_moves(self):
        # Possible moves: right, down, left, up
        return [(1, 0), (0, 1), (-1, 0), (0, -1)]

    def step(self, move):
        new_position = (self.player_position[0] + move[0], self.player_position[1] + move[1])
        # Create a new state object for the new position
        new_state = SimpleGameState(new_position, self.goal_position)
        new_state.last_move = move # Store the move made
        return new_state

    def is_terminal(self):
        return self.player_position == self.goal_position

    def get_result(self):
        # Returns 1 if the player reached the goal (win), 0 otherwise (loss or ongoing)
        return 1 if self.is_terminal() else 0

    def __repr__(self):
        return f"Player: {self.player_position}, Goal: {self.goal_position}, Last Move: {self.last_action}"


# Example usage
if __name__ == "__main__":
    initial_state = SimpleGameState((0, 0), (2, 2))
    print(f"Initial state: {initial_state.player_position}, Goal: {initial_state.goal_position}")

    print("-" * 30)
    print("Running MCTS with simulation depth limit and overall time limit:")
    # Example: time_limit=0.1s, tree max_depth=5, simulation_depth=10 steps
    # The simulation will run for at most 10 steps, unless game ends sooner or time_limit is hit.
    # The tree itself will not grow beyond depth 5.
    mcts_agent_sim_depth_limited = MCTS(
        initial_state,
        time_limit=0.1,      # Overall time for MCTS to run
        max_depth=5,         # Max depth of the explicit MCTS tree
        simulation_depth=None  # Max steps per simulation/rollout
    )
    best_move_sim_depth = mcts_agent_sim_depth_limited.run()
    print(f"Best move (simulation depth limited, time limited): {best_move_sim_depth}")

    if mcts_agent_sim_depth_limited.root.children:
        print("  Root children stats (simulation depth limited):")
        # Sort children by visits for clarity
        sorted_children = sorted(mcts_agent_sim_depth_limited.root.children, key=lambda c: c.visits, reverse=True)
        for child in sorted_children:
            avg_reward = child.q_value / child.visits if child.visits > 0 else 0
            print(f"    Move {child.state.last_move}: Visits={child.visits}, AvgReward={avg_reward:.2f}, NodeDepth={child.depth}")
    else:
        print("  No children were explored from the root node.")

    print("-" * 30)
    if mcts_agent_sim_depth_limited.root:
        visualize_mcts_tree(mcts_agent_sim_depth_limited.root, "mcts_run_1_tree")
    else:
        print("Cannot visualize: MCTS root node is not available.")

    print("Running MCTS with a very short time limit to test termination:")
    mcts_agent_short_time = MCTS(
        initial_state,
        time_limit=0.001,     # Very short time limit
        max_depth=5,
        simulation_depth=5
    )
    best_move_short_time = mcts_agent_short_time.run()
    print(f"Best move (very short time limit): {best_move_short_time}")
    if mcts_agent_short_time.root.children:
        print("  Root children stats (very short time limit):")
        sorted_children = sorted(mcts_agent_short_time.root.children, key=lambda c: c.visits, reverse=True)
        for child in sorted_children:
            avg_reward = child.q_value / child.visits if child.visits > 0 else 0
            print(f"    Move {child.state.last_move}: Visits={child.visits}, AvgReward={avg_reward:.2f}, NodeDepth={child.depth}")
    else:
        print("  No children were explored from the root node (very short time limit).")

    print("Attempting to visualize the MCTS tree from the short run...")
    if mcts_agent_short_time.root:
        visualize_mcts_tree(mcts_agent_short_time.root, "mcts_short_run_tree")
    else:
        print("Cannot visualize: MCTS root node is not available for short run.")
    # --- End of visualization section ---
