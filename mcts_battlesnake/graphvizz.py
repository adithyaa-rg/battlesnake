import graphviz
import os

def get_node_info_label(node_obj):
    """Generates a descriptive label for a node in the MCTS tree."""
    if not node_obj:
        return "None"

    q_val = node_obj.q_value
    visits = node_obj.visits
    avg_reward = q_val / visits if visits > 0 else 0
    
    label_parts = []

    # Display game-specific state if available (e.g., player position)
    if hasattr(node_obj.state, 'player_position'):
         label_parts.append(f"Pos: {node_obj.state.player_position}")
    elif hasattr(node_obj.state, '__str__'): # Fallback to string representation of state
        state_repr = str(node_obj.state)
        if len(state_repr) > 30: state_repr = state_repr[:27] + "..." # Truncate long states
        label_parts.append(f"State: {state_repr}")


    label_parts.append(f"Visits: {visits}")
    label_parts.append(f"Q-Value: {q_val:.2f}")
    label_parts.append(f"AvgReward: {avg_reward:.2f}")
    label_parts.append(f"Depth: {node_obj.depth}")


    if node_obj.state.is_terminal():
        result_text = "WIN" if node_obj.state.get_result() == 1 else "LOSE/DRAW"
        label_parts.append(f"TERMINAL ({result_text})")
    elif node_obj.is_fully_expanded():
        label_parts.append("Fully Expanded")
    else:
        label_parts.append(f"Untried Moves: {len(node_obj.untried_moves)}")
        
    return "\n".join(label_parts)

def get_node_color(node_obj):
    """Determines node color based on its properties."""
    if not node_obj.parent: # Root node
        return "lightblue"
    if node_obj.state.is_terminal():
        return "lightcoral" if node_obj.state.get_result() == 0 else "palegreen"
    if node_obj.visits > 0:
        avg_reward = node_obj.q_value / node_obj.visits
        if avg_reward > 0.7: return "mediumseagreen"
        if avg_reward > 0.4: return "lightgoldenrodyellow"
        if avg_reward < 0.1 and avg_reward !=0 : return "salmon" # distinguish from 0/0
        if avg_reward == 0 and node_obj.q_value == 0: return "whitesmoke" # Neutral or unpromising
    return "lightgrey" # Default for visited, non-terminal


def _add_nodes_edges_recursive(current_node, dot_graph, visited_node_ids):
    """
    Recursively adds nodes and edges to the Graphviz Digraph object.
    """
    current_node_id = str(id(current_node)) # Use object ID for unique node identifier in the graph

    if current_node_id in visited_node_ids:
        return # Avoid cycles or redundant processing if states can repeat (though MCTS builds a tree)
    visited_node_ids.add(current_node_id)

    node_label = get_node_info_label(current_node)
    node_color = get_node_color(current_node)
    
    dot_graph.node(current_node_id, label=node_label, shape="box", style="filled", fillcolor=node_color)

    for child_node in current_node.children:
        child_id = str(id(child_node))
        edge_label = str(child_node.state.last_action) if child_node.state.last_action else ""
        
        # Add child node and edge from current to child
        _add_nodes_edges_recursive(child_node, dot_graph, visited_node_ids)
        dot_graph.edge(current_node_id, child_id, label=edge_label)


def visualize_mcts_tree(root_mcts_node, output_filename="mcts_tree_visualization"):
    """
    Generates and saves a visualization of the MCTS tree.

    Args:
        root_mcts_node: The root Node object of the MCTS tree.
        output_filename: The base name for the output files (e.g., "mcts_tree"
                         will create "mcts_tree.gv" and "mcts_tree.png").
    """
    if not hasattr(root_mcts_node, 'state'): # Basic check for a Node-like object
        print("Error: visualize_mcts_tree expects a valid MCTS root Node.")
        return

    dot = graphviz.Digraph(comment='Monte Carlo Search Tree', format='png')
    dot.attr(rankdir='TB')  # Arrange tree from Top to Bottom
    dot.attr('node', fontsize='10')
    dot.attr('edge', fontsize='8')

    visited_ids = set()
    _add_nodes_edges_recursive(root_mcts_node, dot, visited_ids)

    try:
        # The render function saves the .gv source file and creates the image (e.g., .png)
        # cleanup=True will remove the source .gv file after rendering. Set to False to keep it.
        rendered_path = dot.render(output_filename, view=False, cleanup=True)
        print(f"MCTS tree visualization saved as: {rendered_path}")
    except graphviz.exceptions.ExecutableNotFound:
        gv_path = os.path.join(os.getcwd(), output_filename + ".gv")
        dot.save(gv_path)
        print("\n--- Graphviz Rendering Failed ---")
        print("Graphviz executable not found. Please ensure Graphviz is installed and in your system's PATH.")
        print(f"The Graphviz source file was saved as: {gv_path}")
        print("You can render it manually using a command like: dot -Tpng {gv_path} -o {output_filename}.png")
    except Exception as e:
        gv_path = os.path.join(os.getcwd(), output_filename + ".gv")
        dot.save(gv_path)
        print(f"\n--- An Error Occurred During Visualization ---")
        print(f"Error: {e}")
        print(f"The Graphviz source file was saved as: {gv_path}")
        print("You might be able to render it manually.")
