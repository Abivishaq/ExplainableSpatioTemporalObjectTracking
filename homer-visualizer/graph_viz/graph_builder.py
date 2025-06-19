import networkx as nx
import numpy as np
from typing import List


def build_graph_with_diff(
    prev_adj: np.ndarray,
    curr_adj: np.ndarray,
    node_labels: List[str],
    include_virtual_root: bool = True
) -> nx.DiGraph:
    """
    Build a directed graph from previous and current adjacency matrices, highlighting changes.

    Parameters:
        prev_adj (np.ndarray): Previous timestep adjacency matrix (shape: [N, N]).
        curr_adj (np.ndarray): Current timestep adjacency matrix (shape: [N, N]).
        node_labels (list[str]): Labels for each node.
        include_virtual_root (bool): Whether to add a virtual root node ("Home").

    Returns:
        nx.DiGraph: Graph with visual diff (added, removed, unchanged edges).
    """
    G = nx.DiGraph()
    num_nodes = len(node_labels)

    # Add nodes
    for i, label in enumerate(node_labels):
        G.add_node(i, label=f"{i}: {label}", color="lightblue")

    # Compare adjacency matrices
    added_edges = []
    removed_edges = []
    unchanged_edges = []

    for i in range(num_nodes):
        for j in range(num_nodes):
            prev = prev_adj[i, j]
            curr = curr_adj[i, j]
            edge = (j, i)  # flipped for child → parent

            if prev and curr:
                unchanged_edges.append(edge)
            elif not prev and curr:
                added_edges.append(edge)
            elif prev and not curr:
                removed_edges.append(edge)

    # Add edges with color coding
    for src, tgt in unchanged_edges:
        G.add_edge(src, tgt, color="lightgray", title="unchanged")
    for src, tgt in added_edges:
        G.add_edge(src, tgt, color="green", title="added")
    for src, tgt in removed_edges:
        G.add_edge(src, tgt, color="red", dashes=True, title="removed")

    # Optional: add virtual root node
    if include_virtual_root:
        root_id = num_nodes
        G.add_node(root_id, label="Home", color="red")
        parentless = [n for n in range(num_nodes) if G.in_degree(n) == 0]
        for n in parentless:
            G.add_edge(root_id, n, color="blue", title="attached to root")

    return G
