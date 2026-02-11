import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
helpers_path = os.path.abspath(os.path.join(current_dir, '..', 'helpers'))
sys.path.append(helpers_path)
# sys.path.append('../helpers')
import os
import pickle
import json
from reader import RoutinesDataset
from encoders import TimeEncodingOptions

def get_dataset(data_dir, batch_size=32, train_days=30, time_encoding_type='sine_informed'):
    """
    Function to create and return a RoutinesDataset object.
    
    Parameters:
    - data_dir: str, path to the dataset folder
    - batch_size: int, desired batch size
    - train_days: int, max routines (None if not limited)
    - time_encoding_type: str, type of time encoding to use
    
    Returns:
    - RoutinesDataset object
    """
    
    # Load DATA_INFO
    with open(os.path.join(data_dir, 'processed', 'common_data.json')) as f:
        data_info = json.load(f)['info']

    # Initialize time encoder
    weekend_days = data_info.get('weeekend_days', None)
    time_options = TimeEncodingOptions(weekend_days)
    time_encoding = time_options(time_encoding_type)

    # Create the dataset object
    data = RoutinesDataset(
        data_path=os.path.join(data_dir, 'processed'),
        time_encoder=time_encoding,
        batch_size=batch_size,
        max_routines=(train_days, None)
    )
    
    return data



import streamlit as st
import os
import sys
import json
import networkx as nx
from pyvis.network import Network
import tempfile




# def build_graph(adj_matrix, node_labels):
#     G = nx.DiGraph()
#     num_nodes = len(node_labels)

#     # Add actual nodes
#     for i, label in enumerate(node_labels):
#         G.add_node(i, label=f"{i}: {label}")

#     # Add edges from the adjacency matrix
#     for i in range(num_nodes):
#         for j in range(num_nodes):
#             if adj_matrix[i, j] > 0:
#                 G.add_edge(j, i)

#     # Add root node "Home"
#     root_id = num_nodes  # new node index
#     G.add_node(root_id, label="Home", color="red")

#     # Find nodes with no parents (in-degree 0)
#     parentless_nodes = [n for n in range(num_nodes) if G.in_degree(n) == 0]

#     for n in parentless_nodes:
#         G.add_edge(root_id, n)

#     return G
def build_graph_with_diff(prev_adj, curr_adj, node_labels):
    G = nx.DiGraph()
    num_nodes = len(node_labels)

    # Add all nodes
    for i, label in enumerate(node_labels):
        G.add_node(i, label=f"{i}: {label}", color='lightblue')

    # Track edge states
    added_edges = []
    removed_edges = []
    unchanged_edges = []

    for i in range(num_nodes):
        for j in range(num_nodes):
            prev = prev_adj[i, j]
            curr = curr_adj[i, j]
            if prev == 1 and curr == 1:
                unchanged_edges.append((j, i))  # flipped for child → parent
            elif prev == 0 and curr == 1:
                added_edges.append((j, i))
            elif prev == 1 and curr == 0:
                removed_edges.append((j, i))

    # Draw unchanged edges in gray
    for src, tgt in unchanged_edges:
        G.add_edge(src, tgt, color="lightgray", title="unchanged")

    # Draw added edges in green
    for src, tgt in added_edges:
        G.add_edge(src, tgt, color="green", title="added")

    # Draw removed edges in red (optional: dashed)
    for src, tgt in removed_edges:
        G.add_edge(src, tgt, color="red", dashes=True, title="removed")

    # Add virtual root "Home"
    root_id = num_nodes
    G.add_node(root_id, label="Home", color="red")

    # Recompute in-degrees *after* adding all edges
    parentless = [n for n in range(num_nodes) if G.in_degree(n) == 0]
    for n in parentless:
        G.add_edge(root_id, n, color='blue', title="attached to root")

    return G

def visualize_graph(G):
    net = Network(height="600px", width="100%", directed=True)
    net.from_nx(G)
    net.set_options("""
    {
      "edges": {
        "arrows": {
          "to": {
            "enabled": true,
            "scaleFactor": 1.2
          }
        }
      },
      "layout": {
          "improvedLayout": true
        }  
    }
    """)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "graph.html")
        net.save_graph(path)
        html = open(path, "r", encoding="utf-8").read()
        st.components.v1.html(html, height=650, scrolling=True)


# -------------------- Streamlit UI --------------------

st.title("Scene Graph Visualizer from VirtualHome Dataset")

data_dir = st.text_input("Enter path to dataset directory (should contain 'processed' subfolder):")

if data_dir and os.path.exists(data_dir):
    try:
        dataset = get_dataset(data_dir)
        routine, _ = dataset.test_routines.get_routine(0)
        max_step = len(routine) - 1
        step = st.slider("Select time step", 0, max_step, 0, key="step_slider")

        prev_edge_tensor = routine[step][0]
        edge_tensor = routine[step][3]
        node_labels = dataset.node_classes

        st.success("Dataset loaded. Displaying first graph of first test day:")
        G = build_graph_with_diff(prev_edge_tensor.numpy() ,edge_tensor.numpy(), node_labels)
        visualize_graph(G)

    except Exception as e:
        st.error(f"Error loading dataset: {e}")
