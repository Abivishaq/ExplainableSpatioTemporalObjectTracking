import streamlit as st
import os
import tempfile
import networkx as nx
from pyvis.network import Network
import numpy as np
import traceback
import time

class ContextVisualizer:
    def __init__(self, log_handler):
        self.log_handler = log_handler  # instance of LogHandler

    def build_graph_with_diff(self, prev_adj, curr_adj, node_labels):
        G = nx.DiGraph()
        num_nodes = len(node_labels)

        # Add nodes
        for i, label in enumerate(node_labels):
            G.add_node(i, label=f"{i}: {label}", color='lightblue')

        # Edge categorization
        added_edges, removed_edges, unchanged_edges = [], [], []

        for i in range(num_nodes):
            for j in range(num_nodes):
                prev = prev_adj[i, j]
                curr = curr_adj[i, j]
                if prev == 1 and curr == 1:
                    unchanged_edges.append((j, i))  # child → parent
                elif prev == 0 and curr == 1:
                    added_edges.append((j, i))
                elif prev == 1 and curr == 0:
                    removed_edges.append((j, i))

        for src, tgt in unchanged_edges:
            G.add_edge(src, tgt, color="lightgray", title="unchanged")
        for src, tgt in added_edges:
            G.add_edge(src, tgt, color="green", title="added")
        for src, tgt in removed_edges:
            G.add_edge(src, tgt, color="red", dashes=True, title="removed")

        # Add virtual root "Home"
        root_id = num_nodes
        G.add_node(root_id, label="Home", color="red")
        parentless = [n for n in range(num_nodes) if G.in_degree(n) == 0]
        for n in parentless:
            G.add_edge(root_id, n, color='blue', title="attached to root")

        return G

    def visualize_graph(self, G):
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
            print("tmp_dir:", tmpdir)
            path = os.path.join(tmpdir, "graph.html")
            net.save_graph(path)    
            
            html = open(path, "r", encoding="utf-8").read()
            st.components.v1.html(html, height=650, scrolling=True)

    def show_scene_graph(self):
        routine_data = self.log_handler.active_log_data
        routine_no = self.log_handler.active_routine_no
        node_labels = self.log_handler.node_classes

        if routine_data is None or routine_no is None:
            st.warning("No routine data available for visualization.")
            return

        try:
            context, _ = routine_data[routine_no]
            time_tensor, prev_edge_tensor, edge_tensor = context  # edge_tensor is the current edge state
            

            # Convert to numpy
            prev_edge_np = prev_edge_tensor.squeeze(0).detach().cpu().numpy()
            edge_np = edge_tensor.squeeze(0).detach().cpu().numpy()

            st.subheader("Scene Graph")
            st.caption(f"Visualizing changes from routine #{routine_no - 1} → {routine_no}")
            G = self.build_graph_with_diff(prev_edge_np, edge_np, node_labels)
            self.visualize_graph(G)
        except Exception as e:
            st.error(f"Error visualizing scene graph: {e}")
            traceback.print_exc()

