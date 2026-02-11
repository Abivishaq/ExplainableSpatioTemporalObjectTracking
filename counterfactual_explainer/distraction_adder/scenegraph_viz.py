import os
import sys
pth = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(pth)
from explantion_visualizer.context_visualizer import ContextVisualizer
import networkx as nx

class SceneGraphViz(ContextVisualizer):
    def __init__(self):
    
        # Initialize parent class with no log handler
        super().__init__(log_handler=None)
    
    def build_graph_with_diff(self, prev_adj, curr_adj, mod_adj, node_labels):
        G = nx.DiGraph()
        num_nodes = len(node_labels)

        # Add nodes
        for i, label in enumerate(node_labels):
            G.add_node(i, label=f"{i}: {label}", color='lightblue')

        # Edge categorization
        added_edges, removed_edges, unchanged_edges, mod_edges, unchanged_edges_dashed = [], [], [], [], []

        for i in range(num_nodes):
            unchanged_added = False
            unchanged_added_index = None
            mod_added = False
            for j in range(num_nodes):
                prev = prev_adj[i, j]
                curr = curr_adj[i, j]
                mod = mod_adj[i, j] 
                if prev == 1 and curr == 1:
                    if mod_added:
                        unchanged_edges_dashed.append((j, i))
                    else:
                        unchanged_edges.append((j, i))  # child → parent
                        unchanged_added = True
                        unchanged_added_index = (j, i)
                elif prev == 0 and curr == 1:
                    added_edges.append((j, i))
                elif prev == 1 and curr == 0:
                    removed_edges.append((j, i))
                elif prev == 0 and mod == 1:
                    mod_edges.append((j, i))
                    mod_added = True
                    if unchanged_added:
                        unchanged_edges.remove(unchanged_added_index)
                        unchanged_edges_dashed.append(unchanged_added_index)
                        

        for src, tgt in unchanged_edges:
            G.add_edge(src, tgt, color="lightgray", title="unchanged")
        for src, tgt in added_edges:
            G.add_edge(src, tgt, color="green", title="added")
        for src, tgt in removed_edges:
            G.add_edge(src, tgt, color="red", dashes=True, title="removed")
        for src, tgt in mod_edges:
            G.add_edge(src, tgt, color="orange", title="modified")
        for src, tgt in unchanged_edges_dashed:
            G.add_edge(src, tgt, color="lightgray", dashes=True, title="unchanged")

        # Add virtual root "Home"
        root_id = num_nodes
        G.add_node(root_id, label="Home", color="red")
        parentless = [n for n in range(num_nodes) if G.in_degree(n) == 0]
        for n in parentless:
            G.add_edge(root_id, n, color='blue', title="attached to root")

        return G
        

    def show_scene_graph(self,routine, distractor_routine, node_classes):
        prev_edges = routine['edges'].squeeze(0).detach().cpu().numpy()
        next_edges = routine['y_edges'].squeeze(0).detach().cpu().numpy()
        distractor_routine = distractor_routine['edges'].squeeze(0).detach().cpu().numpy()

        G = self.build_graph_with_diff(
            prev_edges,
            next_edges,
            distractor_routine,
            node_classes
        )
        self.visualize_graph(G)