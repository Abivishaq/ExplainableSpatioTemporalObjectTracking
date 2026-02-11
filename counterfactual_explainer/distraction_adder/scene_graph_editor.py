import streamlit as st
import torch

class SceneGraphEditor:
    def __init__(self, navigator):
        self.navigator = navigator  # Access to active and distractor routines

    def get_parent(self, edge_tensor, node_idx):
        """
        Finds the parent of the given node in a 108x108 edge tensor.
        Returns the index of the parent node or None if not found.
        """
        incoming_edges = edge_tensor[node_idx,:]  # parents → node_idx
        parent_indices = torch.nonzero(incoming_edges).squeeze()
        if parent_indices.numel() == 0:
            return None
        elif parent_indices.numel() > 1:
            # If multiple parents, return the first one (could be improved)
            st.warning(f"Node {node_idx} has multiple parents: {parent_indices.tolist()}. Returning the first one.")
            parent_indices = parent_indices[0]
        return parent_indices.item()

    def editor(self):
        st.header("✏️ Scene Graph Editor")

        if self.navigator.active_routine is None or self.navigator.distractor_routine is None:
            st.warning("No routine loaded. Please load one from the HOMER Navigator.")
            return

        node_labels = self.navigator.node_classes
        active_edges = self.navigator.active_routine['edges'].squeeze(0)
        distractor_edges = self.navigator.distractor_routine['edges'].squeeze(0)

        # Node selector
        node_idx = st.selectbox("Select a node", range(len(node_labels)),
                                format_func=lambda i: f"{i}: {node_labels[i]}")
        
        # Parent from active routine
        true_parent = self.get_parent(active_edges, node_idx)
        true_parent_str = (f"{true_parent}: {node_labels[true_parent]}"
                           if isinstance(true_parent, int) else str(true_parent))
        st.write("→ Parent in original graph:", true_parent_str)

        # Editable parent from distractor routine
        distractor_parent = self.get_parent(distractor_edges, node_idx)

        new_parent = st.selectbox(
            "Select new parent (distractor graph):",
            options=[None] + list(range(len(node_labels))),
            index=0 if distractor_parent is None else (distractor_parent + 1),  # shift for None
            format_func=lambda i: "None" if i is None else f"{i}: {node_labels[i]}"
        )

        # Update the distractor routine if selection changed
        if new_parent != distractor_parent:
            # Ensure movement history container exists
            if "manual_movements" not in st.session_state:
                st.session_state.manual_movements = []

            # Log this edit as a movement: (node_idx, prev_parent, new_parent)
            prev_p = distractor_parent
            # If there was no parent before, we encode as -1 to keep the triple structure.
            prev_parent_idx = -1 if prev_p is None else int(prev_p)
            new_parent_idx = -1 if new_parent is None else int(new_parent)
            st.session_state.manual_movements.append((int(node_idx), prev_parent_idx, new_parent_idx))

            # Reset all incoming edges to 0
            self.navigator.distractor_routine['edges'][0, node_idx,:] = 0
            if new_parent is not None:
                self.navigator.distractor_routine['edges'][0, node_idx, new_parent] = 1
            st.success(f"Updated parent of node {node_idx} to: {new_parent}")
