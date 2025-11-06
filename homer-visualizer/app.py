import streamlit as st
import os

from graph_viz.data_loader import get_dataset, get_routine
from graph_viz.graph_builder import build_graph_with_diff
from graph_viz.visualizer import visualize_graph

import numpy as np

st.set_page_config(page_title="VirtualHome Scene Graph Viewer", layout="wide")
st.title("🔍 Scene Graph Visualizer – VirtualHome Dataset")

# -- Input: Dataset directory path
data_dir = st.text_input(
    "Enter path to dataset directory (must contain a 'processed' subfolder):",
    placeholder="/path/to/VirtualHome/data"
)

# -- Load dataset
if data_dir and os.path.exists(data_dir):
    try:
        dataset = get_dataset(data_dir)

        # Get the first routine (e.g. day)
        num_days = dataset.test_routines.num_routines()
        day = st.slider("Select day", 0, num_days - 1, 0, key="day_slider")

        routine = get_routine(dataset, routine_idx=day)
        
        # chande detection:
        change_steps = []
        for i, step in enumerate(routine):
            prev = step[0].numpy()
            curr = step[3].numpy()
            if not np.array_equal(prev, curr):
                change_steps.append(i)

        
        max_step = len(routine) - 1
        
        
        # Optional: quick jump using selectbox
        jump = st.selectbox("Jump to a changed step:", change_steps, index=0 if step not in change_steps else change_steps.index(step))

        # -- Slider to choose time step
        step = st.slider("Select time step", 0, max_step, jump, key="step_slider")
        st.markdown("### 🟢 Steps with graph changes:")
        st.write(change_steps)


        # -- Extract edge tensors
        prev_edges = routine[step][0].numpy()
        curr_edges = routine[step][3].numpy()
        node_labels = dataset.node_classes

        # -- Build and visualize graph
        st.success(f"Displaying graph for Day {day}, Step {step}")
        G = build_graph_with_diff(prev_edges, curr_edges, node_labels)
        visualize_graph(G)

        # Optional: timestamp / debug info
        time = routine[step][6].item()
        st.caption(f"⏱️ Encoded time: {time:.2f} minutes")

    except Exception as e:
        st.error(f"🚨 Error loading or visualizing dataset: {e}")
else:
    st.info("Please enter a valid dataset path.")
