import streamlit as st
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from homer_navigator import HOMERNavigator
from scenegraph_viz import SceneGraphViz
from stot_model_runner import ModelRunner
from scene_graph_editor import SceneGraphEditor



# Initialize shared state
if "navigator" not in st.session_state:
    st.session_state.navigator = HOMERNavigator()
if "scene_viz" not in st.session_state:
    st.session_state.scene_viz = SceneGraphViz()
if "model_runner" not in st.session_state:
    st.session_state.model_runner = ModelRunner(st.session_state.navigator)
if "graph_editor" not in st.session_state:
    st.session_state.graph_editor = SceneGraphEditor(st.session_state.navigator)

# -----------------------
# Sidebar: Dataset Selector
# -----------------------
with st.sidebar:
    st.title("HOMER Navigator")
    dataset, day_idx, routine_idx = st.session_state.navigator.get_homer_dataset()
    if dataset:
        routine, _ = dataset.test_routines.get_routine(day_idx)

# -----------------------
# Main Area: Graph + Model Output
# -----------------------

routine = st.session_state.navigator.active_routine
distractor_routine = st.session_state.navigator.distractor_routine

if routine is not None:
    st.header("🔍 Scene Graph Visualization")
    st.session_state.scene_viz.show_scene_graph(
        routine, distractor_routine,
        st.session_state.navigator.node_classes
    )
    st.divider()
    st.header("🧩 Scene Graph Editor")
    st.session_state.graph_editor.editor()
    
    st.divider()
    st.header("🧠 STOT Model Inference")
    st.session_state.model_runner.run_model()
else:
    st.warning("Please select a routine from the HOMER Navigator (left panel).")