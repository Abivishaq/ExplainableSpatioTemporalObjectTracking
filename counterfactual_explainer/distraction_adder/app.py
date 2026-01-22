import streamlit as st
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from homer_navigator import HOMERNavigator
from scenegraph_viz import SceneGraphViz
from stot_model_runner import ModelRunner
from scene_graph_editor import SceneGraphEditor
from explanation_runner import ExplanationRunner



# Initialize shared state
if "navigator" not in st.session_state:
    st.session_state.navigator = HOMERNavigator()
if "scene_viz" not in st.session_state:
    st.session_state.scene_viz = SceneGraphViz()
if "model_runner" not in st.session_state:
    st.session_state.model_runner = ModelRunner(st.session_state.navigator)
if "graph_editor" not in st.session_state:
    st.session_state.graph_editor = SceneGraphEditor(st.session_state.navigator)
if "explanation_runner" not in st.session_state:
    st.session_state.explanation_runner = ExplanationRunner(st.session_state.navigator)
if "manual_movements" not in st.session_state:
    st.session_state.manual_movements = []

# -----------------------
# Sidebar: Dataset Selector
# -----------------------
with st.sidebar:
    st.title("HOMER Navigator")
    dataset, day_idx, routine_idx = st.session_state.navigator.get_homer_dataset()
    if dataset:
        routine, _ = dataset.test_routines.get_routine(day_idx)

    # -----------------------
    # Manual Movements Summary + Reset
    # -----------------------
    st.subheader("Manual Movements")
    if st.session_state.manual_movements:
        node_labels = getattr(st.session_state.navigator, "node_classes", None)
        for (obj, prev_p, new_p) in st.session_state.manual_movements:
            if node_labels and 0 <= obj < len(node_labels):
                obj_str = f"{obj}: {node_labels[obj]}"
            else:
                obj_str = str(obj)

            def parent_str(idx):
                if idx is None or idx < 0:
                    return "None"
                if node_labels and 0 <= idx < len(node_labels):
                    return f"{idx}: {node_labels[idx]}"
                return str(idx)

            st.write(f"{obj_str} | {parent_str(prev_p)} → {parent_str(new_p)}")
    else:
        st.write("No manual movements yet.")

    if st.button("Reset Manual Movements"):
        st.session_state.manual_movements = []
        # Also reset the distractor routine to match the active routine
        if st.session_state.navigator.active_routine is not None:
            st.session_state.navigator.distractor_routine = (
                st.session_state.navigator.copy_active_routine()
            )

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

    st.divider()
    st.header("📘 Explanations (Counterfactual XAI)")
    # Default: use the current time of the active routine as the target.
    current_time = None
    if "time" in routine and isinstance(routine["time"], type(st.session_state.navigator.active_routine["time"])):
        try:
            current_time = int(routine["time"].item())
        except Exception:
            current_time = None

    use_custom_time = st.checkbox("Specify custom time target", value=False)
    custom_time_val = None
    if use_custom_time:
        # Use current_time as a starting point if available; otherwise fall back to a typical HOMER range start.
        default_time = current_time if current_time is not None else 70
        custom_time_val = st.number_input(
            "Custom time target",
            min_value=0,
            max_value=1600,
            value=int(default_time),
            step=10,
        )

    time_target = int(custom_time_val) if use_custom_time and custom_time_val is not None else current_time

    if st.button("Run Explanations for Manual Movements"):
        st.session_state.explanation_runner.run_with_manual_movements(
            manual_movements=st.session_state.manual_movements,
            time_target=time_target,
        )
else:
    st.warning("Please select a routine from the HOMER Navigator (left panel).")