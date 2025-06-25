import streamlit as st
import sys
import os
import traceback

# Ensure local module imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from log_handler import LogHandler
from context_visualizer import ContextVisualizer
from mechanistic_explainer import MechanisticExplainer  

def main():
    st.set_page_config(page_title="Scene Graph Log Explorer", layout="wide")
    st.title("Scene Graph Counterfactual Explorer")

    # --- Load and manage logs ---
    log_handler = LogHandler()
    log_handler.get_log_folder()
    
    result = log_handler.traverser()
    if result:
        routine_no, routine_data = result

        # --- Context Graph ---
        st.subheader(f"Routine #{routine_no}: Scene Graph & Context")
        context_viz = ContextVisualizer(log_handler)
        context_viz.show_scene_graph()

        # --- Raw Data (optional) ---
        with st.expander("Raw Context Data (time, edge_prev, edge_new)", expanded=False):
            st.write(routine_data[0])
            st.write("No of elements:", len(routine_data[0]))

        # --- Mechanistic Explanation ---
        st.subheader("Mechanistic Explanations")
        explainer = MechanisticExplainer(log_handler.node_classes)
        explainer.get_mechanistic_explanation(routine_data[1]) 

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        traceback.print_exc()
