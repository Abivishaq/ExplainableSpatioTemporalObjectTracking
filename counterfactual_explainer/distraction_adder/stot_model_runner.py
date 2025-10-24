import streamlit as st
import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from homer_navigator import HOMERNavigator

from explainer.stot_model import STOTModel  # adjust if placed elsewhere

class ModelRunner:
    def __init__(self, homer_navigator: HOMERNavigator):
        self.homer_navigator = homer_navigator
        self.model = STOTModel(step_size=1)
    
    def predicted_mov_to_str(self, predicted_movs):
        pred_str = ""
        for pred_mov in predicted_movs:
            node, old_parent, new_parent = pred_mov
            pred_str += f"{self.homer_navigator.node_classes[node]}: {self.homer_navigator.node_classes[old_parent]} -> {self.homer_navigator.node_classes[new_parent]}\n"
        return pred_str


    def run_model(self):
        """
        Streamlit interface to run model on the active routine from HOMERNavigator.
        """
        st.header("STOT Model Runner")

        if self.homer_navigator.active_routine is None:
            st.warning("Please load a routine first from the HOMER Navigator.")
            return

        if st.button("Run Model Inference"):
            routine = self.homer_navigator.active_routine
            distractor_routine = self.homer_navigator.distractor_routine
            try:
                input_tensor, output_tensor, gt_tensor, edge_probs = self.model.infer([routine])

                dist_inp_tensor, dist_out_tensor, dist_gt_tensor, dist_edge_probs = self.model.infer([distractor_routine])

                st.success("Inference complete.")
                
                predicted_movs = self.model.get_predicted_movements(input_tensor, output_tensor)
                st.subheader("Predicted Movements:")
                predicted_movs_str = self.predicted_mov_to_str(predicted_movs)
                st.text(predicted_movs_str)

                dist_predicted_movs = self.model.get_predicted_movements(dist_inp_tensor, dist_out_tensor)
                st.subheader("Distractor Predicted Movements:")
                dist_predicted_movs_str = self.predicted_mov_to_str(dist_predicted_movs)
                st.text(dist_predicted_movs_str)

                # st.subheader("Input Tensor (agent location):")
                # st.write(input_tensor.tolist())

                # st.subheader("Predicted Output:")
                # st.write(output_tensor.tolist())

                # st.subheader("Ground Truth:")
                # st.write(gt_tensor.tolist())

                # st.subheader("Edge Probabilities:")
                # st.write(edge_probs.squeeze(0).detach().cpu().numpy())

            except Exception as e:
                st.error(f"Error during model inference: {e}")
