import streamlit as st

class MechanisticExplainer:
    def __init__(self, node_classes):
        """
        Initialize with the list of node class labels.

        Args:
            node_classes (List[str]): List mapping node indices to their class names.
        """
        self.node_classes = node_classes

    def get_mechanistic_explanation(self, pred_n_expl):
        """
        Renders Streamlit expandable blocks for each prediction and its explanation.

        Args:
            pred_n_expl (List[Dict]): List of prediction + explanation entries.
        """
        st.markdown("### Mechanistic Explanations")
        for idx, pred_info in enumerate(pred_n_expl):
            predicted_mov = pred_info.get('predicted_mov')
            explanation = pred_info.get('explanation')

            obj = self.node_classes[predicted_mov[0]]
            src = self.node_classes[predicted_mov[1]]
            dst = self.node_classes[predicted_mov[2]]
            label = f"{obj}: {src} → {dst}"

            with st.expander(label, expanded=False):
                self.render_single_explanation(predicted_mov, explanation)

    def render_single_explanation(self, predicted_mov, explanation):
        """
        Renders a detailed explanation for a single prediction.

        Args:
            predicted_mov (Tuple[int, int, int]): (object, previous_parent, new_parent)
            explanation (Dict): Explanation metadata
        """
        obj_name = self.node_classes[predicted_mov[0]]
        prev_name = self.node_classes[predicted_mov[1]]
        next_name = self.node_classes[predicted_mov[2]]

        st.markdown(f"**Prediction:** {obj_name} moved from *{prev_name}* to *{next_name}*")

        movement_perturbation = explanation.get("movement_perturbation", [])
        time_perturbation = explanation.get("time_perturbation", [])

        if movement_perturbation:
            st.markdown("**Movements that are important for this:**")
            for mov in movement_perturbation:
                pert_obj = self.node_classes[mov["object"]]
                pert_prev = self.node_classes[mov["previous_parent"]]
                pert_next = self.node_classes[mov["curr_parent"]]
                st.write(f"- {pert_obj} moved from {pert_prev} to {pert_next}")

        if time_perturbation:
            st.markdown("**Time steps where changing context influenced prediction:**")
            st.write(time_perturbation)
