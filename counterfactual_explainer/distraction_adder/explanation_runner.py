import os
import sys
from typing import List, Tuple, Any, Optional

import torch
import streamlit as st

# Make sure we can import the explainer package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from explainer.explainer import Explainer  # type: ignore
from explainer.generate_single_explanation import templated_explanation  # type: ignore


class ExplanationRunner:
    """
    Thin wrapper around `Explainer` to run counterfactual explanations
    for the currently selected HOMER routine using manually edited
    scene-graph movements.
    """

    def __init__(self, navigator):
        """
        Args:
            navigator (HOMERNavigator): Provides dataset path, node classes,
                and current (day, routine) selection.
        """
        self.navigator = navigator
        self._explainer_cache = {}

    def _get_household_id(self) -> int:
        """
        Infer household_id from dataset_path, which is expected to end with
        'household{ID}' (e.g., '.../data/HOMER/household0').
        """
        if not self.navigator.dataset_path:
            raise ValueError("Navigator dataset_path is not set.")

        base = os.path.basename(self.navigator.dataset_path.rstrip("/"))
        # Expect patterns like 'household0', 'household1', etc.
        digits = "".join(ch for ch in base if ch.isdigit())
        if digits == "":
            raise ValueError(
                f"Could not infer household_id from dataset path '{self.navigator.dataset_path}'."
            )
        return int(digits)

    def _get_explainer(self) -> Explainer:
        """
        Lazily construct and cache an `Explainer` instance per (step_size, household_id)
        to avoid re-loading models repeatedly in a Streamlit session.
        """
        step_size = 1  # We run on a single time step in the distraction UI.
        household_id = self._get_household_id()
        key = (step_size, household_id)
        if key not in self._explainer_cache:
            self._explainer_cache[key] = Explainer(step_size=step_size, household_id=household_id)
        return self._explainer_cache[key]

    def _format_explanations(self, explanations: Any) -> str:
        """
        Render explanations using the templated explanation helper used
        in the main explainer code.
        """
        if explanations is None:
            return "No explanations generated."

        node_classes = getattr(self.navigator, "node_classes", None)
        if node_classes is None:
            return "Node classes not available; cannot render templated explanations."

        if not explanations:
            return "No movements were predicted."

        rendered: List[str] = []
        for item in explanations:
            try:
                rendered.append(templated_explanation(item, node_classes))
            except Exception as e:
                rendered.append(f"[Error rendering explanation: {e}]")

        return "\n\n".join(rendered)

    def run_with_manual_movements(
        self,
        manual_movements: List[Tuple[int, int, int]],
        time_target: Optional[int] = None,
    ):
        """
        Run the explainer on the currently selected (day, routine) using the
        manually edited movements as input to the movement tracker and to
        update the scene graph.

        This method must not create any Streamlit widgets; it relies on
        `HOMERNavigator.get_homer_dataset` having already been called in
        the UI (e.g., in the sidebar) to set the current selection.
        """
        dataset, day_idx, routine_idx = self.navigator.get_current_selection()
        _ = dataset  # not used directly; kept to mirror navigator API

        explainer = self._get_explainer()

        with st.spinner("Running explanation with manual movements..."):
            explanations = explainer.run_for_single_instance_with_manual_movements(
                day_no=day_idx,
                routine_no=routine_idx,
                manual_movements=manual_movements,
                time_target=time_target,
            )

        st.success("Explanation run complete.")
        formatted = self._format_explanations(explanations)
        st.text(formatted)


