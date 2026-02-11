import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from counterfactual_explainer.explainer.explainer import Explainer
from counterfactual_explainer.util import get_node_classes

def pretty_mechanistic_explanation(pred_n_expl, node_classes):
    predicted_mov = pred_n_expl.get('predicted_mov')
    explanation = pred_n_expl.get('explanation')

    obj = node_classes[predicted_mov[0]]
    src = node_classes[predicted_mov[1]]
    dst = node_classes[predicted_mov[2]]
    label = f"{obj}: {src} → {dst}"

    
    print(f"Prediction: {obj} moves from *{src}* to *{dst}")

    movement_perturbation = explanation.get("movement_perturbation", [])
    time_perturbation = explanation.get("time_perturbation", [])

    if movement_perturbation:
        print("Movements that are important for this:")
        for mov in movement_perturbation:
            pert_obj = node_classes[mov["object"]]
            pert_prev = node_classes[mov["previous_parent"]]
            pert_next = node_classes[mov["curr_parent"]]
            print(f"- {pert_obj} moves from {pert_prev} to {pert_next}")

    if time_perturbation:
        print("Time steps where changing context influenced prediction:")
        print(time_perturbation)
    
def templated_explanation(pred_n_expl, node_classes):
    predicted_mov = pred_n_expl.get('predicted_mov')
    explanation = pred_n_expl.get('explanation')

    obj = node_classes[predicted_mov[0]]
    src = node_classes[predicted_mov[1]]
    dst = node_classes[predicted_mov[2]]
    label = f"{obj}: {src} → {dst}"

    
    explanation_str = f"I  moved {obj} from {src} to {dst}, "

    movement_perturbation = explanation.get("movement_perturbation", [])
    time_perturbation = explanation.get("time_perturbation", None)

    if movement_perturbation:
        explanation_str += "since I noticed:\n"
        for mov in movement_perturbation:
            pert_obj = node_classes[mov["object"]]
            pert_prev = node_classes[mov["previous_parent"]]
            pert_next = node_classes[mov["curr_parent"]]
            explanation_str += f"- {pert_obj} moves from {pert_prev} to {pert_next}\n"

    if time_perturbation:
        time_perturb_string = time_perturbation["time_perturb_string"]
        if time_perturb_string:
            explanation_str += f"I felt more confident since {time_perturb_string}\n"
        # explanation_str += "Time steps where changing context influenced prediction:\n"
        # explanation_str += f"{time_perturbation}\n"
    
    return explanation_str
if __name__ == "__main__":
    # parameters
    home_id = 2
    day_no = 6
    routine_no = 39
    time_target = None
    explainer = Explainer(step_size=1, household_id=home_id)  # Example params

    pred_n_expl_s = explainer.run_for_single_instance(day_no=day_no, routine_no=routine_no, time_target=time_target)
    print("Mechanistic explanation rendering completed.")

    # print("pred_n_expl:", pred_n_expl_s)
    # assert len(pred_n_expl_s) == 1
    if(len(pred_n_expl_s) == 0 ):
        print("No action")

    # print("mechnasitic explanation:", pred_n_expl[0])
    for pred_n_expl in pred_n_expl_s:
        node_classes = get_node_classes()
        # pretty_mechanistic_explanation(pred_n_expl, node_classes)
        explanation_str = templated_explanation(pred_n_expl, node_classes)
        print("\nTemplated Explanation:\n", explanation_str)
    # st.rerun()