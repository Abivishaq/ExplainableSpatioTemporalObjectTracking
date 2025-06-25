import os
import json

class Debugger:
    def __init__(self, node_classes,movement_detector):
        self.node_classes = node_classes
        self.movement_detector = movement_detector

    def visualize_model_run(self, input, gt, pred):
        # print("Input:", input)
        # print("Ground Truth:", gt)
        # print("Prediction:", pred)
        
        
        # true movements
        true_movement = self.vector_diff(input, gt)
        print("#######################################")
        print("True Movements:")
        for i in true_movement:
            print(f"{self.node_classes[i[0]]}: {self.node_classes[i[1]]} -> {self.node_classes[i[2]]}")
        # predicted movements
        pred_movement = self.vector_diff(input, pred)
        print("------------------------------------------")
        print("Predicted Movements:")
        for i in pred_movement:
            print(f"{self.node_classes[i[0]]}: {self.node_classes[i[1]]} -> {self.node_classes[i[2]]}")
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    
    def verify_model_returns(self, input_tensor, gt_tensor, routine_window):
        """
        Verifies that the input is the first element of the routine window, and gt is the last.
        """
        print(type(routine_window))
        print(routine_window[0].keys())

        print(f"Input tensor shape: {input_tensor.shape}")
        print(f"Ground Truth tensor shape: {gt_tensor.shape}")
        print(f"Routine window length: {len(routine_window)}")
        print(f"Routine window first edges shape: {routine_window[0]['edges'].shape}")
        print(f"Routine window last y_edges shape: {routine_window[-1]['y_edges'].shape}")
        rw0_squeezed = routine_window[0]['edges'].squeeze(0)
        rw_last_squeezed = routine_window[-1]['y_edges'].squeeze(0)
        print(f"Routine window first edges squeezed shape: {rw0_squeezed.shape}")
        print(f"Routine window last y_edges squeezed shape: {rw_last_squeezed.shape}")

        rw0_argmax = rw0_squeezed.argmax(-1)
        rw_last_argmax = rw_last_squeezed.argmax(-1)
        print(f"Routine window first edges argmax shape: {rw0_argmax.shape}")
        print(f"Routine window last y_edges argmax shape: {rw_last_argmax.shape}")
    
        
        input_verified = self.compare_vectors(input_tensor, rw0_argmax)
        print(f"Input verified: {input_verified}")

        gt_verified = self.compare_vectors(gt_tensor, rw_last_argmax)

        print(f"Ground Truth verified: {gt_verified}")

    def compare_vectors(self, vec1, vec2):
        """
        Compares two vectors and returns True if they are equal, False otherwise.
        """
        if len(vec1) != len(vec2):
            return False
        for i in range(len(vec1)):
            if vec1[i] != vec2[i]:
                return False
        return True


    def pretty_print_movement_detected(self):
        """
        Pretty prints the movement detected in the movement detector.
        """
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print("Movement Detected:")
        for node_idx, prev_parent in self.movement_detector.movement_dict.items():
            print(f"{self.node_classes[node_idx]}  from  {self.node_classes[prev_parent]}")
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

    
    def vector_diff(self, vec1, vec2):
        """
        returns the a list (index, val1, val2) for each index where the two vectors differ
        """
        differences = []
        for i in range(len(vec1)):
            if vec1[i] != vec2[i]:
                differences.append((i, vec1[i], vec2[i]))
        return differences
    
    def perturb_print_movement_detected(self, pred_movements):
        print("+++++++++++++++++++++++++++++")
        print("predicted movements from perturbation:")
        for movement in pred_movements:
            obj, prev_parent, new_parent = movement[0], movement[1], movement[2]
            print(f"Object {self.node_classes[obj]} moved from {self.node_classes[prev_parent]} to {self.node_classes[new_parent]}")
        print("==============================")