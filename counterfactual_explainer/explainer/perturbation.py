import torch

class PerturbationEngine:
    def __init__(self, model, movement_tracker, debugger=None):
        """
        PerturbationEngine for analyzing model predictions by perturbing movements and time.
        
        Args:
            model (STOTModel): STOT model used to run inference on perturbed routines.
            mvoement_tracker (MovementTracker): Used to access the recorded movement history.
        """
        self.model = model
        self.movement_tracker = movement_tracker
        self.debugger = debugger
    
    def clone_routine_window(self, routine_window):
        """
        Clones a routine window to avoid modifying the original during perturbation tests.
        
        Args:
            routine_window (List[Dict]): Sequence of 1 or more time steps in the HOMER routine.
        """
       
        cloned_window = []
        for step in routine_window:
            cloned_step = {k: v.clone() for k, v in step.items()}
            cloned_window.append(cloned_step)
        return cloned_window
    
    def explain_predicted_movement(self, routine_window, pred_mov, true_pred):
        """
        Explains one predicted movement. 

        Args:
            routine_window (List[Dict])
            pred_mov (List[int]): List containing [object, previous parent, new parent] ie the model predicted that object moved from previous parent to new parent.
        """
        valid_movement_perturbation = []
        #############################################
        # Stage 1: movement perturbation
        movement_dict = self.movement_tracker.movement_dict
        for obj in movement_dict.keys():
            perturb_routine = self.clone_routine_window(routine_window)

            # Perturbing the input
            mov_perturb_obj_curr_parent = torch.argmax(perturb_routine[0]['edges'][0][obj], dim=-1).item()
            perturb_routine[0]['edges'][0][obj][mov_perturb_obj_curr_parent] = 0
            perturb_routine[0]['edges'][0][obj][movement_dict[obj]] = 1
            
            # Run inference on perturbed routine
            inp, pred, gt, edge_probs = self.model.infer(perturb_routine)
            
            # check if predicted movement is influential
            if(pred[pred_mov[0]]!=true_pred[pred_mov[0]]):  
                # prediction is influenced by the perturbation
                valid_movement_perturbation.append({"object": obj, "previous_parent": movement_dict[obj], "curr_parent": mov_perturb_obj_curr_parent})
        ############### END OF STAGE 1 ################
          
        ###################################################################
        # Stage 2: time perturbation from 380 to 1550 (Range obeserved in HOMER dataset)
        influential_time_perturbation = []
        for i in range(380, 1551, 10):
            break  # Skipping the time perturbation stage for now for faster testing.
            # cloning the routine window to avoid modifying the original
            perturb_routine = self.clone_routine_window(routine_window)

            # Perturbing the time
            perturb_time_target = i
            for step in perturb_routine:
                
                step['time'] = torch.tensor(perturb_time_target, device=step['time'].device, dtype=step['time'].dtype)
                step['context_time'] = self.model.time_encoder(perturb_time_target).unsqueeze(0) # Note : This returns a float 32 tensor. Which defers from the HOMER dataset context_time which is a float 64 tensor. Very minor difference in values. e-5 . Modify time encoder fucntion to return float 64 if needed.
                perturb_time_target += 10  # Increment time for each step

            # Run inference on perturbed routine window
            inp, pred, gt, edge_probs = self.model.infer(perturb_routine)        

            # check if predicted movement is influential
            if(pred[pred_mov[0]]!=true_pred[pred_mov[0]]):
                # prediction is influenced by the perturbation
                influential_time_perturbation.append(i)
        ############### END OF STAGE 2 ################    
        
        return {"movement_perturbation":valid_movement_perturbation, "time_perturbation": influential_time_perturbation}
    
    
    def get_predicted_movements(self, inp, pred):
        """
        Extracts predicted movements from the model's output.
        
        Args:
            inp (Tensor): Input tensor before perturbation with shape [num_nodes]
            pred (Tensor): Predicted tensor before perturbation with shape [num_nodes]
        
        Returns:
            List: List of predicted movements in the format [object, previous parent, new parent]
        """
        assert inp.shape == pred.shape, "Input and predicted tensors must have the same shape."
        assert inp.dim() == 1, "Input and predicted tensors must be 1D."
        
        movements = []
        for obj in range(inp.shape[0]):
            prev_parent = inp[obj].item()
            new_parent = pred[obj].item()
            if prev_parent != new_parent:
                movements.append([obj, prev_parent, new_parent])
        
        return movements
        
    def run(self, routine_window, inp, pred):
        """
        """
        routine_window_copy = self.clone_routine_window(routine_window)
        # Step 1: Get predicted movements
        pred_movements = self.get_predicted_movements(inp, pred)
        
 
        # Step 2: loop through each prediction and explain it
        results = []
        for pred_mov in pred_movements:
            exp_pred = self.explain_predicted_movement(
                routine_window_copy,
                pred_mov,
                true_pred=pred
            )
            results.append({"predicted_mov":pred_mov,"explanation":exp_pred})

        return results

        
