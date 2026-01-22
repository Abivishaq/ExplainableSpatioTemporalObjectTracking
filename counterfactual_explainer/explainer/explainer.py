import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','helpers')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from stot_model import STOTModel
from data_utils import DatasetManager
from movement_tracker import MovementTracker
from perturbation import PerturbationEngine
from logger import Logger  
from debugger import Debugger
import torch




class Explainer:
    def __init__(self, step_size, household_id):

        self.household_id = household_id
        # Initialize core components
        self.model = STOTModel(step_size=step_size, household_id=self.household_id)
        # self.logger = Logger()

        dataset_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data','HOMER' ,f'household{self.household_id}')

        # Load dataset
        self.data_manager = DatasetManager(
            data_dir=dataset_dir,
            time_encoder=self.model.time_encoder,
            batch_size=self.model.model_configs['batch_size'],
            train_days=30
        )

        self.num_nodes = self.model.num_nodes
        self.movement_tracker = MovementTracker(num_nodes=self.num_nodes)
        self.debugger = Debugger(self.data_manager.dataset.node_classes, self.movement_tracker)
        
        self.perturb_engine = PerturbationEngine(
            model=self.model,
            movement_tracker=self.movement_tracker,
            debugger=self.debugger
        )
        self.step_size = step_size
        curr_file_location = os.path.dirname(os.path.abspath(__file__))
        log_folder = os.path.join(curr_file_location, '..', 'logs')
        self.logger = Logger(log_folder=log_folder, household_id=self.household_id, step_size=step_size)

        self.skip_if_log_exists = True

        assert self.household_id == self.data_manager.household_id

    def run(self):
        """
        Main inference + counterfactual analysis loop.
        Loads test routines, detects movement, and performs perturbation tests.
        """
        test_routines = self.data_manager.test_routines
        
        for day_no,(day_routine, additional_info) in enumerate(test_routines):
            if self.skip_if_log_exists:
                day_log_path = os.path.join(self.logger.log_folder, f'day_{day_no:02d}_log.pt')
                if os.path.exists(day_log_path):
                    print(f"Skipping day {day_no + 1} as log already exists at {day_log_path}")
                    continue
            print(f"Processing day {day_no + 1}/{len(test_routines)}...")
            routine_iterator = self.data_manager.get_iterator(day_routine, step_size=self.step_size)
            self.movement_tracker.reset()  # Reset movement tracker for each day 

            for no, routine_window in enumerate(routine_iterator):
                # print(f"Processing routine {no + 1}/{len(day_routine) - self.step_size + 1}...")
                # Step 1: Inference:
                inp, pred, gt, edge_probs = self.model.infer(routine_window)
                # self.debugger.verify_model_returns(inp, gt, routine_window)
                # self.debugger.visualize_model_run(inp, gt, pred)
                # self.debugger.pretty_print_movement_detected()
                
                # # Step 2: Peturbation:
                pred_n_expl = self.perturb_engine.run(routine_window, inp, pred)
                # # Step 3: log explanation results
                self.logger.log_explanation(day_no, no, pred_n_expl, routine_window[0], routine_window[-1])
                # # Step 4: Movement tracking
                self.movement_tracker.update(routine_window)
                # input('')
            self.movement_tracker.reset()  # Reset movement tracker for the next day
            self.logger.save_day_log(day_no)
        
    def run_for_single_instance(self, day_no, routine_no, time_target=None):
        """
        Run explainer for a single routine instance for debugging.
        """
        test_routines = self.data_manager.test_routines
        day_routine, additional_info = test_routines[day_no]
        routine_iterator = self.data_manager.get_iterator(day_routine, step_size=self.step_size)
        self.movement_tracker.reset()  # Reset movement tracker for each day 
        
        for no, routine_window in enumerate(routine_iterator):
            if no == routine_no:
                if time_target is not None:
                    perturb_time_target = time_target  # Example: perturb to afternoon time
                    for step in routine_window:
                        
                        step['time'] = torch.tensor(perturb_time_target, device=step['time'].device, dtype=step['time'].dtype)
                        step['context_time'] = self.model.time_encoder(perturb_time_target).unsqueeze(0)
                # Step 1: Inference:
                inp, pred, gt, edge_probs = self.model.infer(routine_window)
            
                # Step 2: Peturbation:
                pred_n_expl = self.perturb_engine.run(routine_window, inp, pred, edge_probs)
                
                # Step 3: log explanation results
                # self.logger.log_explanation(day_no, no, pred_n_expl, routine_window[0], routine_window[-1])
                break  # Exit after processing the specified routine
            else:    
                # Step 4: Movement tracking
                self.movement_tracker.update(routine_window)
        self.movement_tracker.reset()  # Reset movement tracker for the next day

        return pred_n_expl

    def run_for_single_instance_with_manual_movements(self, day_no, routine_no, manual_movements, time_target=None):
        """
        Run explainer for a single routine instance, using manually provided
        historic movements instead of (or in addition to) automatically
        detected ones.

        Args:
            day_no (int): Index of the day in the test set.
            routine_no (int): Index of the routine window within the day.
            manual_movements (Iterable[Tuple[int, int, int]]): Movements in the
                same format as `condensed_movements` from `MovementTracker.detect`,
                i.e. (node_idx, prev_parent, new_parent).
            time_target (int, optional): If provided, overrides the time for
                all steps in the target window (same semantics as
                `run_for_single_instance`).
        """
        test_routines = self.data_manager.test_routines
        day_routine, additional_info = test_routines[day_no]
        routine_iterator = self.data_manager.get_iterator(day_routine, step_size=self.step_size)

        # Reset for this run
        self.movement_tracker.reset()

        pred_n_expl = None

        for no, routine_window in enumerate(routine_iterator):
            if no == routine_no:
                # Seed movement history with user-provided movements for this run
                # and apply them directly to the scene graph of the current window.
                if manual_movements is not None:
                    # 1) Record movements in the tracker (for perturbation engine).
                    self.movement_tracker.add_movements(manual_movements)

                    # 2) Apply movements to the current routine window's edges tensor.
                    #    Each movement is (node_idx, prev_parent, new_parent).
                    edges = routine_window[0]['edges']  # shape: [1, num_nodes, num_nodes]
                    for node_idx, prev_parent, new_parent in manual_movements:
                        # Clear existing parent(s) for this node.
                        edges[0, node_idx, :] = 0
                        # Set the new parent.
                        edges[0, node_idx, int(new_parent)] = 1

                if time_target is not None:
                    perturb_time_target = time_target
                    for step in routine_window:
                        step['time'] = torch.tensor(
                            perturb_time_target,
                            device=step['time'].device,
                            dtype=step['time'].dtype
                        )
                        step['context_time'] = self.model.time_encoder(perturb_time_target).unsqueeze(0)
                # Step 1: Inference:
                inp, pred, gt, edge_probs = self.model.infer(routine_window)

                # Step 2: Perturbation (uses movement_tracker.movement_dict,
                # which now includes manual movements and any previously
                # accumulated movements).
                pred_n_expl = self.perturb_engine.run(routine_window, inp, pred, edge_probs)
                break
            else:
                # Populate natural movement history up to the target routine.
                self.movement_tracker.update(routine_window)

        # Clean up tracker state after this single-instance run.
        self.movement_tracker.reset()

        return pred_n_expl



if __name__ == "__main__":
    
    for i in range(0, 5): # Households 0 to 4
        for j in range(1,10): # step iterations 1 to 9
            print(f"Running explainer for household {i}, step {j}...")
            explainer = Explainer(step_size=j, household_id=i)
            explainer.run()
            print(f"Explainer run completed for household {i}, step {j}.\n")
    # explainer = Explainer(step_size=1, household_id=1)
    # explainer.run()
    print("Explainer run completed.")