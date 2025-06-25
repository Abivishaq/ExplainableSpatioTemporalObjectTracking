import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','helpers')))


from stot_model import STOTModel
from data_utils import DatasetManager
from movement_tracker import MovementTracker
from perturbation import PerturbationEngine
from logger import Logger  
from debugger import Debugger  




class Explainer:
    def __init__(self, step_size):

        # Initialize core components
        self.model = STOTModel(step_size=step_size)
        # self.logger = Logger()

        # Load dataset
        self.data_manager = DatasetManager(
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
        self.logger = Logger(log_folder=log_folder, household_id=self.data_manager.household_id, step_size=step_size)

        self.skip_if_log_exists = True

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
                print(f"Processing routine {no + 1}/{len(day_routine) - self.step_size + 1}...")
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




if __name__ == "__main__":
    explainer = Explainer(step_size=1)
    explainer.run()
    print("Explainer run completed.")