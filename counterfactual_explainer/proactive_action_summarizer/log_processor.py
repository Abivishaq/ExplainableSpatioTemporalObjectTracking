import sys
import os
import pandas as pd
from log_loader import LogLoader


class LogProcessor:
    """
    A class to put loaded logs into a pandas DataFrame for further processing.
    """
    
    def __init__(self, log_loader: LogLoader):
        self.log_loader = log_loader
        self.step_size = log_loader.step_size
        self.df = pd.DataFrame()
        self.node_classes = log_loader.node_classes
        self.process_logs()
        this_file_dir = os.path.dirname(os.path.abspath(__file__))
        this_file_parent_dir = os.path.dirname(this_file_dir)
        household_id = log_loader.household_id
        if not os.path.exists(os.path.join(this_file_parent_dir, 'processed_logs')):
            os.makedirs(os.path.join(this_file_parent_dir, 'processed_logs'))
        self.save_to_csv(os.path.join(this_file_parent_dir, 'processed_logs', 'diff_steps', f'step_size_{self.step_size}', f'household_{household_id}_logs.csv'))

    
    def predicted_mov_to_str(self, predicted_mov):
        """
        Convert the predicted movement dictionary to a string representation.
        """
        if not predicted_mov:
            return "No movement"
        obj = predicted_mov[0]
        previous_parent = predicted_mov[1]
        next_parent = predicted_mov[2]
        obj_name = self.node_classes[obj]
        previous_parent_name = self.node_classes[previous_parent]
        next_parent_name = self.node_classes[next_parent]
        return f"{obj_name} from {previous_parent_name} to {next_parent_name}"
    
    def explanation_to_str(self, explanation):
        """
        Convert the explanation dictionary to a string representation.
        """
        if not explanation:
            return "No explanation"
        explanation_str = ""
        movement_perturbation = explanation["movement_perturbation"]
        time_perturbation = explanation["time_perturbation"]
        for movement in movement_perturbation:
            obj = movement['object']
            previous_parent = movement['previous_parent']
            next_parent = movement['curr_parent']
            obj_name = self.node_classes[obj]
            previous_parent_name = self.node_classes[previous_parent]
            next_parent_name = self.node_classes[next_parent]
            explanation_str += f"Since {obj_name} from {previous_parent_name} to {next_parent_name}\n"
        
        # raise NotImplementedError("Explanation formatting not implemented yet.")
        explanation_str.strip() 
        return(explanation_str)

    def process_logs(self):
        """
        Process the logs loaded by LogLoader and convert them into a DataFrame.
        # log data is a dictionary with the following structure: (key: day_number, value: day_dictionary)
        # { 0: day0_dictionary, 1: day1_dictionary, ... }
        # where day_dictionary is a dictionary with the following structure:
        # { 0: (context, pred_n_expl),
        #   1: (context, pred_n_expl),
        #   ... }
        # where context is a tuple of (time, edges, y_edges) and pred_n_expl is a list of dicts
        # each dict in pred_n_expl has the following structure:
        # {'predicted_mov": (obj, previous_parent,next_parent),
        #  'explaination': dict with explanation details}
        # the explanation details is a dictionary with the following structure:
        # {"movement_perturbation": list of dicts,
        #  "time_perturbation": list of ints}
        # movement_perturbation is a list of dicts with the following structure:
        # {'obj': obj, 'previous_parent': previous_parent, 'next_parent': next_parent}

        """
        if not self.log_loader.log_data:
            raise ValueError("No log data available to process.")
        
        # data = ["day_number, routine_no, time, edges, predicted_mov, explanation"]
        data = []
        for day, day_log in self.log_loader.log_data.items():
            for routine_no, routine_data in day_log.items():
                context, pred_n_expl = routine_data
                
                time, edges, y_edges = context
                for pred in pred_n_expl:
                    # print("Predicted Movement:", pred)
                    predicted_mov = pred.get("predicted_mov", {})
                    explanation = pred.get("explanation", {})
                    row = {
                        "day_number": day,
                        "routine_no": routine_no,
                        "time": time,
                        "edges": edges,
                        "predicted_mov": self.predicted_mov_to_str(predicted_mov),
                        "explanation": self.explanation_to_str(explanation)
                    }
                    data.append(row)
        
        self.df = pd.DataFrame(data)
    
    def save_to_csv(self, filename):
        """
        Save the processed DataFrame to a CSV file.
        """
        if self.df.empty:
            raise ValueError("DataFrame is empty. Nothing to save.")
        self.df.to_csv(filename, index=False)
        # print(f"Data saved to {filename}")

