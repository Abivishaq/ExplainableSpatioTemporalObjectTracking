import os
import torch
class Logger:
    def __init__(self, log_folder: str,household_id: int, step_size: int):
        os.makedirs(log_folder, exist_ok=True)
        log_folder = os.path.join(log_folder, f'step_size_{step_size}')
        os.makedirs(log_folder, exist_ok=True)
        log_folder = os.path.join(log_folder, f'household_{household_id}')
        os.makedirs(log_folder, exist_ok=True)
        self.log_folder = log_folder
        self.log_data = {}
    def cleanup_pred_moves_n_expl(self, pred_n_expl: dict):
        """
        removes the string keys from the log data to decrease the size of the log data
        """
        # Not implemented yet 
        # Can add this if you want to save space that logs take.
        return pred_n_expl


    def log_explanation(self, day_no: int, routine_no: int, pred_n_expl: list, routine, routine_last):
        """
        This function just updates the self.log_data variable with the explanation results.
        """
        countext_time = routine['time']
        context_edges = routine['edges']
        context_y_edges = routine_last['y_edges']
        context = (countext_time, context_edges, context_y_edges)
        cleaned_pred_n_expl = self.cleanup_pred_moves_n_expl(pred_n_expl)
        if day_no not in self.log_data:
            self.log_data[day_no] = {}
        self.log_data[day_no][routine_no] = (context, cleaned_pred_n_expl)


    def save_day_log(self, day_no: int):
        """
        Saves the log data for a specific day to a file.
        """
        if day_no in self.log_data:
            day_log_path = os.path.join(self.log_folder, f'day_{day_no:02d}_log.pt')
            torch.save(self.log_data[day_no], day_log_path)
            print(f"Day {day_no} log saved to {day_log_path}")

            