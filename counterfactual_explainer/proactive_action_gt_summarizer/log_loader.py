import os
import torch
import warnings
import json
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))



class LogLoader:
    def __init__(self,household_id: int, step_size: int=1):
        self.log_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..' ,'logs')
        self.active_log_data = None
        self.active_routine_no = None
        self.homer_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data','HOMER')
        self.homer_folder = None
        self.node_classes = None
        self.household_id = household_id
        self.step_size = step_size

        self.load_logs()
    
    def get_homer_folder(self,household_id: int):
        """
        Set the HOMER folder path based on the household ID and load node classes.
        """
        self.homer_folder = os.path.join(self.homer_root, f'household{household_id}')
        if not os.path.exists(self.homer_folder):
            raise FileNotFoundError(f"HOMER folder for household {household_id} does not exist at {self.homer_folder}.")
        else:
            # get the node classes from the homer folder
            node_classes_file = os.path.join(self.homer_folder,'processed' ,'common_data.json')
            if not os.path.exists(node_classes_file):
                raise FileNotFoundError(f"Node classes file not found in {self.homer_folder}.")
            with open(node_classes_file, 'r') as f:
                node_classes = json.load(f)
            self.node_classes = node_classes['node_classes']
            # st.success(f"HOMER folder for household {household_id} found at {self.homer_folder}.")
            return True
    
    def load_logs(self):
        """
        Load the log data for the specified household ID.
        """
        self.log_data = {}
        self.get_homer_folder(self.household_id)
        step_dir = os.path.join(self.log_folder, f'step_size_{self.step_size}')
        log_household_dir = os.path.join(step_dir, f'household_{self.household_id}')
        if not os.path.exists(log_household_dir):
            raise FileNotFoundError(f"Log directory for household {self.household_id} does not exist at {log_household_dir}.")
        
        for file in os.listdir(log_household_dir):
            if file.endswith('.pt'):
                day_number = int(file.split('_')[1].split('.')[0])
                log_path = os.path.join(log_household_dir, file)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # Load the log data for the specific day
                    self.log_data[day_number] = torch.load(log_path, map_location=torch.device('cpu'))
            else:
                raise ValueError(f"Unexpected file format: {file}. Expected .pt files.")
            



   