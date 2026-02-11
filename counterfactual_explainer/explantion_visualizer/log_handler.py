import os
import streamlit as st
import torch
import warnings
import json
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from log_summary import LogSummary


class LogHandler:
    def __init__(self):
        self.log_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..' ,'logs')
        self.active_log_data = None
        self.active_routine_no = None
        self.homer_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data','HOMER')
        self.homer_folder = None
        self.node_classes = None
        self.log_summary = LogSummary()
    
    def get_homer_folder(self,household_id: int):
        """
        Set the HOMER folder path based on the household ID.
        """
        self.homer_folder = os.path.join(self.homer_root, f'household{household_id}')
        if not os.path.exists(self.homer_folder):
            st.error(f"HOMER folder for household {household_id} does not exist at {self.homer_folder}.")
            self.homer_folder = None
            return False
        else:
            # get the node classes from the homer folder
            node_classes_file = os.path.join(self.homer_folder,'processed' ,'common_data.json')
            if not os.path.exists(node_classes_file):
                st.error(f"Node classes file not found in {self.homer_folder}.")
                self.homer_folder = None
                return False
            with open(node_classes_file, 'r') as f:
                node_classes = json.load(f)
            self.node_classes = node_classes['node_classes']
            # st.success(f"HOMER folder for household {household_id} found at {self.homer_folder}.")
            return True
        
        
    
    def get_log_folder(self):
        # This function should have 3 dropdown selectors:
        # 1. Step size
        # 2. Household ID
        # 3. Day number
        # ls(self.log_fodler) -> ['step_size_1', 'step_size_2', ...]
        # ls(self.log_fodler/step_size_1) -> ['household_0', 'household_1', ...]
        # ls(self.log_fodler/step_size_1/household_0)
        # -> ['day_0_log.pt', 'day_1_log.pt', ...]
        # the dropdown selectors should be populated with the available options
        # Once selected, load the .pt file and save to self.active_log_data


        st.sidebar.title("Log File Selector")

        # Step size
        step_dirs = sorted([d for d in os.listdir(self.log_folder) if os.path.isdir(os.path.join(self.log_folder, d))])
        if not step_dirs:
            st.sidebar.warning("No step_size directories found.")
            return

        step_size = st.sidebar.selectbox("Step size", step_dirs)
        step_path = os.path.join(self.log_folder, step_size)

        # Household
        household_dirs = sorted([d for d in os.listdir(step_path) if os.path.isdir(os.path.join(step_path, d))])
        if not household_dirs:
            st.sidebar.warning("No household directories found.")
            return

        household = st.sidebar.selectbox("Household", household_dirs)
        household_path = os.path.join(step_path, household)
        self.get_homer_folder(int(household.split('_')[-1])) 
        # error if homer_folder is None
        if not self.homer_folder:
            st.sidebar.error("HOMER folder not found. Please check the household ID.")
    


        # Day logs
        log_files = sorted([f for f in os.listdir(household_path) if f.endswith("_log.pt")])
        if not log_files:
            st.sidebar.warning("No log files found in selected household.")
            return

        log_file = st.sidebar.selectbox("Day log file", log_files)
        log_path = os.path.join(household_path, log_file)

        # Load the selected .pt log file
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                self.active_log_data = torch.load(log_path, map_location='cpu')
            self.log_summary.summarize_log(self.active_log_data)
            st.sidebar.success(f"Loaded: {log_file}")
        except Exception as e:
            st.sidebar.error(f"Failed to load log file: {e}")
            self.active_log_data = None

    def traverser(self):
        # more about the log data structure:
        # log data is a dictionary with the following structure:
        # { 0: (context, pred_n_expl),
        #   1: (context, pred_n_expl),
        #   ... }
        # where context is a tuple of (time, edges) and pred_n_expl is a list of dicts
        # each dict in pred_n_expl has the following structure:
        # {'predicted_mov": (obj, previous_parent,next_parent),
        #  'explaination': dict with explanation details}
        # the explanation details is a dictionary with the following structure:
        # {"movement_perturbation": list of dicts,
        #  "time_perturbation": list of ints}
        # movement_perturbation is a list of dicts with the following structure:
        # {'obj': obj, 'previous_parent': previous_parent, 'next_parent': next_parent}

        # if active_log_data is not None
        # then allow a slider to select the routine number which is the key in the active_log_data dictionary
        # get the keys and select the range from the keys. it should be from 0 but just in case get the min and max keys and slider shoud have a step of 1
        # Store the value in self.active_routine_no.
        if self.active_log_data is None:
            st.warning("No log data loaded. Please select a log file.")
            return None

        st.sidebar.title("Routine Traverser")

        if self.log_summary.predicted_move_frequency:
            with st.sidebar:
                self.log_summary.plot_frequency_graph()

        keys = sorted(self.active_log_data.keys())
        if not keys:
            st.warning("Loaded log file is empty.")
            return None

        min_key, max_key = min(keys), max(keys)
        self.active_routine_no = st.sidebar.slider("Select routine number", min_value=min_key, max_value=max_key, step=1)

        if self.log_summary.true_movement_frequency:
            with st.sidebar:
                self.log_summary.plot_frequency_graph(true_movement=True)


        return self.active_routine_no, self.active_log_data[self.active_routine_no]