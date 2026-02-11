import streamlit as st
import os
import json
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'helpers')))
from reader import RoutinesDataset
from encoders import TimeEncodingOptions
import torch

class HOMERNavigator:
    def __init__(self):
        self.active_routine = None
        self.distractor_routine = None
        self.dataset = None
        self.dataset_path = None
        self.prev_day_idx = None
        self.prev_routine_idx = None

    def get_dataset(self, data_dir, batch_size=32, train_days=30, time_encoding_type='sine_informed'):
        """
        Returns a RoutinesDataset object from the provided directory.

        Args:
            data_dir (str): Path to HOMER dataset root directory.
            batch_size (int): Batch size for data loader.
            train_days (int): Number of routines to load.
            time_encoding_type (str): Encoding strategy for time.

        Returns:
            RoutinesDataset
        """
        with open(os.path.join(data_dir, 'processed', 'common_data.json')) as f:
            data_info = json.load(f)['info']

        time_options = TimeEncodingOptions(data_info.get('weeekend_days', None))
        time_encoding = time_options(time_encoding_type)

        return RoutinesDataset(
            data_path=os.path.join(data_dir, 'processed'),
            time_encoder=time_encoding,
            batch_size=batch_size,
            max_routines=(train_days, None)
        )

    def get_homer_dataset(self):
        st.header("HOMER Dataset Navigator")

        data_dir = st.text_input("Enter path to HOMER dataset directory (must contain 'processed'):",
                                 value=self.dataset_path or 'data/HOMER/household0', key='data_dir_input')

        if not data_dir or not os.path.exists(data_dir):
            st.warning("Please enter a valid path to the dataset directory.")
            return

        # Only reload dataset if path changed
        if self.dataset is None or data_dir != self.dataset_path:
            self.dataset_path = data_dir
            self.dataset = self.get_dataset(data_dir)

        dataset = self.dataset
        num_days = len(dataset.test_routines)
        day_idx = st.slider("Select Day", 0, num_days - 1, self.prev_day_idx or 0)
        routines, _ = dataset.test_routines.get_routine(day_idx)

        max_routine_idx = len(routines) - 1
        routine_idx = st.slider("Select Routine Step", 0, max_routine_idx, self.prev_routine_idx or 0)

        # Only update state if selection changed
        if (day_idx != self.prev_day_idx) or (routine_idx != self.prev_routine_idx):
            self.prev_day_idx = day_idx
            self.prev_routine_idx = routine_idx
            self.active_routine = dataset.test_routines.collate_fn([routines[routine_idx]])
            self.distractor_routine = self.copy_active_routine()
            self.node_classes = dataset.node_classes

        return dataset, day_idx, routine_idx

    def get_current_selection(self):
        """
        Return the currently loaded dataset and the last selected
        (day_idx, routine_idx) without creating any Streamlit widgets.

        Intended for use in non-UI code paths (e.g., explanation runner)
        to avoid duplicating widget definitions/keys.
        """
        if self.dataset is None or self.prev_day_idx is None or self.prev_routine_idx is None:
            raise ValueError("Dataset or selection not initialized. Call get_homer_dataset() first.")

        return self.dataset, self.prev_day_idx, self.prev_routine_idx

    def copy_active_routine(self):
        copy_routine = {}
        for key, value in self.active_routine.items():
            if isinstance(value, torch.Tensor):
                # Clone tensors to avoid modifying the original routine
                copy_routine[key] = value.clone()
            else:
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                raise TypeError(f"Unsupported type for key '{key}': {type(value)}")
        return copy_routine