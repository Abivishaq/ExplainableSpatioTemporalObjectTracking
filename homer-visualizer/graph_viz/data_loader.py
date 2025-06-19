import os
import json
import sys

# Ensure helpers path (one level up from the visualizer directory)
current_dir = os.path.dirname(os.path.abspath(__file__))
helpers_path = os.path.abspath(os.path.join(current_dir, '..', '..','helpers'))
sys.path.append(helpers_path)

from reader import RoutinesDataset
from encoders import TimeEncodingOptions


def get_dataset(data_dir, batch_size=32, train_days=30, time_encoding_type='sine_informed'):
    """
    Initializes a RoutinesDataset from a processed VirtualHome dataset directory.

    Parameters:
        data_dir (str): Path to the dataset folder (must contain 'processed' subfolder).
        batch_size (int): Batch size for the dataset.
        train_days (int): Max number of routines to use for training split.
        time_encoding_type (str): Time encoding strategy ('sine', 'sine_informed', etc.)

    Returns:
        RoutinesDataset: Initialized dataset object.
    """
    processed_path = os.path.join(data_dir, 'processed')
    with open(os.path.join(processed_path, 'common_data.json')) as f:
        data_info = json.load(f)['info']

    weekend_days = data_info.get('weeekend_days', None)
    time_options = TimeEncodingOptions(weekend_days)
    time_encoding = time_options(time_encoding_type)

    return RoutinesDataset(
        data_path=processed_path,
        time_encoder=time_encoding,
        batch_size=batch_size,
        max_routines=(train_days, None)
    )


def get_routine(dataset: RoutinesDataset, routine_idx: int = 0):
    """
    Loads a single routine (e.g., day) from the test split.

    Parameters:
        dataset (RoutinesDataset): Dataset object.
        routine_idx (int): Index of the routine to retrieve.

    Returns:
        list: List of timestep tuples for that day.
    """
    routine, _ = dataset.test_routines.get_routine(routine_idx)
    return routine
