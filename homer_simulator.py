import sys
sys.path.append("helpers")
import os
import json
from homer_reader import HomerReader


class HomerSimulator:
    def __init__(self, data_dir='data/Full_HOMER/household0/', split='test'):
        """
        Initializes the HomerSimulator with the specified dataset directory and split.
        Args:
            data_dir (str): The directory to the household.
            split (str): The dataset split to use ('train' or 'test').
        """
        # Load configuration
        self.path_to_days = data_dir
        if split == 'train':
            self.split = 'train'
            self.path_to_days = os.path.join(data_dir, 'routines_train')
        elif split == 'test':
            self.split = 'test'
            self.path_to_days = os.path.join(data_dir, 'routines_test')
        else:
            raise ValueError("Invalid split. Choose 'train' or 'test'.")
        
    
    def simulate_day(self, day=0):
        
        
    def run_time_step(self):
        pass


if __name__ == "__main__":
    data_dir = 'data/HOMER/household0/'
    cfg_filename = 'model_configs.pkl'
    simulator = HomerSimulator(data_dir, cfg_filename)
    simulator.simulate_day(0)

    