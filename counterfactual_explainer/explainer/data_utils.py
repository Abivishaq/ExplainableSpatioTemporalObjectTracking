import os
from helpers.reader import RoutinesDataset

class DatasetManager:
    def __init__(
        self,
        data_dir,
        time_encoder=None,
        batch_size=32,
        train_days=30
    ):
        """
        Initializes and loads the RoutinesDataset.
        """
        self.data_dir = data_dir
        self.processed_path = os.path.join(data_dir, 'processed')
        self.time_encoder = time_encoder
        self.batch_size = batch_size
        self.train_days = train_days
        self.household_id = int(data_dir.split('household')[-1].split('/')[0])

        self.dataset = RoutinesDataset(
            data_path=self.processed_path,
            time_encoder=self.time_encoder,
            batch_size=self.batch_size,
            max_routines=(self.train_days, None)
        )

    def get_dataset(self):
        """
        Returns the full RoutinesDataset object (if needed externally).
        """
        return self.dataset

    @property
    def test_routines(self):
        return self.dataset.test_routines
    
    def get_iterator(self, day_routine, step_size):
        """
        Yields routine windows of length `step_size`, each formatted as a list of collated dicts.

        Args:
            day_routine: sequence of scene graph steps for a day
            step_size (int): number of consecutive steps to include in the routine window

        Yields:
            List[Dict]: each dict is ready to be passed to the model
        """
        routine_length = len(day_routine)
        for start in range(0, routine_length - step_size + 1):
            window = [day_routine[i] for i in range(start, start + step_size)]
            collated = [self.dataset.test_routines.collate_fn([step]) for step in window]
            yield collated
    
    

