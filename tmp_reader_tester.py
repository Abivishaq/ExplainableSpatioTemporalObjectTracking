import sys
sys.path.append("helper")
from helpers.reader import ProcessDataset


def test_process_dataset():
    data_path = "/home/abivishaq/projects/Explainations_for_PA/SpatioTemporalObjectTracking/data/Full_HOMER/household0"
    classes_path = "data/Full_HOMER/household0/classes.json"
    info_path = "data/Full_HOMER/household0/info.json"
    output_path = "data/tmp_HOMER"
    dataset = ProcessDataset(data_path, classes_path, info_path, output_path)
