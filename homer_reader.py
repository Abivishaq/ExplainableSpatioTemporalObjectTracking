import json
import os
import torch
import numpy as np
from helpers.reader import ProcessDataset

class StotConvetionData:
    def __init__(self,dataset_dir = 'data/STOT_convetion_files'):
        with open(os.path.join(dataset_dir,'common_data.json'),'r') as f:
            self.common_data = json.load(f)
        self.node_idx_from_id = self.common_data['node_idx_from_id']
        self.STOT_num_elements = len(self.node_idx_from_id)
        print("node_idx_from_id:",self.node_idx_from_id)
        print("common_data:",self.common_data.keys())
        print("length of node_idx_from_id:",len(self.node_idx_from_id))

        
        # raise NotImplementedError("STOT convention data not implemented.")

        
class HomerReader:
    """
    A class to read and parse Homer data files.
    """

    def __init__(self, dataset_dir='data/Full_HOMER/household0/', split='test'):
        """
        Initializes the HomerReader with the specified dataset directory.
        
        Args:
            dataset_dir (str): The directory where the different days are stored.
            split (str): The dataset split to use ('train' or 'test').
        """
        self.dataset_dir = dataset_dir
        if split == 'train':
            self.__split = 'train'
            self.days_dir = os.path.join(dataset_dir, 'routines_train')
        elif split == 'test':
            self.__split = 'test'
            self.days_dir = os.path.join(dataset_dir, 'routines_test')
        else:
            raise ValueError("Invalid split. Choose 'train' or 'test'.")
        self.days_files = os.listdir(self.days_dir)
        print(f"Days files: {self.days_files}")
        self.days_files.sort()

        self.stot_convetion_data = StotConvetionData()
        self.not_found_keys = []
        self.relation_types =[]


    def read_day(self, day_index):
        """
        Reads the data for a specific day. 
        
        Args:
            day_index (int): The index of the day to read. Assumes days are stored in order starting from 0.
        
        Returns:
            dict: The parsed data for the specified day.
        """
        if day_index < 0 or day_index >= len(self.days_files):
            raise IndexError("Day index out of range.")
        
        day_file = os.path.join(self.days_dir, self.days_files[day_index])
        print(f"Reading day file: {day_file}")
        with open(day_file, 'r') as f:
            data = json.load(f)
        
        return data
    def calculate_node_id_to_STOT_id(self):
        for i in range(len(self.STOT_node_classes)):
            node_class = self.STOT_node_classes[i]
            if node_class['class_name'] == "unknown":
                continue
            node_id = node_class['id']
            print(f"Node class: {node_class['class_name']}, Node ID: {node_id}")


    # def get_stot_day(self, day_index):
    #     """
    #     Converts the data for a specific day to the STOT format.
    #     This method uses a uniform time sampling from 380 to 1490 (steps of 10 minutes).
        
    #     For each uniform time t:
    #         - prev_edges: is taken from the last graph that came before (or at) t.
    #         - edges: is taken from the graph immediately following that.
        
    #     Returns:
    #         list: A list of dictionaries formatted for STOT with keys:
    #             ['prev_edges', 'prev_nodes', 'time', 'edges', 'nodes', 'change_type', 'activity']
    #     """
    #     day_data = self.read_day(day_index)
    #     # The raw times (length N) and graphs (length N+1).
    #     raw_times = day_data['times']
    #     raw_graphs = day_data['graphs']
    #     processed_data = []

    #     # Helper function: Process a single graph into an incidence matrix.
    #     def process_edges(graph):
    #         # Assume the graph has a list of nodes and edges.
            
    #         num_nodes = self.stot_convetion_data.STOT_num_elements
    #         matrix = np.zeros((num_nodes, num_nodes), dtype=np.int64)
    #         # for node in nodes:
    #         #     print(f"node_id: {node['id']} -> node_name: {node['class_name']}")
    #         for edge in graph["edges"]:
            
    #             try:

    #                 print(f"edge: {edge}")
    #                 if(edge["relation_type"] == "CLOSE"):
    #                     continue
    #                 # if(edge["relation_type"] == "ON"):
    #                 #     continue
                    
    #                 i = self.stot_convetion_data.node_idx_from_id[str(edge["from_id"])]
    #                 j = self.stot_convetion_data.node_idx_from_id[str(edge["to_id"])]
    #                 if np.sum(matrix[:][j]) == 0:
    #                     matrix[i, j] = 1
                    
    #             except KeyError as e:
    #                 if str(edge["from_id"]) not in self.stot_convetion_data.node_idx_from_id:
    #                     if edge["from_id"] not in self.not_found_keys:
    #                         self.not_found_keys.append(edge["from_id"])
    #                         print(f"KeyError: {e} for edge {edge}")
    #                 if str(edge["to_id"]) not in self.stot_convetion_data.node_idx_from_id:
    #                     if edge["to_id"] not in self.not_found_keys:
    #                         self.not_found_keys.append(edge["to_id"])
    #                         print(f"KeyError: {e} for edge {edge}")
                        
    #                 # print(f"KeyError: {e} for edge {edge}")
    #                 continue
    #         return torch.tensor(matrix)

    #     # Create an identity matrix for nodes.
    #     # Note: Here we assume the number of nodes is consistent.
    #     num_nodes = self.stot_convetion_data.STOT_num_elements
    #     identity = torch.eye(num_nodes, dtype=torch.int64)
        
    #     # Initialize: use the initial graph (index 0) as the starting "prev_edges"
    #     prev_graph = raw_graphs[0]


    #     # We sample uniformly in the range 380 to 1490 with step 10.
    #     for t in range(380, 1500, 10):
    #         # Find the largest index in raw_times such that raw_times[idx] <= t.
    #         candidate_indices = [i for i, time in enumerate(raw_times) if time <= t]
    #         if not candidate_indices:
    #             # No raw time is less than or equal to t; use the initial state.
    #             current_idx = 0
    #         else:
    #             current_idx = max(candidate_indices)
    #         # The corresponding "current" graph is at index current_idx+1, because raw_times[0] corresponds to graphs[1].
    #         # (This works because raw_graphs has one extra, the initial state)
    #         current_graph = raw_graphs[current_idx + 1]
            
    #         # Build the incidence matrices.
    #         prev_edges = process_edges(prev_graph)
    #         cur_edges = process_edges(current_graph)
            
    #         # Compute change_type: compare each row between prev_edges and cur_edges.
    #         diff = (prev_edges != cur_edges)
    #         change_type = torch.where(diff.sum(dim=1) > 0, torch.tensor(1), torch.tensor(0))
            
    #         data_point = {
    #             'time': t,  # The uniform time stamp (int)
    #             'prev_edges': prev_edges,
    #             'edges': cur_edges,
    #             'prev_nodes': identity,
    #             'nodes': identity,
    #             'change_type': change_type,
    #             'activity': "unknown"  # Or set to a value from another source if available
    #         }
    #         processed_data.append(data_point)
            
    #         # Update prev_graph for next interval.
    #         prev_graph = current_graph

    #     return processed_data
    
    # def compare_processed_data_and_gt(self, processed_data, gt_data):
    #     """
    #     Compares the processed data with the ground truth data from STOT.
    #     For each datapoint, it loops over all keys and prints differences.
        
    #     Args:
    #         processed_data (list): List of dictionaries obtained from day_to_pt.
    #         gt_data (list): List of dictionaries from a .pt file.
    #     """
    #     for i, (proc, gt) in enumerate(zip(processed_data, gt_data)):
    #         print(f"----- Data point {i} -----")
    #         # checking nodes
    #         # print("GT nodes shape:", gt['nodes'].shape)
    #         # print("Processed nodes shape:", proc['nodes'].shape)

    #         for key in ['prev_edges']:
    #             p_val = proc[key]
    #             g_val = gt[key]
    #             # For tensors, we can compare with allclose
    #             if isinstance(p_val, torch.Tensor):
    #                 same = torch.equal(p_val, g_val)
    #             else:
    #                 same = (p_val == g_val)
    #             print(f"{key}: {same}")
    #             if not same:
    #                 print(f"Processed: {p_val}")
    #                 print(f"Ground Truth: {g_val}")
    #         print("\n")

    #         prev_edge_gt = gt['prev_edges']
    #         prev_edge_compressed = torch.sum(prev_edge_gt, dim=1)

    #         print(f"prev_edge_gt: {prev_edge_compressed}")
    #         prev_edge_pr = proc['prev_edges']
    #         prev_edge_compressed_pr = torch.sum(prev_edge_pr, dim=1)
    #         print(f"prev_edge_pr: {prev_edge_compressed_pr}")

    #         # argmax
    #         prev_edge_gt_max = torch.argmax(prev_edge_gt, dim=1)
    #         prev_edge_pr_max = torch.argmax(prev_edge_pr, dim=1)
    #         print(f"prev_edge_gt_max: {prev_edge_gt_max}")
    #         print(f"prev_edge_pr_max: {prev_edge_pr_max}")
    #         break

        
        
    
    
if __name__ == "__main__":
    dataset_dir = 'data/Full_HOMER/household0/'
    homer_reader = HomerReader(dataset_dir)
    
    # Example: Read the first day
    # day_data = homer_reader.read_day(0)

    # print(day_data.keys())
    # print(day_data['times'])
    # print(f"len_times: {len(day_data['times'])}")
    # print(f"len_graphs: {len(day_data['graphs'])}")
    # print(f"type(Graphs): {type(day_data['graphs'])}")
    # print(f"type(graphs[0]): {type(day_data['graphs'][0])}")
    # print(f"graph[0].keys(): {day_data['graphs'][0].keys()}")
    # # print(f"graph[0]['nodes']: {day_data['graphs'][0]['nodes']}")
    # print(f"type(graph[0]['nodes']): {type(day_data['graphs'][0]['nodes'])}")
    # print(f"len(graph[0]['nodes']): {len(day_data['graphs'][0]['nodes'])}")
    # print(f"graph[0]['nodes'][0] {day_data['graphs'][0]['nodes'][0]}")
    # nodes = day_data['graphs'][0]['nodes']
    # edges = day_data['graphs'][0]['edges']
    # print(f"edges: {edges}")
    # homer_reader.calculate_node_id_to_STOT_id()
    processed_data = homer_reader.get_stot_day(0)
    
    


    gt_data = torch.load('data/HOMER/household0/processed/test/000.pt')
    # print(type(gt_data))
    # for data_point in gt_data:
    #     print(data_point['time'])

    homer_reader.compare_processed_data_and_gt(processed_data=processed_data,gt_data=gt_data)
    print("not_found_keys:",homer_reader.not_found_keys)
    print(homer_reader.stot_convetion_data.node_idx_from_id)

    print("relation_types:",homer_reader.relation_types)
    