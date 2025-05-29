import sys
sys.path.append("helpers")
import json
import os
import torch
import numpy as np
from helpers.reader import _sparsify, not_a_tree
import shutil

from math import floor, ceil

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

class StotProcessDataset():
    def __init__(self, data_path, 
                 classes_path,
                 info_path):

        self.common_data = {}
        with open(info_path) as f:
            self.common_data['info'] = json.load(f)
        self.dt = self.common_data['info']['dt']
        self.read_classes(classes_path)


        self.data_path = data_path
        # with open(os.path.join(output_path, 'common_data.json'), 'w') as f:
        #     json.dump(self.common_data, f)
        # torch.save(self.seen_edges, os.path.join(output_path, 'seen_edges.pt'))
        # torch.save(self.nonstatic_edges, os.path.join(output_path, 'nonstatic_edges.pt'))
        # torch.save(self.home_graph, os.path.join(output_path, 'home_graph.pt'))

    def read_classes(self, classes_path):
        with open(classes_path, 'r') as f:
            classes = json.load(f)
        def ignore_node(node):
            return node['class_name'].startswith('clothes_')
        self.common_data['node_ids'] = [n['id'] for n in classes['nodes'] if not ignore_node(n)]
        self.common_data['node_classes'] = [n['class_name'] for n in classes['nodes'] if not ignore_node(n)]
        self.common_data['node_categories'] = [n['category'] for n in classes['nodes'] if not ignore_node(n)]
        self.common_data['node_idx_from_id'] = {int(n):i for i,n in enumerate(self.common_data['node_ids'])}
        self.common_data['actions'] = []

        # Diagonal nodes are always irrelevant
        self.nonstatic_edges = 1 - np.eye(len(self.common_data['node_ids']))
        # Rooms, furniture and appliances nodes don't move
        self.nonstatic_edges[np.where(np.array(self.common_data['node_categories']) == "Rooms"),:] = 0
        self.nonstatic_edges[np.where(np.array(self.common_data['node_categories']) == "Furniture"),:] = 0
        self.nonstatic_edges[np.where(np.array(self.common_data['node_categories']) == "Decor"),:] = 0
        self.nonstatic_edges[np.where(np.array(self.common_data['node_categories']) == "Appliances"),:] = 0
        self.seen_edges = np.zeros_like(self.nonstatic_edges)

        self.home_graph = None

        self.common_data['edge_keys'] = classes['edges']
        static = lambda category : category in ["Furniture", "Room"]
        self.common_data['static_nodes'] = [n['id'] for n in classes['nodes'] if static(n['category']) and not ignore_node(n)]

    def read_day_data(self, day_index):
        """
        Reads the data for a specific day and processes it into the STOT format.
        
        Args:
            day_index (int): The index of the day to read. Assumes days are stored in order starting from 0.
        """
        if day_index < 0 or day_index >= len(os.listdir(self.data_path)):
            raise IndexError("Day index out of range.")
        day_file = os.path.join(self.data_path, f'{day_index:03d}.json')
        with open(day_file) as f:
            routine = json.load(f)
        nodes, edges = self.read_graphs(routine["graphs"])
        self.home_graph = edges[0,:,:]
        times = torch.Tensor(routine["times"])
        activity_func = self.activity_from_time(day_file.replace('routines','scripts').replace('json','txt'))
        samples = self.make_pairwise(nodes, edges, times, activity_func)
        return samples
        
    def activity_from_time(self, script_file):
        scr_header_lines = open(script_file).read().split('\n\n\n')[0].split('\n')
        def parse_time(ts):
            parts = [int(t) for t in ts.split(':')]
            if len(parts) == 2:
                return parts[0]*60 + parts[1]
            if len(parts) == 3:
                return (parts[0]*24 + parts[1])*60 + parts[2]
            raise RuntimeError()
        def parse_line(l):
            activity = l[:l.index('(')-1].strip()
            timerange = l[l.index('(')+1: l.index(')')]
            timerange = timerange.replace('1day - ','01:')
            start_time = parse_time(timerange.split('-')[0].strip())
            end_time = parse_time(timerange.split('-')[1].strip())
            return activity, lambda t: t>=start_time and t<end_time
        activities = {parse_line(l)[0]:parse_line(l)[1] for l in scr_header_lines}
        def activity_func(t):
            return 0
            options = [self.common_data['activities'].index(a) for a,fun in activities.items() if fun(t)]
            if len(options) > 0:
                return options[0]
            else:
                return 0 #self.common_data['activities'].index(None)
        return activity_func

    def read_graphs(self, graphs):
        num_nodes = len(self.common_data['node_ids'])
        node_features = np.zeros((len(graphs), num_nodes, num_nodes))
        edge_features = np.zeros((len(graphs), num_nodes, num_nodes))
        for i,graph in enumerate(graphs):
            node_features[i,:,:num_nodes] = np.eye(num_nodes)
            for e in graph['edges']:
                if e['relation_type'] in self.common_data['edge_keys'] and e['from_id'] in self.common_data['node_ids'] and e['to_id'] in self.common_data['node_ids']:
                    edge_features[i,self.common_data['node_idx_from_id'][e['from_id']],self.common_data['node_idx_from_id'][e['to_id']]] = 1
            original_edges = edge_features[i,:,:]
            edge_features[i,:,:] = _sparsify(edge_features[i,:,:])
            if (edge_features[i,:,:].sum(axis=-1)).max() != 1:
                print(f"Matrix {i} not really a tree \n{edge_features[i,:,:]}")
                not_a_tree(original_edges, edge_features[i,:,:], self.common_data['node_classes'])
            assert (edge_features[i,:,:].sum(axis=-1)).max() == 1, f"Matrix {i} not really a tree \n{edge_features[i,:,:]}"
            self.seen_edges[:,:] += edge_features[i,:,:]
        return torch.Tensor(node_features), torch.Tensor(edge_features)

    def make_pairwise(self, nodes, edges, times, activity_func):
        pairwise_samples = []
        self.time_min = torch.Tensor([float("Inf")])
        self.time_max = -torch.Tensor([float("Inf")])
    
        assert times[0]==min(times), 'Times need to be monotonically increasing. First element should be min.'
        assert times[-1]==max(times), 'Times need to be monotonically increasing. Last element should be max.'
        time_min = floor(times[0]/self.dt)*self.dt
        time_max = ceil(times[-1]) + self.dt
        if time_min < self.time_min: self.time_min = time_min
        if time_max > self.time_max: self.time_max = time_max
        times = torch.cat([times,torch.Tensor([float("Inf")])], dim=-1)
        data_idx = -1
        prev_edges = None
        for t in range(time_min, time_max, self.dt):
            while t >= times[data_idx+1]:
                data_idx += 1
            if data_idx < 0:
                continue
            # 3 = to home state; 1 = from home state; 2 = neither; 0 = no change
            if prev_edges is not None:
                change_type = (np.absolute(edges[data_idx] - prev_edges)).sum(-1).to(int)
                change_type += (self.home_graph * (edges[data_idx] - prev_edges)).sum(-1).to(int)
                pairwise_samples.append({'prev_edges': prev_edges, 'prev_nodes': prev_nodes, 'time': t, 'edges': edges[data_idx], 'nodes': nodes[data_idx], 'change_type':change_type, 'activity':None})
            prev_edges = edges[data_idx]
            prev_nodes = nodes[data_idx]
        return pairwise_samples

        
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

        ### 
        self.stot_reader = StotProcessDataset(
            data_path=self.days_dir,
            classes_path=os.path.join(dataset_dir, 'classes.json'),
            info_path=os.path.join(dataset_dir, 'info.json')
        )


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

    def get_stot_day(self, day_index):
        """
        Reads the data for a specific day and processes it into the STOT format.
        
        Args:
            day_index (int): The index of the day to read. Assumes days are stored in order starting from 0.
        
        Returns:
            ...
        """
        return(self.stot_reader.read_day_data(day_index))

    
    def compare_processed_data_and_gt(self, processed_data, gt_data):
        """
        Compares the processed data with the ground truth data from STOT.
        For each datapoint, it loops over all keys and prints differences.
        
        Args:
            processed_data (list): List of dictionaries obtained from day_to_pt.
            gt_data (list): List of dictionaries from a .pt file.
        """
        for i, (proc, gt) in enumerate(zip(processed_data, gt_data)):
            print(f"----- Data point {i} -----")
            # checking nodes
            # print("GT nodes shape:", gt['nodes'].shape)
            # print("Processed nodes shape:", proc['nodes'].shape)

            for key in ['prev_edges', 'prev_nodes', 'edges', 'nodes']:
                p_val = proc[key]
                g_val = gt[key]
                # For tensors, we can compare with allclose
                if isinstance(g_val, torch.Tensor):
                    same = torch.equal(p_val, g_val)
                else:
                    same = (p_val == g_val)
                print(f"{key}: {same}")
                if not same:
                    print(f"Processed: {p_val}")
                    print(f"Ground Truth: {g_val}")
            for key in ['change_type', 'activity']:
                p_val = proc[key]
                g_val = gt[key]
                # For tensors, we can compare with allclose
                if isinstance(g_val, torch.Tensor):
                    same = torch.equal(p_val, g_val)
                else:
                    same = (p_val == g_val)
                print(f"{key}: {same}")
                if not same:
                    print(f"Processed: {p_val}")
                    print(f"Ground Truth: {g_val}")
                    diff = p_val - g_val
                    print(f"Difference: {diff}")
                    # get non-zero indices
                    non_zero_indices = torch.nonzero(diff)
                    for index in non_zero_indices:
                        print(index)
                        print(diff)
                        print(self.stot_reader.common_data['node_classes'][index])
                        # print(f"Index: {index}, Processed: {self.common_data['class_names'][index]}, parent: {self.common_data['class_names'][index]}")

            # print("\n")

            # prev_edge_gt = gt['prev_edges']
            # prev_edge_compressed = torch.sum(prev_edge_gt, dim=1)

            # print(f"prev_edge_gt: {prev_edge_compressed}")
            # prev_edge_pr = proc['prev_edges']
            # prev_edge_compressed_pr = torch.sum(prev_edge_pr, dim=1)
            # print(f"prev_edge_pr: {prev_edge_compressed_pr}")

            # # argmax
            # prev_edge_gt_max = torch.argmax(prev_edge_gt, dim=1)
            # prev_edge_pr_max = torch.argmax(prev_edge_pr, dim=1)
            # print(f"prev_edge_gt_max: {prev_edge_gt_max}")
            # print(f"prev_edge_pr_max: {prev_edge_pr_max}")
            break

        
        
    
    
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
    processed_data = homer_reader.get_stot_day(1)
    
    


    gt_data = torch.load('data/HOMER/household0/processed/test/000.pt')
    # print(type(gt_data))
    # for data_point in gt_data:
    #     print(data_point['time'])

    homer_reader.compare_processed_data_and_gt(processed_data=processed_data,gt_data=gt_data)
    # print("not_found_keys:",homer_reader.not_found_keys)
    # print(homer_reader.stot_convetion_data.node_idx_from_id)

    # print("relation_types:",homer_reader.relation_types)
    