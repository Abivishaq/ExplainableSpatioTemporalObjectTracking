import unittest
import torch
import os
from explainer import Explainer  # Ensure this is correctly imported from your module

class TestExplainer(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.explainer = Explainer()
        cls.explainer.num_nodes = 5
    
    def test_label_coding(self):
        onehot_tensor = torch.zeros(1, self.explainer.num_nodes, self.explainer.num_nodes)
        onehot_tensor[0, 4, 2] = 1  # Setting a one-hot encoded value at index 5
        true_res = torch.tensor([0,0,0,0,2])
        
        result = self.explainer.label_coding(onehot_tensor)
        print("result: ", result)
        self.assertTrue(torch.equal(result, true_res))
    
    def test_label_coding_2(self):
        onehot_tensor = [[[0,0,0,0,0],
                         [1,0,0,0,0],
                         [0,0,0,0,1],
                         [0,1,0,0,0],
                         [0,1,0,0,0]]]
        onehot_tensor = torch.tensor(onehot_tensor)
        true_res = torch.tensor([0,0,4,1,1])
        result = self.explainer.label_coding(onehot_tensor)
        print("result: ", result)
        self.assertTrue(torch.equal(result, true_res))
    
    # def test_detect_movement(self):
    #     prev_edges = torch.zeros(self.explainer.num_nodes)
    #     edges = torch.zeros(self.explainer.num_nodes)
    #     edges[3] = 1  # Simulating a movement at index 3
    #     movement_detected, movement_inds, movements = self.explainer.detect_movement(prev_edges, edges)
    #     self.assertTrue(movement_detected)
    #     self.assertEqual(movement_inds.item(), 3)
    #     self.assertEqual(movements.shape, (2, self.explainer.num_nodes))
    
    # def test_onehot(self):
    #     ind = 7
    #     result = self.explainer.onehot(ind)
    #     self.assertEqual(result[ind].item(), 1)
    #     self.assertEqual(result.sum().item(), 1)  # Only one element should be 1
    
    # def test_add_movements(self):
    #     true_movements = {}
    #     movement_inds = torch.tensor([2, 5])
    #     movements = torch.zeros(2, self.explainer.num_nodes)
    #     movements[0, 2] = 1  # Initial state at index 2
    #     movements[1, 5] = 1  # Moved to index 5
    #     self.explainer.add_movements(true_movements, movement_inds, movements, 1)
    #     self.assertIn(2, true_movements)
    #     self.assertIn(5, true_movements)
    #     self.assertEqual(len(true_movements[2]), 1)
    #     self.assertEqual(len(true_movements[5]), 1)
    
    # def test_logger(self):
    #     logger = self.explainer.logger
    #     data = {"test": "value"}
    #     logger.log(data)
    #     log_files = os.listdir(logger.active_log_fold)
    #     self.assertGreater(len(log_files), 0)  # Ensure a log file is created

    # def test_model_infer(self):
    #     dummy_routine = {
    #         "edges": torch.zeros(1, self.explainer.num_nodes, self.explainer.num_nodes),
    #         "nodes": torch.zeros(1, self.explainer.num_nodes, self.explainer.num_nodes),
    #         "context_time": torch.zeros(1, 14),
    #         "y_edges": torch.zeros(1, self.explainer.num_nodes, self.explainer.num_nodes),
    #         "y_nodes": torch.zeros(1, self.explainer.num_nodes, self.explainer.num_nodes),
    #         "dynamic_edges_mask": torch.zeros(1, self.explainer.num_nodes, self.explainer.num_nodes),
    #         "time": torch.zeros(1),
    #         "change_type": torch.zeros(1, self.explainer.num_nodes)
    #     }
    #     input_tensor, output_tensor, gt_tensor, edge_probs = self.explainer.model_infer(dummy_routine)
    #     self.assertEqual(input_tensor.shape, (self.explainer.num_nodes,))
    #     self.assertEqual(output_tensor.shape, (self.explainer.num_nodes,))
    #     self.assertEqual(gt_tensor.shape, (self.explainer.num_nodes,))
    #     self.assertEqual(edge_probs.shape, (1, self.explainer.num_nodes, self.explainer.num_nodes))

if __name__ == "__main__":
    unittest.main()
