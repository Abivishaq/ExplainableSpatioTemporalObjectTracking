import torch
import sys
sys.path.append('helpers')

from adict import adict

from GraphTranslatorModule import GraphTranslatorModule
import yaml
import pickle
import os

from reader import RoutinesDataset
from encoders import TimeEncodingOptions


import numpy as np
from copy import deepcopy

def analyze_routine_in_window(routines_in_window):
    print("Routines in window")
    print("type", type(routines_in_window))
    raise NotImplementedError

class Explainer:
    def __init__(self):
        with open("model_configs.pkl", "rb") as f:
            model_configs = pickle.load(f)
        ckpt_file = "logs_default/ours_50epochs/epoch=49-step=162749.ckpt"
        self.model = GraphTranslatorModule.load_from_checkpoint(ckpt_file, model_configs = model_configs)
        cfg = model_configs
        train_days = 30
        time_options = TimeEncodingOptions(cfg['DATA_INFO']['weeekend_days'] if 'weeekend_days' in cfg['DATA_INFO'].keys() else None)
        time_encoding = time_options(cfg['time_encoding'])

        data_dir = 'data/HOMER/household0/'
        data = RoutinesDataset(data_path=os.path.join(data_dir,'processed'), 
                                time_encoder=time_encoding, 
                                batch_size=cfg['batch_size'],
                                max_routines = (train_days, None))
        self.data = data
        self.lookahead_steps = 3
        self.confidences = [0.0]
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda: self.model.to('cuda')
        else: print(f'Learned Model NOT USING CUDA. THIS WILL TAKE AGESSSSS!!!!!!!!!!!!')
        self.lookahead_steps = 3


    def model_infer(self,routine):
        if self.use_cuda:
            for k in routine.keys():
                routine[k] = routine[k].cuda()
        _, details,_ = self.model.step(routine)
        # print("details", details.keys())
        # print("details_output", details['output'].keys())
        # print("details_evaluate_node", details['evaluate_node'])
        # print("details_output_location", details['output']['location'].shape)
        gt_tensor = details['gt']['location']#[details['evaluate_node']].cpu()
        output_tensor = details['output']['location']#[details['evaluate_node']].cpu()
        # output_probs = details['output_probs']['location']#[details['evaluate_node']].cpu()
        input_tensor = details['input']['location']#[details['evaluate_node']].cpu()
        edge_probs = details['output_probs']['location'].to(torch.float32)
        # print("gt_tensor_shape", gt_tensor.shape)
        # print("output_tensor_shape", output_tensor.shape)
        # print("input_tensor_shape", input_tensor.shape)
        # print("routine_shape", routine['edges'].shape)
        # print("output_probs_shape", output_probs.shape)
        # raise NotImplementedError
        return input_tensor, output_tensor, gt_tensor, edge_probs

    def infer_runner(self):
        # test routine
        test_routines = self.data.test_routines
        print(type(test_routines))
        # raise NotImplementedError

        # loop through test_routines
        for (day_routine, additonal_info) in test_routines:
            routine_length = len(day_routine)
            # loop through routine
            true_movements = {}
            for i, step_routine in enumerate(day_routine):
                curr_routine = test_routines.collate_fn([step_routine])
                # keys: ['edges', 'nodes', 'context_time', 'y_edges', 'y_nodes', 'dynamic_edges_mask', 'time', 'change_type']
                # print("curr_routine_edges_shape", curr_routine['edges'].shape)
                # print("curr_routine_nodes_shape", curr_routine['nodes'].shape)
                # print("curr_routine_context_time_shape", curr_routine['context_time'].shape)
                # print("context_time:", curr_routine['context_time'])
                
                # calculate with time encoding:
                # 
                # time_tensor_float_64 = torch.tensor([curr_routine['time']], dtype=torch.float64)
                encoded_time = self.data.time_encoder(time_tensor_float_64)
                print("time  precision", curr_routine['time'].dtype)
                print("encoded_time:", encoded_time)
                diff = encoded_time - curr_routine['context_time']
                diff_sum = torch.sum(diff)
                print("diff_sum:", diff_sum)
                # print("curr_routine_y_edges_shape", curr_routine['y_edges'].shape)
                # print("curr_routine_y_nodes_shape", curr_routine['y_nodes'].shape)
                # print("curr_routine_dynamic_edges_mask_shape", curr_routine['dynamic_edges_mask'].shape)
                # print("curr_routine_time_shape", curr_routine['time'].shape)
                print("time:", curr_routine['time'])
                # print("curr_routine_change_type_shape", curr_routine['change_type'].shape)
                # print nodes
                for i in range(108):
                    one_hot=curr_routine["nodes"][0,i,:]
                    # verify that the one hot encoding is correct
                    assert torch.sum(one_hot)==1
                    assert torch.max(one_hot)==1
                    assert torch.min(one_hot)==0
                    # print value
                    ind = torch.argmax(one_hot)
                    # print(f'Node {i} has value {ind}')
                    assert ind==i
                continue
                raise NotImplementedError


                if i>0:
                    curr_routine['edges'] = prev_edges
                inp, pred, gt, prev_edges = self.model_infer(curr_routine)
                movement_detected, movement_inds, movements = self.detect_movement(inp, pred)
                if movement_detected:
                    print(f'Movement detected at step {step}')
                    # Do perturbation test
                    self.perturbation_test(curr_routine, pred, true_movements, window_pred_movements,movements)
                    
                else:
                    print(f'No movement detected at step {step}')
                    
                self.add_movements(window_pred_movements, movement_inds, movements, step)
                if i==0:
                    # decode the labels
                    curr_step_edge = self.label_coding(curr_routine['edges'])
                    next_step_edge = self.label_coding(curr_routine['y_edges'])
                    movement_detected, movement_inds, movements = self.detect_movement(curr_step_edge, next_step_edge)
                    self.add_movements(true_movements, movement_inds, movements, step)
            break
                
            



def main():
    explainer = Explainer()
    explainer.infer_runner()

if __name__ == "__main__":
    main()
        