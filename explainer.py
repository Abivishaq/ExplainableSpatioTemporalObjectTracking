import torch
import sys
# sys.path.append('helpers')

from adict import adict

from GraphTranslatorModule import GraphTranslatorModule
import yaml
import pickle
import os

from helpers.reader import RoutinesDataset
from helpers.encoders import TimeEncodingOptions


import numpy as np
from copy import deepcopy
import os

def analyze_routine_in_window(routines_in_window):
    print("Routines in window")
    print("type", type(routines_in_window))
    raise NotImplementedError


class Logger:
    def __init__(self):
        self.log_pth = 'logs'
        self.log_file = 'logs/log_no.txt'
        if not os.path.exists(self.log_pth):
            os.makedirs(self.log_pth)
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                f.write('0')
                self.run_no = 0
        else:
            with open(self.log_file, 'r') as f:
                self.run_no = int(f.read())
        with open(self.log_file, 'w') as f:
            f.write(str(self.run_no+1))
        self.active_log_fold = os.path.join(self.log_pth, f'run_{self.run_no}')
        assert not os.path.exists(self.active_log_fold)
        os.makedirs(self.active_log_fold)
        self.curr_log_file_no = 1
    def log(self, data):
        self.curr_log_file = os.path.join(self.active_log_fold, f'log_{self.curr_log_file_no}.pt')
        torch.save(data, self.curr_log_file)
        self.curr_log_file_no += 1
    


        

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
        self.logger = Logger()

        data_dir = 'data/HOMER/household0/'
        data = RoutinesDataset(data_path=os.path.join(data_dir,'processed'), 
                                time_encoder=time_encoding, 
                                batch_size=cfg['batch_size'],
                                max_routines = (train_days, None))
        self.data = data
        self.lookahead_steps = 1
        self.confidences = [0.0]
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda: self.model.to('cuda')
        else: print(f'Learned Model NOT USING CUDA. THIS WILL TAKE AGESSSSS!!!!!!!!!!!!')
        self.lookahead_steps = 3
        self.num_nodes = 108
    
    def label_coding(self, onehot_tensor):
        assert len(onehot_tensor.shape) == 3
        assert onehot_tensor.shape[0] == 1
        assert onehot_tensor.shape[2] == self.num_nodes
        assert onehot_tensor.shape[1] == self.num_nodes
        #print("onehot_tensor_shape", onehot_tensor.shape)
        # Does the inverse of onehot encoding
        # raise NotImplementedError
        tnsr = torch.argmax(onehot_tensor, dim=2)
        tnsr = tnsr.squeeze(0)
        # print("tnsr_shape", tnsr.shape)
        # raise NotImplementedError
        # print("tnsr_shape", tnsr.shape)
        # raise NotImplementedError
        assert len(tnsr.shape) == 1
        assert tnsr.shape[0] == self.num_nodes
        return tnsr

    def detect_movement(self, prev_edges_unonehot, edges_unonehot):

        # shape expectations:
        # Input:
        # prev_edges_unonehot: [108]
        # edges_unonehot: [108]
        #
        # Output:
        # movement_detected: bool
        # movement_inds: [num_movements]
        # movements: [2, 108]
        print("prev_edges_unonehot_shape", prev_edges_unonehot.shape)
        print("edges_unonehot_shape", edges_unonehot.shape)
        
        num_nodes = self.num_nodes
        assert prev_edges_unonehot.shape[0] == num_nodes
        assert edges_unonehot.shape[0] == num_nodes
        assert len(prev_edges_unonehot.shape) == 1
        assert len(edges_unonehot.shape) == 1




        diff = edges_unonehot - prev_edges_unonehot
        movement_inds = torch.nonzero(diff).squeeze(1)
        movement_detected = movement_inds.shape[0] > 0
        # movements = torch.cat((prev_edges_unonehot.unsqueeze(1), edges_unonehot.unsqueeze(1)), dim=1)
        movements = torch.stack((prev_edges_unonehot, edges_unonehot))
        
        # if movement_detected:
        #     # print(f'Movements shape: {movement_inds.shape}')
        #     # print("movement_inds", movement_inds)
        #     # print("Movements_inds_shape", movement_inds.shape)
        #     # print("movement_inds", movement_inds)
        #     movement_inds = movement_inds.squeeze(1)
        #     # print(f'Movement detected')
        #     # print(f'Movement indices: {movement_inds}')
        #     # print(f'Movements: {movements}')
        print("movements_shape", movements.shape)
        print("movement detected", movement_detected)
        print("movement_inds_shape", movement_inds.shape)
        print("##################")
        assert movements.shape[0] == 2
        assert movements.shape[1] == num_nodes
        assert len(movement_inds.shape) == 1
        

        return movement_detected, movement_inds, movements


    def model_infer(self,routine):
        # Routine is the dictionary of the routine at a given step
        # dictionary keys: ['edges', 'nodes', 'context_time', 'y_edges', 'y_nodes', 'dynamic_edges_mask', 'time', 'change_type']
        # shapes:
        # edges: [1, 108, 108]
        # nodes: [1, 108, 108]
        # context_time: [1, 14]
        # y_edges: [1, 108, 108]
        # y_nodes: [1, 108, 108]
        # dynamic_edges_mask: [1, 108, 108]
        # time: [1]
        # change_type: [1, 108]
        

        if self.use_cuda:
            for k in routine.keys():
                routine[k] = routine[k].cuda()
        _, details,_ = self.model.step(routine)
        
        gt_tensor = details['gt']['location']#[details['evaluate_node']].cpu()
        output_tensor = details['output']['location']#[details['evaluate_node']].cpu()
        # output_probs = details['output_probs']['location']#[details['evaluate_node']].cpu()
        input_tensor = details['input']['location']#[details['evaluate_node']].cpu()
        edge_probs = details['output_probs']['location'].to(torch.float32)

        # return shapes:
        # input_tensor: [1, 108]
        # output_tensor: [1, 108]
        # gt_tensor: [1, 108]
        # edge_probs: [1, 108, 108]
        
        assert input_tensor.shape[1] == self.num_nodes
        assert output_tensor.shape[1] == self.num_nodes
        assert gt_tensor.shape[1] ==  self.num_nodes
        assert edge_probs.shape[1] == self.num_nodes
        assert edge_probs.shape[2] == self.num_nodes

        # reducing shape to [108]
        input_tensor = input_tensor.squeeze(0)
        output_tensor = output_tensor.squeeze(0)
        gt_tensor = gt_tensor.squeeze(0)

        # asserting shapes len = 1
        assert len(input_tensor.shape) == 1
        assert len(output_tensor.shape) == 1
        assert len(gt_tensor.shape) == 1
        assert len(edge_probs.shape) == 3

        assert input_tensor.shape[0] == 108
        assert output_tensor.shape[0] == 108
        assert gt_tensor.shape[0] == 108
        assert edge_probs.shape == (1, 108, 108)

        # return shapes:
        # input_tensor: [108]
        # output_tensor: [108]
        # gt_tensor: [108]
        # edge_probs: [1, 108, 108]

        return input_tensor, output_tensor, gt_tensor, edge_probs

    def add_movements(self, true_movements, movement_inds, movements, step):
        print("movement_inds", movement_inds)
        print("movements_shape", movements.shape)
        # raise NotImplementedError
        for ind in movement_inds:
            if ind.item() not in true_movements.keys():
                true_movements[ind.item()] = []
            true_movements[ind.item()].append(movements[0,ind.item()])

    def detect_pred_diff(self, pred_true, pred, movement_mask):
        diff = pred - pred_true
        movement_detected = diff[movement_mask]
        movement_detected = movement_detected!=0
        return movement_detected
    def onehot(self, ind):
        onehot = torch.zeros(self.num_nodes)
        onehot[ind] = 1
        return onehot
    def print_tensor(self, tensor, ind=None):
        tensor = tensor[0]
        for i in range(tensor.shape[0]):
            if ind is not None:
                if i!=ind:
                    continue
            print("row", i, end = ":")
            for j in range(tensor.shape[1]):
                num = tensor[i,j].item()
                # convert to 2 decimal places
                num = round(num, 2)
                print(num, end = " ")
            print()

    def perturbation_test(self, curr_routine, pred_true, pred_prob, historic_movements, pred_movements,curr_movements,curr_movement_inds):
        # inputs:
        # curr_routine: input data for the model: 
        # dictionary keys: ['edges', 'nodes', 'context_time', 'y_edges', 'y_nodes', 'dynamic_edges_mask', 'time', 'change_type']
        #
        # pred_true: Original predicted output for reference : 
        # Tensor [108]
        #
        # Pred_prob: Original predicted output probabilities
        #
        # historic_movements: true movements that happened in the past :
        # dictionary: {obj1: [[prev_pos1, prev_pos2, ...], obj2: [prev_pos1, prev_pos2, ...]} 
        # 
        # pred_movements: in the window we are trying to predict, all the historic movements.
        # dictionary: {obj1: [[prev_pos1, prev_pos2, ...], obj2: [prev_pos1, prev_pos2, ...]}
        #  
        # curr_movements: movements that happened in the current step ie. pred_graph - curr_graph
        # Tensor [2, 108]
        #
        # curr_movement_inds: indices of the movements that happened in the current step :
        # Tensor [num_movements] 
        
        # print types
        # print("type curr_rooutine", type(curr_routine))
        # print("keys curr_routine", curr_routine.keys())
        # print("type pred_true", type(pred_true))
        # print("pred_true_shape", pred_true.shape)
        # print("type historic_movements", type(historic_movements))
        # print("keys historic_movements", historic_movements.keys())
        # print("type pred_movements", type(pred_movements))
        # print("keys pred_movements", pred_movements.keys())
        # print("type curr_movements", type(curr_movements))
        # print("curr_movements_shape", curr_movements.shape)
        # print("type curr_movement_inds", type(curr_movement_inds))
        # print("curr_movement_inds_shape", curr_movement_inds.shape)
        # raise NotImplementedError

            

        # outputs:
        # [curr_graph, predicted movements, movements that affect]
        # curr_gragh: Adjanency matrix
        # predicted movements: {obj1: [curr_pose, pred_pose], obj2: [curr_pose, pred_pose],  .... }
        # influential_movements: {obj1: [[influential_obj1, old_pose, new_pose],[influential_obj2, old_pose, new_pose], .... ], obj2: [...]}
        curr_graph = self.label_coding(curr_routine['edges'])
        predicted_movements = {}
        influential_movements = {}
        # time: 

        print("--------------------------------------------------")
        for mov in curr_movement_inds:
            print("num_of_historic_movements", len(historic_movements.keys()))
            print("num_of_pred_movements", len(pred_movements.keys()))
            mov_key = mov.item()
            predicted_movements[mov_key] = [curr_movements[0,mov], curr_movements[1,mov]]
            influential_movements[mov_key] = []
            for obj in historic_movements.keys():
                tmp = curr_routine['edges'][0,obj,:]
                # curr_routine['edges'][0,obj,:] = self.onehot(curr_movements[1,mov])
                for obj_mov in historic_movements[obj]:
                    curr_routine['edges'][0,obj,:] = self.onehot(obj_mov)
                    inp, pred, gt, out_probs = self.model_infer(curr_routine)
                    # print("pred_shape", pred.shape)
                    # print("pred_true_shape", pred_true.shape)
                    # raise NotImplementedError
                    # is the movement influential
                    
                    # if(pred_true[mov]  == pred[mov]):
                    if True: ### TODO: Is no filtering fine?
                        influence_level = pred_prob[0,mov,pred_true[mov]] - out_probs[0,mov,pred_true[mov]]

                        influential_movements[mov_key].append([obj, obj_mov, curr_movements[1,mov], influence_level])
                curr_routine['edges'][0,obj,:] = tmp
            for obj in pred_movements.keys():
                tmp = curr_routine['edges'][0,obj,:]
                # curr_routine['edges'][0,obj,:] = self.onehot(curr_movements[1,mov])
                for obj_mov in pred_movements[obj]:
                    curr_routine['edges'][0,obj,:] = self.onehot(obj_mov)
                    inp, pred, gt, out_probs = self.model_infer(curr_routine)
                    # raise NotImplementedError
                    # is the movement influential
                    # if(pred_true[mov]  == pred[mov]):
                    if True: ### TODO: Is no filtering fine?
                        # print("mov:", mov)
                        # print("pred_probs:")
                        # self.print_tensor(pred_prob,mov)
                        # print("out_probs:")
                        # self.print_tensor(out_probs,mov)
                        
                        influence_level = pred_prob[0,mov,pred_true[mov]] - out_probs[0,mov,pred_true[mov]]
                        print("influence_level", influence_level)
                        # raise NotImplementedError
                        influential_movements[mov_key].append([obj, obj_mov, curr_movements[1,mov], influence_level,pred_prob[0,mov,:],out_probs[0,mov,:]])
                curr_routine['edges'][0,obj,:] = tmp
        
            data = [curr_graph, predicted_movements, influential_movements]
            self.logger.log(data)
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print("--------------------------------------------------")
        # time perturbations:
        
        return curr_graph, predicted_movements, influential_movements

        

    def infer_runner(self):
        # test routine
        test_routines = self.data.test_routines
        # test routine is of datatype -> <class 'reader.DataSplit'>
        print(type(test_routines))
        # raise NotImplementedError

        # loop through test_routines
        for (day_routine, additonal_info) in test_routines:
            # About day_routine: probably record of graph through the day
            # day_routine[i] will return:  [prev_edges, prev_nodes, encoded_time, edges, nodes, self.active_edges, tensor(time), change_type]
            routine_length = len(day_routine)

            # loop through routine
            historic_movements = {}
            print("routine_length", routine_length)
            # raise NotImplementedError
            for step in range(routine_length):
                
                routines_in_window = [test_routines.collate_fn([day_routine[j]]) for j in range(step, min(step+self.lookahead_steps, routine_length))]
                # collate_fn: will wrap up day_routine[j] into a dictionary
                # routines_in_window is a list of dictionaries
                # each dictionary is a step in the routine with keys: ['edges', 'nodes', 'context_time', 'y_edges', 'y_nodes', 'dynamic_edges_mask', 'time', 'change_type']
                # shapes:
                # edges: [1, 108, 108]
                # nodes: [1, 108, 108]
                # context_time: [1, 14]
                # y_edges: [1, 108, 108]
                # y_nodes: [1, 108, 108]
                # dynamic_edges_mask: [1, 108, 108]
                # time: [1]
                # change_type: [1, 108]

                ############ tmps analysis ###########
                # printing the routines_in_window[0] shapes 
                # print("routines_in_window[0] keys", routines_in_window[0].keys())
                # print("routines_in_window[0]['edges']", routines_in_window[0]['edges'].shape)
                # print("routines_in_window[0]['nodes']", routines_in_window[0]['nodes'].shape)
                # print("routines_in_window[0]['context_time']", routines_in_window[0]['context_time'].shape)
                # print("routines_in_window[0]['y_edges']", routines_in_window[0]['y_edges'].shape)
                # print("routines_in_window[0]['y_nodes']", routines_in_window[0]['y_nodes'].shape)
                # print("routines_in_window[0]['dynamic_edges_mask']", routines_in_window[0]['dynamic_edges_mask'].shape)
                # print("routines_in_window[0]['time']", routines_in_window[0]['time'].shape)
                # print("routines_in_window[0]['change_type']", routines_in_window[0]['change_type'].shape)

                # analyze_routine_in_window(routines_in_window)
                # print("type day_routine[step]", type(day_routine[step]))
                # print("len day_routine[step]", len(day_routine[step]))
                # print("day_routine[step]", day_routine[step])
                
                # raise NotImplementedError
                ########################################

                window_pred_movements = {}
                for i, curr_step in enumerate(routines_in_window):
                    assert i<self.lookahead_steps
                    if i>0:
                        curr_step['edges'] = prev_edges
                    inp, pred, gt, prev_edges = self.model_infer(curr_step)
                    ############### tmp analysis ################
                    # verify that the shapes are correct
                    print("inp_shape", inp.shape)
                    print("pred_shape", pred.shape)
                    print("gt_shape", gt.shape)
                    print("prev_edges_shape", prev_edges.shape)
                    # raise NotImplementedError

                    ##################################################
                    movement_detected, movement_inds, movements = self.detect_movement(inp, pred)
                    if movement_detected:
                        print(f'Movement detected at step {step}')
                        # Do perturbation test
                        pred_prob = prev_edges
                        self.perturbation_test(curr_step, pred, pred_prob, historic_movements, window_pred_movements,movements, movement_inds)
                        
                    else:
                        print(f'No movement detected at step {step}')
                        
                    self.add_movements(window_pred_movements, movement_inds, movements, step)
            
                # check for movements in histry and update historic movements
                curr_step_edge = self.label_coding(curr_step['edges'])
                next_step_edge = self.label_coding(curr_step['y_edges'])

                movement_detected, movement_inds, movements = self.detect_movement(curr_step_edge, next_step_edge)
                self.add_movements(historic_movements, movement_inds, movements, step)
            break
        # run inference on each routine


# class tester:
#     def __init__(self):
#         self.explainer = Explainer()
#     def test_movements

def main():
    explainer = Explainer()
    explainer.infer_runner()

    # test movements functions
    # x = [1,2,9,4,5]
    # y = [1,2,3,4,6]
    # x = torch.tensor(x)
    # y = torch.tensor(y)
    # movement_detected, movement_inds, movements = explainer.detect_movement(x, y)
    # print("movement_detected", movement_detected)
    # print("movement_inds", movement_inds)
    # print("movements", movements)

if __name__ == "__main__":
    main()
        