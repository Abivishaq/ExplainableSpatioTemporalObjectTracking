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
# print(f'Movement predicted at step {step}')   

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
        self.lookahead_steps = 1
        self.num_nodes = 108
        self.node_name = torch.load("node_classes.pt")
    
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
        # Input shapes:
        # prev_edges_unonehot: [108]
        # edges_unonehot: [108]
        #
        # Output shapes:
        # movement_detected: bool
        # movement_inds: [num_movements]
        # movements: [2, 108]
        
        num_nodes = self.num_nodes
        assert prev_edges_unonehot.shape[0] == num_nodes
        assert edges_unonehot.shape[0] == num_nodes
        assert len(prev_edges_unonehot.shape) == 1
        assert len(edges_unonehot.shape) == 1




        diff = edges_unonehot - prev_edges_unonehot
        movement_inds = torch.nonzero(diff).squeeze(1)
        movement_inds = movement_inds.tolist()
        movement_detected = len(movement_inds) > 0
        movements = torch.stack((prev_edges_unonehot, edges_unonehot))
        assert movements.shape[0] == 2
        assert movements.shape[1] == num_nodes
        

        return movement_detected, movement_inds, movements


    def model_infer(self,routines, steps = 1):
        # Routines is a list of dictionary of the routine at a given step
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
        assert len(routines) == steps
        # print("routines_shape", len(routines))
        # print("type(routines)", type(routines))
        # print("routines[0].keys()", routines[0].keys())
        # raise NotImplementedError

        for i, routine in enumerate(routines):
            if i != 0:
                routine['edges'] = prev_edges
            if self.use_cuda:
                for k in routine.keys():
                    routine[k] = routine[k].cuda()
            _, details,_ = self.model.step(routine)
            if i == 0:
                input_tensor = details['input']['location']#[details['evaluate_node']].cpu()
            prev_edges = details['output_probs']['location'].to(torch.float32)#[details['evaluate_node']].cpu()
        
        gt_tensor = details['gt']['location']#[details['evaluate_node']].cpu()
        output_tensor = details['output']['location']#[details['evaluate_node']].cpu()
        # output_probs = details['output_probs']['location']#[details['evaluate_node']].cpu()
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
        for ind in movement_inds:
            if ind not in true_movements.keys():
                true_movements[ind] = []
            if movements[0,ind] not in true_movements[ind]:
                true_movements[ind].append(movements[0,ind])


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

    def perturbation_test(self, curr_routine_window, pred_true, pred_prob, historic_movements, curr_transitions,change_inds):
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
        # curr_transitions: movements that happened in the current step ie. pred_graph - curr_graph
        # Tensor [2, 108]
        #
        # change_inds: indices of the movements that happened in the current step :
        # Tensor [num_movements]   

        # outputs:
        # [curr_graph, predicted movements, movements that affect]
        # curr_gragh: Adjanency matrix
        # predicted movements: {obj1: [curr_pose, pred_pose], obj2: [curr_pose, pred_pose],  .... }
        # influential_movements: {obj1: [[influential_obj1, old_pose, new_pose],[influential_obj2, old_pose, new_pose], .... ], obj2: [...]}
        # print("Number of historic movements", len(historic_movements.keys()))
        # print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        curr_graph = self.label_coding(curr_routine_window[0]['edges'])
        predicted_changes = {}
        influential_movements = {}
        for chg_ind in change_inds:
            # print("num_of_historic_movements", len(historic_movements.keys()))
            # print("num_of_pred_movements", len(pred_movements.keys()))
            # chg_ind = mov
            predicted_changes[chg_ind] = [curr_transitions[0,chg_ind], curr_transitions[1,chg_ind]]
            influential_movements[chg_ind] = []
            for obj in historic_movements.keys():
                tmp = curr_routine_window[0]['edges'][0,obj,:]
                obj_curr_pos = torch.argmax(tmp).item()
                # print("obj_curr_pos", obj_curr_pos)
                # raise NotImplementedError
                # curr_routine['edges'][0,obj,:] = self.onehot(curr_transitions[1,chg_ind])
                for obj_mov in historic_movements[obj]:
                    curr_routine_window[0]['edges'][0,obj,:] = self.onehot(obj_mov)
                    inp, pred, gt, out_probs = self.model_infer(curr_routine_window,len(curr_routine_window))
                    # print("pred_shape", pred.shape)
                    # print("pred_true_shape", pred_true.shape)
                    # raise NotImplementedError
                    # is the movement influential
                    
                    # if(pred_true[mov]  == pred[mov]):
                    if True: ### TODO: Is no filtering fine?
                        influence_level = pred_prob[0,chg_ind,pred_true[chg_ind]] - out_probs[0,chg_ind,pred_true[chg_ind]]

                        influential_movements[chg_ind].append([obj, obj_mov, obj_curr_pos, influence_level, pred_prob[0,chg_ind,:], out_probs[0,chg_ind,:]])
                curr_routine_window[0]['edges'][0,obj,:] = tmp
            # for obj in pred_movements.keys():
            #     raise ValueError("Logic error: pred movements should not be considered!!!")
            
            # Time perturbations


                
        data = [curr_graph, predicted_changes, influential_movements]
        self.logger.log(data)
        # raise NotImplementedError
        # time perturbations:
        
        return curr_graph, predicted_changes, influential_movements

        

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
                
                inp, pred, gt, prev_edges = self.model_infer(routines_in_window,len(routines_in_window))
                movement_detected, movement_inds, movements = self.detect_movement(inp, pred)
                if movement_detected:
                    # print(f'Movement predicted at step {step}')
                    # Do perturbation test
                    pred_prob = prev_edges
                    self.perturbation_test(routines_in_window, pred, pred_prob, historic_movements,movements, movement_inds)
                else:
                    # print(f'No movement detected at step {step}')
                    pass
                    
                # self.add_movements(window_pred_movements, movement_inds, movements, step)
        
                # check for movements in histry and update historic movements
                curr_step_edge = self.label_coding(routines_in_window[0]['edges'])
                next_step_edge = self.label_coding(routines_in_window[-1]['y_edges'])

                movement_detected, movement_inds, movements = self.detect_movement(curr_step_edge, next_step_edge)
                if movement_detected:
                    print(f'Movement occured at step {step}')
                    prct_mv_txt = ''
                    for ind in movement_inds:
                        prct_mv_txt += f'{self.node_name[ind]}+({ind}) : {self.node_name[movements[0,ind]]}+({movements[0,ind]}) -> {self.node_name[movements[1,ind]]}+({movements[1,ind]})\n'
                    print(prct_mv_txt)
                    # self.add_movements(historic_movements, movement_inds, movements, step)   
                    

                    self.add_movements(historic_movements, movement_inds, movements, step)
                    hst_mv_str = 'historic movements:\n'
                    for obj in historic_movements.keys():
                        hst_mv_str += f'{self.node_name[obj]}({obj}):'
                        for mv in historic_movements[obj]:
                            hst_mv_str += f'{self.node_name[mv]}+({mv}), '
                        hst_mv_str += '\n'
                    print(hst_mv_str)
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
        