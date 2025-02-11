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

    def model_infer(self,data):
        pass
    def infer_runner(self):
        





# with open('config/default.yaml') as f:
#         cfg = yaml.safe_load(f)

# model_configs = adict(cfg)
with open("model_configs.pkl", "rb") as f:
    model_configs = pickle.load(f)
ckpt_file = "logs_default/ours_50epochs/epoch=49-step=162749.ckpt"

model = GraphTranslatorModule.load_from_checkpoint(ckpt_file, model_configs = model_configs)
cfg = model_configs
train_days = 30
time_options = TimeEncodingOptions(cfg['DATA_INFO']['weeekend_days'] if 'weeekend_days' in cfg['DATA_INFO'].keys() else None)
time_encoding = time_options(cfg['time_encoding'])

data_dir = 'data/HOMER/household0/'
data = RoutinesDataset(data_path=os.path.join(data_dir,'processed'), 
                           time_encoder=time_encoding, 
                           batch_size=cfg['batch_size'],
                           max_routines = (train_days, None))

# model summ
# print(model)
def detect_movements(edges, prev_edges,is_one_hot = True):
    # differnce between edges and prev_edges
    # return torch.bitwise_xor(edges, prev_edges)
    # edges_unone_hot = torch.argmax(edges, dim=1)
    print("one_hot:")
    print(is_one_hot)
    
    if is_one_hot:
        edge_max = torch.max(edges, dim=1)
        edges_unone_hot = edge_max.indices
        edges_mask = edge_max.values > 0
        prev_edges_max = torch.max(prev_edges, dim=1)
        prev_edges_unone_hot = prev_edges_max.indices
        prev_edges_mask = prev_edges_max.values > 0
    else:
        edges_unone_hot = edges
        prev_edges_unone_hot = prev_edges
    
    diff = edges_unone_hot - prev_edges_unone_hot

    # Record movements
    print(torch.nonzero(diff))

    # prev_edges_unone_hot = torch.argmax(prev_edges, dim=1)
    # print(edge_max)
    # print("###########################")
    # print(edges_unone_hot)
    print("^^^^^^^^^^^^^^^^^^^")
    # diff = edges - prev_edges
    # print(edges)
    # print(diff)
    # detect non-zero elements
    movement_inds = torch.nonzero(diff)
    movement_detected = movement_inds.size(0) > 0
    # combine prev_edges and edges
    movements = torch.cat((prev_edges_unone_hot.unsqueeze(1), edges_unone_hot.unsqueeze(1)), dim=1)
    return movement_detected, movement_inds, movements

def get_perturbation(model, pred_movements, true_movements, data, original_output, movement_inds):
    mask = torch.zeros_like(original_output)
    for i in movement_inds:
        mask[i.item()] = 1
    # possible perturbations from  pred_movements
    valid_perturbations = []
    perturbed_input = data['edges'].clone()

    for i in pred_movements.keys():
        for j in pred_movements[i]:
            tmp = perturbed_input[i]
            perturbed_input[i] = j
            _, details, _ = model.step({'edges':perturbed_input.unsqueeze(0)})
            perturbed_input[i] = tmp
            output_tensor = details['output']['location'][details['evaluate_node']].cpu()
            diff = (output_tensor - original_output)*mask
            if torch.sum(diff) > 0:
                valid_perturbations.append((i,j))
    # possible perturbations from true_movements
    for i in true_movements.keys():
        for j in true_movements[i]:
            tmp = perturbed_input['edges'][i]
            perturbed_input['edges'][i] = j
            _, details, _ = model.step({'edges':perturbed_input.unsqueeze(0)})
            perturbed_input['edges'][i] = tmp
            output_tensor = details['output']['location'][details['evaluate_node']].cpu()
            diff = (output_tensor - original_output)*mask
            if torch.sum(diff) > 0:
                valid_perturbations.append((i,j))
    return valid_perturbations

def infer_building(model, test_routines, lookahead_steps=3, confidences = [0.0]):
    use_cuda = torch.cuda.is_available() #and learned_model
    if use_cuda: model.to('cuda')
    else: print(f'Learned Model NOT USING CUDA. THIS WILL TAKE AGESSSSS!!!!!!!!!!!!')

    # num_change_types = 3

    raw_data = {'inputs':[], 'outputs':[], 'ground_truths':[], 'futures':[], 'change_types':[]}
    results = {conf: 
                {'moved':{'correct':[0 for _ in range(lookahead_steps)], 
                            'wrong':[0 for _ in range(lookahead_steps)], 
                            'missed':[0 for _ in range(lookahead_steps)]},
                 'unmoved':{'fp':[0 for _ in range(lookahead_steps)], 
                            'tn':[0 for _ in range(lookahead_steps)]}
                } for conf in confidences}
    results_by_obj = [{'correct':[], 'wrong':[], 'missed':[], 'fp':[]} for _ in range(lookahead_steps)]
    object_stats = []
    figures = []
    figures_imp = []

    results['all_moves'] = []
    total_num_steps = 0

    for (routine, additional_info) in test_routines:

        routine_length = len(routine)
        total_num_steps += routine_length
        num_nodes = additional_info['active_nodes'].sum()

        routine_inputs = torch.empty(routine_length, num_nodes)
        routine_outputs = torch.ones(routine_length, num_nodes).to(int) * -1
        routine_output_step = torch.ones(routine_length, num_nodes).to(int) * (lookahead_steps)
        routine_ground_truths = torch.ones(routine_length, num_nodes).to(int) * -1
        routine_ground_truth_step = torch.ones(routine_length, num_nodes).to(int) * (lookahead_steps)
        routine_futures = torch.ones(routine_length, num_nodes) * -1
        routine_change_types = torch.zeros(routine_length, num_nodes).to(int)
        routine_outputs_conf = {c:{'output':torch.ones(routine_length, num_nodes).to(int) * -1, 'step':torch.ones(routine_length, num_nodes).to(int) * (lookahead_steps)} for c in confidences}

        changes_output_all = torch.zeros(routine_length, num_nodes).to(bool)
        changes_gt_all = torch.zeros(routine_length, num_nodes).to(bool)


        true_movements= {}
        for step in range(routine_length):
            data_list = [test_routines.collate_fn([routine[j]]) for j in range(step, min(step+lookahead_steps, routine_length))]
            movement_detected, movement_inds, movements  = detect_movements(data_list[0]['y_edges'], data_list[0]['edges'])
            if movement_detected:
                print("True movements")
                print(movements)
                print(movement_inds)
                for i in movement_inds[0]:
                    if i.item() in true_movements.keys():
                        true_movements[i.item()].append(movements[i.item()][0].item())
                    else:
                        true_movements[i.item()] = [movements[i.item()][0].item()]

                # raise NotImplementedError
                    
            pred_movements = {}
            for i,data in enumerate(data_list):
                assert i<lookahead_steps
                if i>0:
                    data['edges'] = prev_edges
                
                if use_cuda: 
                    for k in data.keys():
                        data[k] = data[k].to('cuda')
                        
                _, details, _ = model.step(data)
                print(details.keys())

                
                # print("########################################")
                # print(data.keys())
                # print(details.keys())
                # print(details['gt'].keys())
                # print(details['gt']['location'])
                # print(len(details['gt']['location'][0]))
                # print(details)
                # raise NotImplementedError
                # continue

                gt_tensor = details['gt']['location'][details['evaluate_node']].cpu()
                output_tensor = details['output']['location'][details['evaluate_node']].cpu()
                output_probs = details['output_probs']['location'][details['evaluate_node']].cpu()
                input_tensor = details['input']['location'][details['evaluate_node']].cpu()

                # detect movements
                movement_detected, movement_inds, movements = detect_movements(output_tensor, input_tensor,is_one_hot=False)
                if movement_detected:
                    print("%"*50)
                    # print("GT")
                    # print(gt_tensor)
                    print("Output")
                    print(output_tensor)
                    print("Input")
                    print(input_tensor)
                    print("Movements")
                    print(movements)
                    print("Movement indices")
                    print(movement_inds)
                    # apply perturbation
                    effective_perturbation = get_perturbation(model, pred_movements, true_movements, data,output_tensor, movement_inds)
                    print("Effective perturbation")
                    print(effective_perturbation)
                    if len(effective_perturbation) > 0:
                        print("Perturbation detected")
                        raise NotImplementedError
                    for i in movement_inds:
                        if i.item() in pred_movements.keys():
                            pred_movements[i.item()].append(movements[i.item()][0].item())
                        else:
                            pred_movements[i.item()] = [movements[i.item()][0].item()]
                if i == 0:
                    routine_inputs[step,:] = deepcopy(input_tensor)
                new_changes_out = deepcopy(np.bitwise_and(output_tensor != input_tensor , np.bitwise_not(changes_output_all[step,:]))).to(bool)
                new_changes_gt = deepcopy(np.bitwise_and(gt_tensor != routine_inputs[step,:] , np.bitwise_not(changes_gt_all[step,:]))).to(bool)
                
                if i == 0:
                    origins_one = np.arange(num_nodes)
                    object_stats += ([((o,int(d)),step) for o,d in zip(origins_one[(new_changes_gt).to(bool)], gt_tensor[(new_changes_gt).to(bool)])])
                routine_outputs[step, :][new_changes_out] = deepcopy(output_tensor[new_changes_out])
                routine_output_step[step, :][new_changes_out] = i
                routine_ground_truths[step, :][new_changes_gt] = deepcopy(gt_tensor[new_changes_gt])
                routine_ground_truth_step[step, :][new_changes_gt] = i
                # routine_change_types[step,:][new_changes_gt] = deepcopy(data['change_type'].cpu().to(int)[details['evaluate_node']])[new_changes_gt]
                routine_change_types[step, :][new_changes_gt] = deepcopy(data['change_type'].cpu().to(int)[details['evaluate_node'].cpu()])[new_changes_gt]
                changes_output_all[step,:] = deepcopy(np.bitwise_or(changes_output_all[step,:], new_changes_out)).to(bool)
                changes_gt_all[step,:] = deepcopy(np.bitwise_or(changes_gt_all[step,:], new_changes_gt)).to(bool)

                output_conf = (output_probs*(torch.nn.functional.one_hot(output_tensor, num_classes=output_probs.size()[-1]))).sum(-1)*(routine_inputs[step,:]!=output_tensor).to(float)
                
                for conf in confidences:
                    new_changes_conf = deepcopy(np.bitwise_and((output_conf > conf), np.bitwise_not(routine_outputs_conf[conf]['step'][step, :]<lookahead_steps))).to(bool)
                    routine_outputs_conf[conf]['output'][step, :][new_changes_conf] = deepcopy(output_tensor[new_changes_conf])
                    routine_outputs_conf[conf]['step'][step, :][new_changes_conf] = i
                

                # if deterministic_input_loop:
                #     prev_edges = one_hot(details['output']['location'], num_classes = details['output']['location'].size()[-1]).to(torch.float32)
                # else:
                #     prev_edges = (details['output_probs']['location']).to(torch.float32)
                prev_edges = (details['output_probs']['location']).to(torch.float32)
        # continue
        # raise NotImplementedError
        # results = update_confidence_based_metrics(routine_ground_truths, routine_ground_truth_step, routine_outputs_conf, confidences, lookahead_steps, results)
        # results_by_obj, results_other = update_additional_metrics(routine_outputs, routine_output_step, routine_ground_truths, routine_ground_truth_step, routine_change_types, changes_output_all, lookahead_steps, num_nodes, num_change_types, results_by_obj, results_other)
        
        # fig, axs = plt.subplots(1,5)
        # fig.set_size_inches(30,20)

        # labels = []
        # if len(node_names) > 0:
        #     labels = [name for name,active in zip(node_names, additional_info['active_nodes']) if active]

        moves = {}
        for i,act_node in enumerate(labels):
            any_changes = (routine_ground_truths[:,i]+routine_outputs[:,i])>0
            def get_name(idx):
                if idx>=0: return node_names[idx]
                else: return 'No Change'
            changes = [get_name(o)+'({})'.format(o_st)+ '  ' + get_name(gt)+'({})'.format(gt_st) for (o, o_st, gt, gt_st) in zip(routine_outputs[:,i][any_changes], routine_output_step[:,i][any_changes], routine_ground_truths[:,i][any_changes], routine_ground_truth_step[:,i][any_changes])]
            moves[act_node] = {'pred/actual':changes}

        results['all_moves'].append(moves)

        raw_data['inputs'].append(routine_inputs)
        raw_data['outputs'].append(routine_outputs)
        raw_data['ground_truths'].append(routine_ground_truths)
        raw_data['futures'].append(routine_futures)
        # raw_data['change_types'].append(routine_change_types)
    # raise NotImplementedError
    # transition_diff = {}
    # transitions = []
    # for tr,step in object_stats:
    #     if tr in transition_diff:
    #         transition_diff[tr].append(step)
    #     else:
    #         transitions.append(tr)
    #         transition_diff[tr] = [step]
    


    # obj_eval_figs = []
    # for ls in range(lookahead_steps):
    #     def prec(obj_dest_pair):
    #         return results_by_obj[ls]['correct'].count(obj_dest_pair)/(results_by_obj[ls]['correct'].count(obj_dest_pair)+results_by_obj[ls]['wrong'].count(obj_dest_pair)+results_by_obj[ls]['fp'].count(obj_dest_pair)+1e-8)
    #     def recl(obj_dest_pair):
    #         return results_by_obj[ls]['correct'].count(obj_dest_pair)/(results_by_obj[ls]['correct'].count(obj_dest_pair)+results_by_obj[ls]['wrong'].count(obj_dest_pair)+results_by_obj[ls]['missed'].count(obj_dest_pair)+1e-8)
    #     fig, axs = plt.subplots(1,2)
    #     precisions = [prec(tr) for tr in transitions]
    #     recalls = [recl(tr) for tr in transitions]
    #     axs[0].scatter([len(transition_diff[tr]) for tr in transitions], [np.std(transition_diff[tr]) for tr in transitions], color=[[1-p,0,p]for p in precisions])
    #     axs[1].scatter([len(transition_diff[tr]) for tr in transitions], [np.std(transition_diff[tr]) for tr in transitions], color=[[1-r,0,r]for r in recalls])
    #     # axs[0].scatter(prob, results_by_obj[ls][typ].reshape(-1)[by_prob], marker='x', color=colors_obj_eval[typ], label=typ)
    #     # axs[1].plot(var, results_by_obj[ls][typ].reshape(-1)[by_var], color=colors_obj_eval[typ], label=typ)
    #     axs[0].set_ylabel('Time Variability')
    #     axs[1].set_ylabel('Time Variability')
    #     axs[0].set_xlabel('Num. times object moved')
    #     axs[1].set_xlabel('Num. times object moved')
    #     axs[0].set_title(f'Precision-{ls}')
    #     axs[1].set_title(f'Recall-{ls}')
    #     obj_eval_figs.append(fig)

    results['num_steps'] = total_num_steps

    return results, raw_data #figures + figures_imp + obj_eval_figs


infer_building(model, data.test_routines, lookahead_steps=3, confidences = [0.0])