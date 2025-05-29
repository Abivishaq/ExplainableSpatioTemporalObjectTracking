import pickle
from GraphTranslatorModule import GraphTranslatorModule
from helpers.encoders import TimeEncodingOptions
from helpers.reader import RoutinesDataset
import os
import torch


class ProactovityModule:
    def __init__(self, name):
        self.name = name

    def action_predictor(self, state):
        """
        Predicts the next action based on the current state.
        Args:
            state (dict): The current state of the environment.
        Returns:
            action (list): The predicted action. list of dictonaries. each dictionary mentions how an object is supposed to change (location or/and state).
            info_for_explainer (dict): Information for the explainer.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    def explainer(self, state):
        """
        Explains the predicted action based on the current state.
        Args:
            state (dict): The current state of the environment.
        Returns:
            explanation (str): The explanation for the predicted action.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    
class STOT(ProactovityModule):
    def __init__(self):
        name = "STOT"
        super().__init__(name)
        self.name = name
        with open("model_configs.pkl", "rb") as f:
            model_configs = pickle.load(f)
        ckpt_file = "logs_default/ours_50epochs/epoch=49-step=162749.ckpt"
        self.model = GraphTranslatorModule.load_from_checkpoint(ckpt_file, model_configs = model_configs)
        cfg = model_configs
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda: self.model.to('cuda')
        else: print(f'Learned Model NOT USING CUDA. THIS MAY TAKE AGESSSSS!!!!!!!!!!!!')


       

        # dummy state stuff 
        # TODO: Remove after implementing the state_to_stot_state function
        time_options = TimeEncodingOptions(cfg['DATA_INFO']['weeekend_days'] if 'weeekend_days' in cfg['DATA_INFO'].keys() else None)
        self.time_encoding = time_options(cfg['time_encoding'])
        train_days = 30
        data_dir = 'data/HOMER/household0/'

        self.data = RoutinesDataset(data_path=os.path.join(data_dir,'processed'), 
                                time_encoder=self.time_encoding, 
                                batch_size=cfg['batch_size'],
                                max_routines = (train_days, None))
        
        self.lookahead_steps = 1
        self.num_nodes = 108
        self.node_name = torch.load("node_classes.pt")
    
    def state_to_stot_state(self, state):
        # NOTE: Remove the corresponding init function part that initializes the dataset
        # dummy function that returns a specific state
        test_routines = self.data.test_routines
        (day_routine, additonal_info) =  test_routines[0]
        
        # About day_routine: probably record of graph through the day
        # day_routine[i] will return:  [prev_edges, prev_nodes, encoded_time, edges, nodes, self.active_edges, tensor(time), change_type]
        routine_length = len(day_routine)

        # loop through routine
        historic_movements = {}

        step =0   
        routines_in_window = [test_routines.collate_fn([day_routine[j]]) for j in range(step, min(step+self.lookahead_steps, routine_length))]

        return routines_in_window[0]


    def action_predictor(self, state):
        """
        Predicts the next action based on the current state.
        Args:
            state (dict): The current state of the environment.
        Returns:
            action (list): The predicted action. list of dictonaries. each dictionary mentions how an object is supposed to change (location or/and state).
            info_for_explainer (dict): Information for the explainer.
        """
        # Implement the logic for predicting actions based on the current state
        routine = self.state_to_stot_state(state)
        # Routines is a dictionary of the routine at a given step
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
        input_tensor = details['input']['location'] #[details['evaluate_node']].cpu()
        prev_edges = details['output_probs']['location'].to(torch.float32) #[details['evaluate_node']].cpu()
        
        gt_tensor = details['gt']['location']#[details['evaluate_node']].cpu()
        output_tensor = details['output']['location']#[details['evaluate_node']].cpu()
        # output_probs = details['output_probs']['location']#[details['evaluate_node']].cpu()
        edge_probs = details['output_probs']['location'].to(torch.float32)

    

        # reducing shape to [108]
        input_tensor = input_tensor.squeeze(0)
        output_tensor = output_tensor.squeeze(0)
        gt_tensor = gt_tensor.squeeze(0)

        return [input_tensor, output_tensor, gt_tensor, edge_probs], None


    def explainer(self, state):
        """
        Explains the predicted action based on the current state.
        Args:
            state (dict): The current state of the environment.
        Returns:
            explanation (str): The explanation for the predicted action.
        """
        # Implement the logic for explaining the predicted action
        pass

if __name__ == "__main__":
    # Example usage
    stot = STOT()
    state = {}  # Replace with actual state
    action, info_for_explainer = stot.action_predictor(state)
    explanation = stot.explainer(state)
    print("Predicted Action:", action)
    print("Explanation:", explanation)