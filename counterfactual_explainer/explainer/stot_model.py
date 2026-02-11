import torch
import sys
import os
import json
from adict import adict

# torch.set_default_dtype(torch.float64)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from GraphTranslatorModule import GraphTranslatorModule
from helpers.encoders import TimeEncodingOptions

class STOTModel:
    def __init__(
        self,
        step_size,
        household_id=0
    ):
        self.weights_dir = os.path.join(os.path.dirname(__file__), '..','..','model_weights', f'household{household_id}')
        config_path = os.path.join(self.weights_dir, 'config.json')
        self.step_size = step_size
        weight_pt_file = os.path.join(self.weights_dir, f'weights.pt')
        # Load model config
        with open(config_path, "r") as f:
            self.model_configs = json.load(f)
            self.model_configs = adict(self.model_configs)

        # Load model checkpoint
        self.model = GraphTranslatorModule(self.model_configs)
        state_dict = torch.load(weight_pt_file, map_location=torch.device('cuda'))
        self.model.load_state_dict(state_dict)
        self.model.eval()  

        # print("Model loaded with configurations:", self.model_configs)

        # Time encoder
        weekend_days = self.model_configs['DATA_INFO'].get('weeekend_days', None)
        time_options = TimeEncodingOptions(weekend_days)
        self.time_encoder = time_options(self.model_configs['time_encoding'])


        self.num_nodes = 108
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.model.to("cuda")
        else:
            print("⚠️ CUDA not available — model will run on CPU.")

    def infer(self, routines):
        """
        Perform inference on a list of routines (each a dict with input tensors).
        Returns: input_tensor, output_tensor, gt_tensor, edge_probs
        """
        steps = self.step_size
        assert len(routines) == steps, "Number of routines must match the number of steps."

        prev_edges = None
        for i, routine in enumerate(routines):
            if i > 0:
                routine['edges'] = prev_edges

            if self.use_cuda:
                for k in routine:
                    routine[k] = routine[k].cuda()

            _, details, context_pred = self.model.step(routine)

            if i == 0:
                input_tensor = details['input']['location']
            prev_edges = details['output_probs']['location'].to(torch.float32)

        output_tensor = details['output']['location']
        gt_tensor = details['gt']['location']
        edge_probs = details['output_probs']['location'].to(torch.float32)

        # Assertions
        for tensor in [input_tensor, output_tensor, gt_tensor]:
            assert tensor.shape == (1, self.num_nodes)

        assert edge_probs.shape == (1, self.num_nodes, self.num_nodes)

        # Squeeze first dimension
        return (
            input_tensor.squeeze(0),
            output_tensor.squeeze(0),
            gt_tensor.squeeze(0),
            edge_probs
        )

    def get_predicted_movements(self, inp, pred):
        """
        Extracts predicted movements from the model's output.
        
        Args:
            inp (Tensor): Input tensor before perturbation with shape [num_nodes]
            pred (Tensor): Predicted tensor before perturbation with shape [num_nodes]
        
        Returns:
            List: List of predicted movements in the format [object, previous parent, new parent]
        """
        assert inp.shape == pred.shape, "Input and predicted tensors must have the same shape."
        assert inp.dim() == 1, "Input and predicted tensors must be 1D."
        
        movements = []
        for obj in range(inp.shape[0]):
            prev_parent = inp[obj].item()
            new_parent = pred[obj].item()
            if prev_parent != new_parent:
                movements.append([obj, prev_parent, new_parent])
        
        return movements