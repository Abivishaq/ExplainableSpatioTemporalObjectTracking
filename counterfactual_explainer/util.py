import os
import json
def get_node_classes():
    homer_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data','HOMER')
    homer_folder = os.path.join(homer_root, f'household1')  # Example for household 1
    node_calsses_file = os.path.join(homer_folder,'processed' ,'common_data.json')
    if not os.path.exists(node_calsses_file):
        raise FileNotFoundError(f"Node classes file not found in {homer_folder}.")
    with open(node_calsses_file, 'r') as f:
        node_classes = json.load(f)
    return node_classes['node_classes']