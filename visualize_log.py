import streamlit as st
import torch
import networkx as nx
from pyvis.network import Network
import os
import tempfile

node_name = torch.load("node_classes.pt")

def load_log(log_file):
    """Load the PyTorch log file and return the stored data."""
    data = torch.load(log_file, map_location=torch.device('cpu'))
    return data

def generate_text(predicted_movements, influential_movements):
    # predicted movements: {obj1: [curr_pose, pred_pose], obj2: [curr_pose, pred_pose],  .... }
    # influential_movements: {obj1: [[influential_obj1, old_pose, new_pose],[influential_obj2, old_pose, new_pose], .... ], obj2: [...]}
    keys = predicted_movements.keys()
    full_text = ""
    # 
    print("keys: ")
    print("predicted_movements: ")
    print(predicted_movements.keys())
    print("influential_movements: ")
    print(influential_movements.keys())
    print("#############################")
    print("influentaial movements: ")
    print(influential_movements)
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    for key in keys:
        print("key: ", key, "len(influential_movements[key]): ", len(influential_movements[key]))
    # raise NotImplementedError
    for key in keys:
        predicted = predicted_movements[key]
        text = f"I predict that {node_name[key]} moves from {node_name[predicted[0].item()]} to  {node_name[predicted[1].item()]} -- because, --"
        print("error key:")
        print(key)
        for influential_movement in influential_movements[key]:
            # print(influential_movement)
            # raise NotImplementedError
            text += f"{node_name[influential_movement[0]]} moved from {node_name[influential_movement[1].item()]} to {node_name[influential_movement[2].item()]} ---and---\n"
        
        if text[-10:] == "---and---\n":
            text = text[:-10]
        elif text[-13:] == "- because, --":
            text = text[:-14]
        print("text[-13:]")
        print(text[-13:])
        text +="."
        
        full_text += text
        full_text += "\n\n"
        # print(predicted)
        # raise NotImplementedError  # Remove this line and implement the function
    return full_text

def plot_collapsible_tree(graph_data):
    """Create an interactive collapsible tree visualization."""
    G = nx.DiGraph()
    num_nodes = graph_data.shape[0]
    
    # Add nodes and edges
    for i in range(num_nodes):
        G.add_node(node_name[i])
        if graph_data[i] >= 0:  # Assuming each node has a parent
            G.add_edge(node_name[graph_data[i]], node_name[i] )
        else:
            G.add_edge( "ROOT",node_name[i])
    # G.add_node("ROOT")
    # G.add_node("A")
    # G.add_node("B")
    # G.add_node("C")

    # G.add_edge("ROOT", "A")
    # G.add_edge("ROOT", "B")
    # G.add_edge("ROOT", "C")
    
    net = Network(height="600px", width="100%", directed=True)
    net.from_nx(G)
    net.show_buttons(filter_=['physics'])  # Allow users to control physics
    
    # Save and display the HTML
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    net.save_graph(temp_file.name)
    st.components.v1.html(open(temp_file.name, "r").read(), height=700)

def main():
    st.title("Log File Visualization")
    
    # Dropdown to select log file
    # log_files = ["logs/run_4/log_1.pt", "logs/run_4/log_2.pt", "logs/run_4/log_3.pt"]
    # log_files = ["logs/run_7/log_3.pt"]
    base = "logs/run_7/log_"
    log_files = []
    for i in range(1, 50):
        log_files.append(base + str(i) + ".pt")

    only_log_files = ['log_1.pt', 'log_2.pt', 'log_6.pt', 'log_9.pt', 'log_10.pt', 'log_13.pt', 'log_14.pt', 'log_15.pt', 'log_19.pt', 'log_22.pt', 'log_24.pt', 'log_27.pt', 'log_29.pt', 'log_32.pt', 'log_34.pt', 'log_37.pt', 'log_38.pt', 'log_41.pt', 'log_42.pt', 'log_46.pt', 'log_50.pt']
    base = "logs/run_7/"
    log_files = [base + log_file for log_file in only_log_files]

    selected_log = st.selectbox("Select a log file", log_files)
    text_to_display = "Please select a log file first."
    if selected_log:
        log_data = load_log(selected_log)
        
        # Extract adjacency matrix from log data (assuming it's the first element)
        if isinstance(log_data, list) and len(log_data) > 0:
            adjacency_matrix = log_data[0]  # Extract the first entry
            
            if isinstance(adjacency_matrix, torch.Tensor):
                adjacency_matrix = adjacency_matrix.numpy()
                plot_collapsible_tree(adjacency_matrix)
            else:
                st.error("Unexpected data format in log file.")
        else:
            st.error("Log file is empty or improperly formatted.")
        text_to_display = generate_text(log_data[1],log_data[2])
    
    # Placeholder text area
    # st.text_area("Analysis Notes", "Enter your observations here...")
    st.markdown("Explanation")
    
    st.markdown(text_to_display)

if __name__ == "__main__":
    main()
