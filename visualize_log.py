import streamlit as st
import torch
import networkx as nx
from pyvis.network import Network
import os
import tempfile
from helpers.encoders import time_external
from GPT_explainer import GPTExplainer
from scene_context_builder import SceneContextExtractor

node_name = torch.load("node_classes.pt")
node_name[80] = "lemonade"
node_name[81] = "juice_glass"

gpt_explainer = GPTExplainer()

def assert_not_tensor(x, name="variable"):
    assert not isinstance(x, torch.Tensor), f"{name} should not be a torch.Tensor"

def convert_to_markdown(data):
    i=0
    while(True):
        if(data[i]=="\n" and not data[i+1]=="\n"):
            if(not data[i-1] == "\n"):
                data = data[:i] +"  "+data[i:]
                print(data)
                i+=2
        i+=1
        if i>=len(data)-1:
            break
    return data

def load_log(log_file):
    """Load the PyTorch log file and return the stored data."""
    data = torch.load(log_file, map_location=torch.device('cpu'))
    return data
def tensor_to_string(tensor):
    st  = "["
    # print("tensor.shape: ", tensor.shape)
    for i in range(tensor.shape[0]):
        val = tensor[i].item()
        val = round(val, 2)
        st += str(val) + ", "
    st = st[:-2]
    st += "]"
    return st

def get_parents_and_children(curr_graph,active_nodes):
    active_nodes_tensor = torch.tensor(active_nodes, dtype=torch.long, device=curr_graph.device)

    # Get parents directly
    parents = curr_graph[active_nodes_tensor]

    # Vectorized: For each node, find if it's a parent of any other node
    # Create a (len(active_nodes), len(curr_graph)) comparison matrix
    comparisons = active_nodes_tensor[:, None] == curr_graph[None, :]
    # Each row i now contains a boolean mask of where curr_graph == active_nodes[i]
    children_lists = [torch.nonzero(row, as_tuple=False).flatten().tolist() for row in comparisons]

    # Interleave parents and children into result
    result = []
    for p, n in zip(parents.tolist(), active_nodes):
        result.append((n, p))
    
    for children,n in zip(children_lists, active_nodes):
        for c in children:
            result.append((c,n))

    result = list(set(result))

    return result

def edges_to_text(edges):
    context = "" 
    for edge in edges: 
        context += f"{node_name[edge[0]]} is in/on {node_name[edge[1]]}. \n"

    return context

# def get_active_nodes(predicted_movements, influential_movements,key):
#     active_nodes = []
    
#     active_nodes.append(key)
#     active_nodes.append(predicted_movements[key][0])
#     active_nodes.append(predicted_movements[key][1])
#     for influential_movement in influential_movements[key]:
#         active_nodes.append(influential_movement[0])
#         active_nodes.append(influential_movement[1])
#     cleaned_active_nodes = []
#     for node in active_nodes:
#         if type(node) == torch.Tensor:
#             node = node.item()
#         cleaned_active_nodes.append(node)
#     active_nodes = cleaned_active_nodes
#     active_nodes = list(set(active_nodes))
#     return active_nodes

def add_children_to_active_nodes(curr_graph, active_nodes):
    for active_node in active_nodes:
        children = torch.nonzero(curr_graph == active_node, as_tuple=False).flatten().tolist()
        active_nodes.extend(children)
    active_nodes = list(set(active_nodes))


def summarized_curr_graph_to_text(curr_graph, active_nodes):
    # active_nodes = get_active_nodes(predicted_movements, influential_movements,key)
    curr_graph_list = curr_graph.tolist()
    add_children_to_active_nodes(curr_graph, active_nodes)
    ## DEBUG#####################
    for i in active_nodes:
        print("active_nodes: ", node_name[i])

    #######################
    print("creating scene context extractor")
    sce = SceneContextExtractor(curr_graph_list, node_name, active_nodes)
    print("Extracted scene context")
    curr_graph_txt = sce.get_ordered_leaf_active_paths()
    print("curr_graph_txt: ", curr_graph_txt)
    # raise NotImplementedError
    # parents_and_children = get_parents_and_children(curr_graph, active_nodes)
    # curr_graph_txt = edges_to_text(parents_and_children)
    return curr_graph_txt

def curr_graph_to_text(curr_graph):
    print("curr_graph: ", type(curr_graph))
    print("curr_graph.shape: ", curr_graph.shape)   
    print("curr_graph: ", curr_graph)
    curr_graph_text = "The current graph of the household is: "
    for i in range(curr_graph.shape[0]):
        print("node: ", node_name[i],"->", node_name[curr_graph[i]])
        curr_graph_text += node_name[i] +' is connected to '+ node_name[curr_graph[i]] + '.\n'

    
    
    return(curr_graph_text)


def generate_text(curr_graph, predicted_movements, influential_movements,time_influence, true_time):
    # predicted movements: {obj1: [curr_pose, pred_pose], obj2: [curr_pose, pred_pose],  .... }
    # influential_movements: {obj1: [[influential_obj1, old_pose, new_pose],[influential_obj2, old_pose, new_pose], .... ], obj2: [...]}
    # curr_graph_text = curr_graph_to_text(curr_graph)
    


    keys = predicted_movements.keys()
    raw_exp_text = ""
    gpt_explanations = ""
    curr_time_semantic = time_external(true_time).tolist()
    curr_time_semantic_txt = ""
    hours = curr_time_semantic[2]
    minutes = curr_time_semantic[3]
    hours = int(hours)
    minutes = int(minutes)
    if hours > 12:
        hours = hours - 12
        curr_time_semantic_txt = str(hours) + ":" + str(minutes) + " PM"
    else:
        curr_time_semantic_txt = str(hours) + ":" + str(minutes) + " AM"
    
        
    
    # for key in keys:
    #     print("key: ", key, "len(influential_movements[key]): ", len(influential_movements[key]))
    # raise NotImplementedError

    for key in keys:
        active_nodes_for_context = [key]
        assert_not_tensor(key, "key")
        active_nodes_for_context.append(predicted_movements[key][0])
        assert_not_tensor(predicted_movements[key][0], "predicted_movements[key][0]")
        active_nodes_for_context.append(predicted_movements[key][1])
        assert_not_tensor(predicted_movements[key][1], "predicted_movements[key][1]")
        predicted = predicted_movements[key]
        # summarized_cg_txt = "The current state: \n" + summarized_curr_graph_to_text(curr_graph, predicted_movements, influential_movements,key)
        # text = summarized_cg_txt + f"\n\nI predict that {node_name[key]} moves from {node_name[predicted[0].item()]} to  {node_name[predicted[1].item()]}. "
        
        text =  f"\n\nI moved {node_name[key]} from {node_name[predicted[0]]} to  {node_name[predicted[1]]}. The following reasons influenced my decision: \n"
        
        confidences = []
        influential_movements[key] = sorted(influential_movements[key], key=lambda x: x[3], reverse=True)

        
        for influential_movement in influential_movements[key]:
            # print("confidences::: ", influential_movement[3].item())
            confidences.append(influential_movement[3])
        confidences = sorted(confidences, reverse=True)
        no_candidates = min(3, len(confidences))
        if no_candidates != 0:
            threshold = 0.2 #confidences[no_candidates-1]
            # threshold = max(0.5, threshold)
            filtered_influential_movements = {}
            for influential_movement in influential_movements[key]:
                if influential_movement[3] < threshold:
                    continue
                if influential_movement[0] not in filtered_influential_movements:
                    filtered_influential_movements[influential_movement[0]] = [influential_movement[2], influential_movement[3]]
                else:
                    if influential_movement[3] > filtered_influential_movements[influential_movement[0]][1]:
                        filtered_influential_movements[influential_movement[0]] = [influential_movement[2], influential_movement[3]]
                # print(influential_movement)
                # raise NotImplementedError
                # text += f"{node_name[influential_movement[0]]} moved from {node_name[influential_movement[1].item()]} to {node_name[influential_movement[2].item()]} (conf: {influential_movement[3]}) ---and---\n"
                # print("Len of influential_movement: ", len(influential_movement))
                # pred_mov_probs = tensor_to_string(influential_movement[4])
                # out_mov_probs = tensor_to_string(influential_movement[5])
                # verbose: # text += f"{node_name[influential_movement[0]]} moved from {node_name[influential_movement[1].item()]} to {node_name[influential_movement[2].item()]} (conf: {influential_movement[3]}) (pred_prob:{pred_mov_probs}) , (out_probs: {out_mov_probs} ---and---\n"
                # text += f"{node_name[influential_movement[0]]} moving from {node_name[influential_movement[1].item()]} to {node_name[influential_movement[2].item()]} (conf: {influential_movement[3]}) ---and---\n"
                if type(influential_movement[3]) == torch.Tensor:
                    influential_movement[3] = influential_movement[3].item()
                # print(type(influential_movement[0]))
                # print(type(influential_movement[1]))
                # print(type(influential_movement[2]))
                # raise NotImplementedError
                active_nodes_for_context.append(influential_movement[0])
                active_nodes_for_context.append(influential_movement[1])
                active_nodes_for_context.append(influential_movement[2])
                # text += f"My prediction confidence reduces by {round(influential_movement[3], 2)} if {node_name[influential_movement[0]]} did not move from {node_name[influential_movement[1]]} to {node_name[influential_movement[2]]}.\n"
            for key__, value__ in filtered_influential_movements.items():
                # text += f"My prediction confidence reduces by {round(value[1], 2)} if {node_name[key]} did not move from {node_name[key]} to {node_name[value[0]}.\n"
                text += f"{node_name[key__]} is placed on/in {node_name[value__[0]]} (influence: {round(value__[1], 2)}).\n"

        # if text[-10:] == "---and---\n":
        #     text = text[:-10]
        # elif text[-13:] == "- because, --":
        #     text = text[:-14]
        ## Get current graph text
        curr_graph_text = summarized_curr_graph_to_text(curr_graph, active_nodes_for_context)
        text = "current state:\n"+curr_graph_text + "\n\n" + text
        # text +="."
        int_to_time = ""
        # Add time influence
        # moring: 6:00 to 12:00 -> 0 to 33
        # afternoon: 12:00 to 18:00 -> 33 to 69
        # evening: 18:00 to 25:30 -> 69 to 108

        morning_influence = torch.tensor(time_influence[key][1][0:33])
        afternoon_influence = torch.tensor(time_influence[key][1][33:69])
        evening_influence = torch.tensor(time_influence[key][1][69:108])
        

        # Filtering:
        # Morning:
        # print("morning_influence: ", morning_influence)
        morning_mean = torch.mean(morning_influence)
        morning_std = torch.std(morning_influence)
        morning_influence = morning_influence[morning_influence > morning_mean - 2*morning_std]
        morning_influence = morning_influence[morning_influence < morning_mean + 2*morning_std]
        morning_influence = torch.mean(morning_influence).item()
        morning_influence = round(1+morning_influence, 2)

        # Afternoon:
        afternoon_mean = torch.mean(afternoon_influence)
        afternoon_std = torch.std(afternoon_influence)
        afternoon_influence = afternoon_influence[afternoon_influence > afternoon_mean - 2*afternoon_std]
        afternoon_influence = afternoon_influence[afternoon_influence < afternoon_mean + 2*afternoon_std]
        afternoon_influence = torch.mean(afternoon_influence).item()
        afternoon_influence = round(1+afternoon_influence, 2)
        
        # Evening:
        evening_mean = torch.mean(evening_influence)
        evening_std = torch.std(evening_influence)
        evening_influence = evening_influence[evening_influence > evening_mean - 2*evening_std]
        evening_influence = evening_influence[evening_influence < evening_mean + 2*evening_std]
        evening_influence = torch.mean(evening_influence).item() 
        evening_influence = round(1+evening_influence, 2)
        
        time_text = f"\nThe time is morning (influence: {morning_influence}).\n"
        time_text += f"The time is afternoon (influence: {afternoon_influence}).\n"
        time_text += f"The time is evening (influence: {evening_influence}).\n"
        
        # time_text = "The mean confidence of the prediction if it is in the morning "
        # if morning_influence > 0.0:
        #     time_text += "increases by " + str(morning_influence) + ".\n"
        # else:
        #     time_text += "decreases by " + str(-1*morning_influence) + ".\n"
        # time_text += "The mean confidence of the prediction if it is in the afternoon "
        # if afternoon_influence > 0.0:
        #     time_text += "increases by " + str(afternoon_influence) + ".\n"
        # else:
        #     time_text += "decreases by " + str(-1*afternoon_influence) + ".\n"
        # time_text += "The mean confidence of the prediction if it is in the evening "
        # if evening_influence > 0.0:
        #     time_text += "increases by " + str(evening_influence) + ".\n"
        # else:
        #     time_text += "decreases by " + str(-1*evening_influence) + ".\n"

        # time_text = '['
        # for i in range(len(time_influence[key][1])):
        #     val_time = time_influence[key][0][i]
        #     time_semantic = time_external(val_time).tolist()
        #     # time_semantic_txt = 'week:'+str(int(time_semantic[0]))+'day:'+str(int(time_semantic[1]))+'hours:'+str(int(time_semantic[2]))+'mins:'+str(int(time_semantic[3]))
        #     time_semantic_txt = str(int(time_semantic[2]))+':'+str(int(time_semantic[3]))
        #     int_to_time += "("+str(i)+","+time_semantic_txt+","+str(val_time)+"), \n"
        #     val = time_influence[key][1][i]
        #     # val to 2 decimal places
        #     val = round(val, 2
        #     time_text= time_text + '('+time_semantic_txt+','+str(val) + "),"
        # time_text = time_text[:-2]
        # time_text += "]"

        text += f"\nTime influence: The current time is {curr_time_semantic_txt}. {time_text}"
        
        gpt_explained_txt = gpt_explainer.request(text)
        gpt_explanations += gpt_explained_txt +'\n\n'
        
        raw_exp_text += text + '\n\n GPT Explanation: ' + gpt_explained_txt + "\n\n"
        # print("time_influence:",time_influeence[key])
        # print("int_to_time:",int_to_time)
        # print(predicted)
        # raise NotImplementedError  # Remove this line and implement the function

    
    # full_text = curr_graph_text + "\n\n" + raw_exp_text #+ "\n\n" + gpt_explanations[:-1]
    full_text = raw_exp_text #+ "\n\n" + gpt_explanations[:-1]
    return full_text

def plot_collapsible_tree(graph_data):
    """Create an interactive collapsible tree visualization."""
    G = nx.DiGraph()
    num_nodes = graph_data.shape[0]
    
    # Add nodes and edges
    for i in range(num_nodes):
        G.add_node(node_name[i]+"_"+str(i))
        if graph_data[i] >= 0:  # Assuming each node has a parent
            G.add_edge(node_name[graph_data[i]]+"_"+str(graph_data[i]), node_name[i]+"_"+str(i))
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
    # get run number
    with open("logs/log_no.txt", "r") as f:
        run_no = f.read()
    run_no = int(run_no) - 1
    base = "logs/run_" + str(run_no) + "/log_"
    # base = "logs/run_46/log_"
    no_log_files = len(os.listdir("logs/run_" + str(run_no)))
    log_files = []
    for i in range(1, no_log_files+1):
        log_files.append(base + str(i) + ".pt")

    # only_log_files = ['log_1.pt', 'log_2.pt', 'log_6.pt', 'log_9.pt', 'log_10.pt', 'log_13.pt', 'log_14.pt', 'log_15.pt', 'log_19.pt', 'log_22.pt', 'log_24.pt', 'log_27.pt', 'log_29.pt', 'log_32.pt', 'log_34.pt', 'log_37.pt', 'log_38.pt', 'log_41.pt', 'log_42.pt', 'log_46.pt', 'log_50.pt']
    # base = "logs/run_7/"
    # log_files = [base + log_file for log_file in only_log_files]

    selected_log = st.selectbox("Select a log file", log_files)
    text_to_display = "Please select a log file first."
    if selected_log:
        log_data = load_log(selected_log)
        # print("log_data: ", log_data[4])
        # raise NotImplementedError
        
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
        text_to_display = generate_text(log_data[0],log_data[1],log_data[2],log_data[3],log_data[4])
    
    # Placeholder text area
    # st.text_area("Analysis Notes", "Enter your observations here...")
    st.markdown("Explanation")
    text_to_display_md = convert_to_markdown(text_to_display)
    st.markdown(text_to_display_md)

if __name__ == "__main__":
    main()
