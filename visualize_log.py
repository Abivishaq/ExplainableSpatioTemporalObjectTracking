import streamlit as st
import torch
import networkx as nx
from pyvis.network import Network
import os
import tempfile
from helpers.encoders import time_external
from GPT_explainer import GPTExplainer
from scene_context_builder import SceneContextExtractor
import matplotlib.pyplot as plt
import io
import base64

node_name = torch.load("node_classes.pt")
node_name[80] = "lemonade"
node_name[81] = "juice_glass"

gpt_explainer = GPTExplainer()

def create_dummy_plot(time_data) -> str:
    """
    Generates a simple dummy Matplotlib plot and returns it
    as a base64 encoded image embedded in a Markdown string.

    This format is compatible with Streamlit's st.markdown()
    when unsafe_allow_html=True.

    Returns:
        str: A Markdown string representing the plot image.
             Example: "![Dummy Plot](data:image/png;base64,...)"
    """
    # 1. Create a dummy plot
    fig, ax = plt.subplots()
    x = [1, 2, 3, 4, 5]
    y = [2, 3, 5, 7, 11] # Some prime numbers for fun
    ax.plot(x, y, marker='o')
    ax.set_title("Simple Dummy Plot")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.grid(True)

    # 2. Save plot to a BytesIO buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0) # Rewind buffer to the beginning

    # 3. Encode buffer contents as base64
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # 4. Close the plot figure to free memory
    plt.close(fig)

    # 5. Create Markdown string
    markdown_string = f"![Dummy Plot](data:image/png;base64,{image_base64})"

    return markdown_string

def plot_time_influence_to_markdown(
    score_list: list,
    y_label: str = "Score",
    title: str = "Scores from 6:20 AM to Next Day ~2:00 AM"
) -> str:
    """
    Generates a plot of 118 scores (6:20 AM start, 10-min intervals)
    with 2-hourly labels and colored backgrounds, using specified font sizes,
    then returns it as a base64 encoded image embedded in a Markdown string.

    Args:
        score_list: A list containing exactly 118 numerical score values.
        y_label: Label for the Y-axis. Defaults to "Score".
        title: Title for the plot. Used as Alt text in Markdown.

    Returns:
        str: A Markdown string representing the plot image.
             Example: "![Plot Title](data:image/png;base64,...)"
             Ready for use with st.markdown(..., unsafe_allow_html=True).

    Raises:
        ValueError: If the input list does not contain exactly 118 elements
                    or if elements are not numeric.
    """
    # Define background colors for the regions (can be customized)
    COLOR_REGION_1 = 'lightblue'       # Indices 0-33
    COLOR_REGION_2 = 'lightgoldenrodyellow' # Indices 34-69
    COLOR_REGION_3 = 'lightgrey'       # Indices 70-117
    BACKGROUND_ALPHA = 0.4             # Transparency for background colors

    # --- Font Size Definitions ---
    # Adjust these values as needed
    default_fontsize = 24
    TITLE_FONTSIZE = default_fontsize
    AXIS_LABEL_FONTSIZE = default_fontsize
    TICK_LABEL_FONTSIZE = default_fontsize  # For X and Y axis tick numbers/labels
    LEGEND_TITLE_FONTSIZE = default_fontsize
    LEGEND_ITEM_FONTSIZE = default_fontsize


    NUM_EXPECTED_ELEMENTS = 118

    # --- Input Validation ---
    if not isinstance(score_list, list):
        raise ValueError("Input must be a list.")
    if len(score_list) != NUM_EXPECTED_ELEMENTS:
        raise ValueError(f"Input list must contain exactly {NUM_EXPECTED_ELEMENTS} elements, but found {len(score_list)}.")
    if not all(isinstance(item, (int, float)) for item in score_list):
         raise ValueError("All elements in the list must be numeric (int or float).")

    # --- Setup Plot ---
    fig, ax = plt.subplots(figsize=(18, 6))

    try:
        # --- Add Background Coloring ---
        ax.axvspan(-0.5, 34, facecolor=COLOR_REGION_1, alpha=BACKGROUND_ALPHA, zorder=0, label='Morning')
        ax.axvspan(34, 70, facecolor=COLOR_REGION_2, alpha=BACKGROUND_ALPHA, zorder=0, label='Afternoon')
        ax.axvspan(70, NUM_EXPECTED_ELEMENTS - 0.5, facecolor=COLOR_REGION_3, alpha=BACKGROUND_ALPHA, zorder=0, label='Evening')

        # --- Plot Data ---
        for i in range(NUM_EXPECTED_ELEMENTS):
            score_list[i]+=1
        indices = range(NUM_EXPECTED_ELEMENTS)
        ax.plot(indices, score_list, marker='.', linestyle='-', zorder=2, color='black', label=y_label)

        # --- Configure X-axis Ticks and Labels ---
        intervals_per_2_hours = 12
        first_tick_index = 10
        first_tick_hour = 8
        tick_indices = []
        hour_labels = []
        current_tick_index = first_tick_index
        current_hour = first_tick_hour
        while current_tick_index < NUM_EXPECTED_ELEMENTS:
            tick_indices.append(current_tick_index)
            hour_labels.append(f"{current_hour:02d}:00")
            current_tick_index += intervals_per_2_hours
            current_hour = (current_hour + 2) % 24

        ax.set_xticks(tick_indices)
        ax.set_xticklabels(hour_labels, rotation=45, ha='right') # Apply labels *after* setting ticks

        # --- Add Labels, Title, Grid, Legend (with Font Sizes) ---
        ax.set_xlabel("Time", fontsize=AXIS_LABEL_FONTSIZE) # Set fontsize
        ax.set_ylabel(y_label, fontsize=AXIS_LABEL_FONTSIZE) # Set fontsize
        ax.set_title(title, fontsize=TITLE_FONTSIZE) # Set fontsize

        # --- Adjust Tick Label Sizes ---
        ax.tick_params(axis='x', labelsize=TICK_LABEL_FONTSIZE) # Set x-tick label size
        ax.tick_params(axis='y', labelsize=TICK_LABEL_FONTSIZE) # Set y-tick label size

        ax.grid(axis='y', linestyle='--', linewidth=0.5, color='gray', zorder=1)

        # --- Set Legend Font Sizes ---
        ax.legend(
            title="Index Ranges",
            loc='upper left',
            bbox_to_anchor=(1.01, 1),
            fontsize=LEGEND_ITEM_FONTSIZE,        # Font size for legend items
            title_fontsize=LEGEND_TITLE_FONTSIZE # Font size for legend title
        )

        # --- Final Adjustments ---
        fig.tight_layout(rect=[0, 0, 0.9, 1])

        # --- Convert Plot to Markdown ---
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        alt_text = title.replace('"', '&quot;')
        markdown_string = f"![{alt_text}](data:image/png;base64,{image_base64})"

        return markdown_string

    finally:
        plt.close(fig)

def assert_not_tensor(x, name="variable"):
    assert not isinstance(x, torch.Tensor), f"{name} should not be a torch.Tensor"

def convert_to_markdown(data):
    i=0
    while(True):
        if(data[i]=="\n" and not data[i+1]=="\n"):
            if(not data[i-1] == "\n"):
                data = data[:i] +"  "+data[i:]
                # print(data)
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

    #######################
    sce = SceneContextExtractor(curr_graph_list, node_name, active_nodes)
    curr_graph_txt = sce.get_ordered_leaf_active_paths()
    return curr_graph_txt

def curr_graph_to_text(curr_graph):
    
    curr_graph_text = "The current graph of the household is: "
    for i in range(curr_graph.shape[0]):
        curr_graph_text += node_name[i] +' is connected to '+ node_name[curr_graph[i]] + '.\n'

    
    
    return(curr_graph_text)

def get_time_influence_wo_outliers(time_influence_list):
    if type(time_influence_list) != torch.Tensor:
        time_influence_list = torch.tensor(time_influence_list)
    # Note:time influence = perturbation_prob - original_prob
    
    # mean = torch.mean(time_influence_list)
    # std = torch.std(time_influence_list)
    # time_influence_list = time_influence_list[time_influence_list > mean - 2*std]
    # time_influence_list = time_influence_list[time_influence_list < mean + 2*std]

    time_influence = torch.mean(time_influence_list).item()
    time_influence = round(1+time_influence, 2)
    return time_influence

def get_time_text(time_influence,current_time_int,key):
    current_time_index = int((current_time_int-380)/10)
    # 0 index is 380
    #
    # raise NotImplementedError
    return(get_time_text_v2(time_influence,current_time_int,key,current_time_index))


def get_time_text_v1(time_influence,current_time_int,key,current_time_index):

    curr_time_semantic = time_external(current_time_int).tolist()
    curr_time_semantic_txt = ""
    hours = curr_time_semantic[2]
    minutes = curr_time_semantic[3]
    hours = int(hours)
    minutes = int(minutes)
    if hours > 12:
        hours = hours - 12
        curr_time_semantic_txt = f"{hours:02d}:{minutes:02d} PM"
    else:
        curr_time_semantic_txt = f"{hours:02d}:{minutes:02d} AM"

     # Add time influence
    # moring: 6:00 to 11:50 -> 0 to 33
    # afternoon: 12:00 to 17:50 -> 34 to 69
    # evening: 18:00 to 26:00 -> 70 to 118

    morning_influence_list = torch.tensor(time_influence[key][1][0:34])
    afternoon_influence_list = torch.tensor(time_influence[key][1][34:70])
    evening_influence_list = torch.tensor(time_influence[key][1][70:118])
    

    # Filtering:
    # Morning:
    # print("morning_influence: ", morning_influence)
    morning_influence = get_time_influence_wo_outliers(morning_influence_list)

    # Afternoon:
    afternoon_influence = get_time_influence_wo_outliers(afternoon_influence_list)
    
    # Evening:
    evening_influence = get_time_influence_wo_outliers(evening_influence_list)
    
    
    time_text = f"\nThe current time is {curr_time_semantic_txt}"
    time_text += f"\nThe following facts about time influenced my decision and each fact's importance is mentioned in (): \n"
    time_text += f"It is morning (influence: {morning_influence}).\n"
    time_text += f"It is afternoon (influence: {afternoon_influence}).\n"
    time_text += f"It is evening (influence: {evening_influence}).\n"
    return time_text

def get_time_text_v2(time_influence,current_time_int,key,current_time_index):
    curr_time_semantic = time_external(current_time_int).tolist()
    curr_time_semantic_txt = ""
    hours = curr_time_semantic[2]
    minutes = curr_time_semantic[3]
    hours = int(hours)
    minutes = int(minutes)
    if hours > 12:
        hours = hours - 12
        curr_time_semantic_txt = f"{hours:02d}:{minutes:02d} PM"
    else:
        curr_time_semantic_txt = f"{hours:02d}:{minutes:02d} AM"

    # Add time influence
    # moring: 6:00 to 12:00 -> 0 to 33
    # afternoon: 12:00 to 18:00 -> 33 to 69
    # evening: 18:00 to 25:30 -> 69 to 108

    morning_influence_list = torch.tensor(time_influence[key][1][0:33])
    afternoon_influence_list = torch.tensor(time_influence[key][1][33:69])
    evening_influence_list = torch.tensor(time_influence[key][1][69:118])
    print("time_influence_len",len(time_influence[key][1]))
    # raise NotImplementedError
    assert len(time_influence[key][1]) == 118, f"len(time_influence[key][1]): {len(time_influence[key][1])} != 118"


    # Filtering:
    # Morning:
    # print("morning_influence: ", morning_influence)
    morning_influence = get_time_influence_wo_outliers(morning_influence_list)

    # Afternoon:
    afternoon_influence = get_time_influence_wo_outliers(afternoon_influence_list)

    # Evening:
    evening_influence = get_time_influence_wo_outliers(evening_influence_list)

    # checking if the time is morning, afternoon or evening
    if current_time_index >= 0 and current_time_index < 33:
        is_morning = True
        is_afternoon = False
        is_evening = False
    elif current_time_index >= 33 and current_time_index < 69:
        is_morning = False
        is_afternoon = True
        is_evening = False
    else:
        is_morning = False
        is_afternoon = False
        is_evening = True

    time_text = f"\nThe current time is {curr_time_semantic_txt}"
    time_text += f"\nThe following facts about time influenced my decision and each fact's importance is mentioned in (): \n"
    if is_morning:
        time_text += f"It is morning (influence: {morning_influence}).\n"
    else:
        time_text += f"It is NOT morning (influence: {round(1- morning_influence,2)}).\n"
    if is_afternoon:
        time_text += f"It is afternoon (influence: {afternoon_influence}).\n"
    else:
        time_text += f"It is NOT afternoon (influence: {round(1- afternoon_influence,2)}).\n"
    if is_evening:
        time_text += f"It is evening (influence: {evening_influence}).\n"
    else:
        time_text += f"It is NOT evening (influence: {round(1- evening_influence,2)}).\n"
    return time_text

def get_time_text_v3(time_influence,current_time_int,key,current_time_index):
    curr_time_semantic = time_external(current_time_int).tolist()
    curr_time_semantic_txt = ""
    hours = curr_time_semantic[2]
    minutes = curr_time_semantic[3]
    hours = int(hours)
    minutes = int(minutes)
    if hours > 12:
        hours = hours - 12
        curr_time_semantic_txt = f"{hours:02d}:{minutes:02d} PM"
    else:
        curr_time_semantic_txt = f"{hours:02d}:{minutes:02d} AM"

    # Add time influence
    # moring: 6:00 to 12:00 -> 0 to 33
    # afternoon: 12:00 to 18:00 -> 33 to 69
    # evening: 18:00 to 25:30 -> 69 to 118

    morning_influence_list = torch.tensor(time_influence[key][1][0:33])
    afternoon_influence_list = torch.tensor(time_influence[key][1][33:69])
    evening_influence_list = torch.tensor(time_influence[key][1][69:118])


    # Filtering:
    # Morning:
    morning_influence = get_time_influence_wo_outliers(morning_influence_list)
    afternoon_influence = get_time_influence_wo_outliers(afternoon_influence_list)
    evening_influence = get_time_influence_wo_outliers(evening_influence_list)

    # checking if the time is morning, afternoon or evening
    if current_time_index >= 0 and current_time_index < 33:
        is_morning = True
        is_afternoon = False
        is_evening = False
    elif current_time_index >= 33 and current_time_index < 69:
        is_morning = False
        is_afternoon = True
        is_evening = False
    else:
        is_morning = False
        is_afternoon = False
        is_evening = True

    time_text = f"\nThe current time is {curr_time_semantic_txt}"
    time_text += f"\nThe following facts about time influenced my decision and each fact's importance is mentioned in (): \n"
    if is_morning:
        # time_text += f"It is morning (influence: {morning_influence}).\n"
        before_time = time_influence[key][1][0:current_time_index]
        before_time_influence = get_time_influence_wo_outliers(before_time)

        after_time = time_influence[key][1][current_time_index+1:]
        after_time_influence = get_time_influence_wo_outliers(after_time)
        
        time_text += f"It is NOT earlier than {curr_time_semantic_txt} (influence: {round(1-before_time_influence,2)}).\n"
        time_text += f"It is NOT later than {curr_time_semantic_txt} (influence: {round(1- after_time_influence,2)}).\n" 
    else:
        time_text += f"It is NOT morning (influence: {round(1- morning_influence,2)}).\n"
    if is_afternoon:
        before_time = time_influence[key][1][33:current_time_index]
        before_time_influence = get_time_influence_wo_outliers(before_time)

        after_time = time_influence[key][1][current_time_index+1:69]
        after_time_influence = get_time_influence_wo_outliers(after_time)

        time_text += f"It is NOT earlier than {curr_time_semantic_txt} (influence: {round(1-before_time_influence,2)}).\n"
        time_text += f"It is NOT later than {curr_time_semantic_txt} (influence: {round(1- after_time_influence,2)}).\n"
        # before_time_influence
        print("before_time: ", before_time)
        print("before_time_influence: ", before_time_influence)
        print("after_time:", after_time)
        print("after_time_influence: ", after_time_influence)

    else:
        time_text += f"It is NOT afternoon (influence: {round(1- afternoon_influence,2)}).\n"
    
    if is_evening:
        before_time = time_influence[key][1][69:current_time_index]
        before_time_influence = get_time_influence_wo_outliers(before_time)

        after_time = time_influence[key][1][current_time_index+1:]
        after_time_influence = get_time_influence_wo_outliers(after_time)

        time_text += f"It is NOT earlier than {curr_time_semantic_txt} (influence: {round(1-before_time_influence,2)}).\n"
        time_text += f"It is NOT later than {curr_time_semantic_txt} (influence: {round(1- after_time_influence,2)}).\n"
    else:
        time_text += f"It is NOT evening (influence: {round(1- evening_influence,2)}).\n"
    return time_text

def generate_text(curr_graph, predicted_movements, influential_movements,time_influence, true_time):
    # predicted movements: {obj1: [curr_pose, pred_pose], obj2: [curr_pose, pred_pose],  .... }
    # influential_movements: {obj1: [[0- influential_obj1, 1- old_pose, 2- new_pose, 3- influence_level, 4- original_prob, 5-perturbed prob],[influential_obj2, old_pose, new_pose,..], .... ], obj2: [...]}
    # curr_graph_text = curr_graph_to_text(curr_graph)

    keys = predicted_movements.keys()
    raw_exp_text = ""
    gpt_explanations = ""
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
        action_text =f"**ACTION: I moved {node_name[key]} from {node_name[predicted[0]]} to  {node_name[predicted[1]]}**.\n\n" 
        text =  f"The following facts about object location influenced my decision and each fact's importance is mentioned in (): \n"
        ################################# Sorting based on if pertrubation changes predicted parent ##########################
        confidences = []
        influential_movements[key] = sorted(influential_movements[key], key=lambda x: x[3], reverse=True)

        
        for influential_movement in influential_movements[key]:
            confidences.append(influential_movement[3])
        confidences = sorted(confidences, reverse=True)
        # no_candidates = min(3, len(confidences))
        if True: #no_candidates != 0:
            # threshold = 0.5
            filtered_influential_movements = {}
            for influential_movement in influential_movements[key]:
                pred_prob = influential_movement[4]
                perturbed_prob = influential_movement[5]
                pred_true_ind = torch.argmax(pred_prob)
                perturbed_ind = torch.argmax(perturbed_prob)
                recalc_influence = pred_prob[pred_true_ind] - perturbed_prob[pred_true_ind]
                assert abs(recalc_influence-influential_movement[3]) < 0.0001, f"recalc_influence: {recalc_influence}, influential_movement[3]: {influential_movement[3]}"
                perturbed_confidence = influential_movement[5][pred_true_ind]
                # print("influential_movement: ", influential_movement[5])
                if perturbed_ind == pred_true_ind:
                    continue
                if influential_movement[0] not in filtered_influential_movements:
                    filtered_influential_movements[influential_movement[0]] = [influential_movement[2], influential_movement[3]]
                else:
                    if influential_movement[3] > filtered_influential_movements[influential_movement[0]][1]:
                        filtered_influential_movements[influential_movement[0]] = [influential_movement[2], influential_movement[3]]

                if type(influential_movement[3]) == torch.Tensor:
                    influential_movement[3] = influential_movement[3].item()
                
                active_nodes_for_context.append(influential_movement[0])
                active_nodes_for_context.append(influential_movement[1])
                active_nodes_for_context.append(influential_movement[2])

            for key__, value__ in filtered_influential_movements.items():
                text += f"{node_name[key__]} is placed on/in {node_name[value__[0]]} (influence: {round(value__[1], 2)}).\n"
        ################################## End of Sorting based on if pertrubation changes predicted parent ##########################

        ## Get current graph text
        curr_graph_text = summarized_curr_graph_to_text(curr_graph, active_nodes_for_context)
        text = action_text + "The current state of relevant objects are as follows:\n"+curr_graph_text + "\n\n" + text
        # text +="."
        int_to_time = ""
        time_text = get_time_text(time_influence, true_time,key)

        text += time_text #f"\nTime influence: The current time is {curr_time_semantic_txt}. {time_text}"

        time_graph = plot_time_influence_to_markdown(time_influence[key][1], y_label="Influence", title="Time Influence")
        text += time_graph
        
        gpt_explained_txt = gpt_explainer.request(text)
        gpt_explanations += gpt_explained_txt +'\n\n'
        
        raw_exp_text += text + '\n\n GPT Explanation: ' + gpt_explained_txt + "\n\n"
        raw_exp_text += "-------------------------------------------------------------------------------------------------------------------\n\n"
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


    # plot_md = create_dummy_plot_markdown()
    # add dummy plot to markdown here.
    # st.markdown(plot_md, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
