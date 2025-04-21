import torch

file= 'data/HOMER/household0/processed/train/000.pt'
data = torch.load(file)
print(len(data))
print(data[0].keys())
print("change_type",data[0]['change_type'])
print("activity",data[0]['activity'])
# nodes = data[0]['prev_nodes']
# for i in range(len(nodes)):
#     print(i, torch.argmax(nodes[i]).item())
prev_edge = data[0]['prev_edges']
prev_edge = torch.argmax(prev_edge, dim=1)

edges = data[0]['edges']
edges = torch.argmax(edges, dim=1)
edge_change = edges - prev_edge


print("edge_Change:",edge_change)

print("change_type index:",torch.nonzero(data[0]['change_type'], as_tuple=True)[0])
print("edge change index:",torch.nonzero(edge_change, as_tuple=True)[0])

print("len change_type",len(data[0]['change_type']))

print("time",data[0]['time'])

for data_i in data:
    print("activity",data_i['activity'])


# x = torch.tensor([-0.0033751726150512695, -0.01749134063720703, 0.002192676067352295, -0.017548859119415283, -0.04261195659637451, -0.035304129123687744, -0.03959643840789795, -0.2610904574394226, -0.08278727531433105, -0.09093290567398071, -0.37483644485473633, -0.17399728298187256, -0.1599435806274414])
# print("mean(x)",torch.mean(x))
# # def convert_to_markdown(data):
# #     i=0
# #     while(True):
# #         if(data[i]=="\n" and not data[i+1]=="\n"):
# #             if(not data[i-1] == "\n"):
# #                 data = data[:i] +"``"+data[i:]
# #                 print(data)
# #                 i+=2
# #         i+=1
# #         if i>=len(data)-1:
# #             break
# #     return data


# # st = "a\nb\n\nc"
# # md = convert_to_markdown(st)
# # print(md)
# # def get_one_hop_neigbours(curr_graph, active_nodes):
# #     # get parent and all children
# #     print(active_nodes)
# #     print(curr_graph)


# # def get_parents_and_children(curr_graph,active_nodes):
# #     active_nodes_tensor = torch.tensor(active_nodes, dtype=torch.long, device=curr_graph.device)

# #     # Get parents directly
# #     parents = curr_graph[active_nodes_tensor]

# #     # Vectorized: For each node, find if it's a parent of any other node
# #     # Create a (len(active_nodes), len(curr_graph)) comparison matrix
# #     comparisons = active_nodes_tensor[:, None] == curr_graph[None, :]
# #     # Each row i now contains a boolean mask of where curr_graph == active_nodes[i]
# #     children_lists = [torch.nonzero(row, as_tuple=False).flatten().tolist() for row in comparisons]

# #     # Interleave parents and children into result
# #     result = []
# #     for p, n in zip(parents.tolist(), active_nodes):
# #         result.append((p, n))
    
# #     for children,n in zip(children_lists, active_nodes):
# #         for c in children:
# #             result.append((n,c))



# #     return result
    
# def get_child_list(curr_graph):
#     child_list = []
#     for i in range(len(curr_graph)):
#         childs = (curr_graph==i).nonzero(as_tuple=False).flatten().tolist()
#         child_list.append(childs)
#     return  child_list
# if __name__ == "__main__":
#     curr_graph = torch.tensor([ 0,  0,  0,  0,  0,  4,  4,  4,  6,  4,  4, 10,  4,  0, 13,  0, 15, 15,
#             15, 15,  0, 10,  4, 13,  4, 23, 14, 19, 19,  9,  9,  2,  9,  9,  2, 22,
#             9,  9, 22,  2,  2, 15,  8,  5, 20,  9, 18,  9,  2, 18,  9, 19,  2, 24,
#             3,  9,  9,  2,  2,  3,  0,  4,  4, 13, 15,  4,  9, 64,  8, 64,  9, 68,
#             10, 18, 18, 22, 18, 18, 61, 65,  8,  9,  4, 82, 65, 10, 18, 65,  3,  4,
#             89,  9, 10, 10,  2, 10, 10, 22,  9, 20,  9, 23, 21,  3, 10, 15, 19, 11])

#     for i in range(len(curr_graph)):
#         print(i, curr_graph[i])
    
#     active_nodes = [9,4]
#     result = get_child_list(curr_graph)
#     print(result)

# # # [4, [29, 30, 32, 33, 36, 37, 45, 47, 50, 55, 56, 66, 70, 81, 91, 98, 100], 0, [5, 6, 7, 9, 10, 12, 22, 24, 61, 62, 65, 82, 89]]
# # if __name__ == "__main__":
# #     node_name = torch.load("node_classes.pt")
# #     node_name[80] = "Lemonade"
# #     node_name[81] = "glass"
# #     # print(node_name)
# #     for i,nn in enumerate(node_name):
# #         if"wine" in nn.lower():
# #             print(i,nn)
# #         if "glass" in nn.lower():
# #             print(i,nn)


# #[[0, 1, 2, 3, 4, 13, 15, 20, 60], [], [31, 34, 39, 40, 48, 52, 57, 58, 94], [54, 59, 88, 103], [5, 6, 7, 9, 10, 12, 22, 24, 61, 62, 65, 82, 89], [43], [8], [], [42, 68, 80], [29, 30, 32, 33, 36, 37, 45, 47, 50, 55, 56, 66, 70, 81, 91, 98, 100], [11, 21, 72, 85, 92, 93, 95, 96, 104], [107], [], [14, 23, 63], [26], [16, 17, 18, 19, 41, 64, 105], [], [], [46, 49, 73, 74, 76, 77, 86], [27, 28, 51, 106], [44, 99], [102], [35, 38, 75, 97], [25, 101], [53], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [78], [], [], [67, 69], [79, 84, 87], [], [], [71], [], [], [], [], [], [], [], [], [], [], [], [], [], [83], [], [], [], [], [], [], [90], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]