import torch

pth = "data/HOMER/household1/processed/train/003.pt"
data = torch.load(pth)
print(len(data))
print(data[0].keys())
print(data[0]['prev_edges'].shape)


# common_data.json understanding
import json

pth = "data/HOMER/household0/processed/common_data.json"
with open(pth) as f:
    data = json.load(f)
print(data.keys())
# print(data['info'])
# print("########################################")
# for key in data['info'].keys():
#     print(key,":",data['info'][key])
# print(data['node_ids'])
# print("length of node_ids:",len(data['node_ids']))
# # print("########################################")
# print(data['node_classes'])
# print("length of node_classes:",len(data['node_classes']))

##########################################################

# print(data["node_idx_from_id"])
# print(data["edge_keys"])
# print(data["static_nodes"])
print(data["node_categories"])