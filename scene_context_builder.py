import torch



from collections import defaultdict

from collections import defaultdict

class SceneContextExtractor:
    def __init__(self, parents, names, active_nodes=None):
        ########!!!!!!! Hardcoded due to root conection connect 0->bathroom node. !!!!!!!!!!! ##################
        self.roots_childs = [4,13,15,20] 
        parents,names, active_nodes = self.managing_root(parents,names,active_nodes)
        self.parents = parents
        self.names = names
        self.active_nodes = set(active_nodes) if active_nodes else set()
        self.tree, self.root = self._build_tree()
        self.required_nodes = self._find_required_nodes()

    def managing_root(self,parents,names,active_nodes):
        active_nodes = torch.tensor(active_nodes, dtype=torch.long)
        active_nodes+=1
        active_nodes = active_nodes.tolist()

        parents = torch.tensor(parents, dtype=torch.long)
        parents+=1
        for i in self.roots_childs:
            parents[i] = 0
        parents = parents.tolist()
        new_parents = [-1]
        new_parents.extend(parents)
        new_names = ["house"]
        new_names.extend(names)
        return new_parents,new_names,active_nodes

    def _build_tree(self):
        tree = defaultdict(list)
        root = None
        for idx, parent_idx in enumerate(self.parents):
            if parent_idx == -1:
                root = idx
            else:
                tree[parent_idx].append(idx)
        return tree, root

    def _find_required_nodes(self):
        required = set(self.active_nodes)
        for node in self.active_nodes:
            while node != -1:
                required.add(node)
                node = self.parents[node]
        return required

    def _build_tree_string(self, node, prefix="", is_last=True):
        lines = []
        connector = "└── " if is_last else "├── "
        lines.append(prefix + connector + self.names[node])

        children = self.tree.get(node, [])
        display_children = [
            child for child in children
            if child in self.required_nodes or self.parents[child] in self.active_nodes
        ]

        for i, child in enumerate(display_children):
            is_last_child = (i == len(display_children) - 1)
            new_prefix = prefix + ("    " if is_last else "│   ")
            lines.extend(self._build_tree_string(child, new_prefix, is_last_child))

        return lines

    def visualize(self):
        if self.root is None:
            print("No root found.")
            return

        print(self.names[self.root])
        top_level = [
            child for child in self.tree[self.root]
            if child in self.required_nodes or self.root in self.active_nodes
        ]

        for i, child in enumerate(top_level):
            is_last = (i == len(top_level) - 1)
            self._print_tree(child, "", is_last)

    def _print_tree(self, node, prefix="", is_last=True):
        connector = "└── " if is_last else "├── "
        print(prefix + connector + self.names[node])

        children = self.tree.get(node, [])
        display_children = [
            child for child in children
            if child in self.required_nodes or self.parents[child] in self.active_nodes
        ]

        for i, child in enumerate(display_children):
            is_last_child = (i == len(display_children) - 1)
            new_prefix = prefix + ("    " if is_last else "│   ")
            self._print_tree(child, new_prefix, is_last_child)

    def get_tree_string(self):
        if self.root is None:
            return "No root found."

        lines = [self.names[self.root]]
        top_level = [
            child for child in self.tree[self.root]
            if child in self.required_nodes or self.root in self.active_nodes
        ]

        for i, child in enumerate(top_level):
            is_last = (i == len(top_level) - 1)
            lines.extend(self._build_tree_string(child, "", is_last))

        return "\n".join(lines)


# class SceneContextExtractor:
#     def __init__(self, parents, names, active_nodes=None):
#         self.roots_childs = [4,13,15,20] ############## Hardcoded due to root conection connect 0->bathroom node.
#         parents,names, active_nodes = self.managing_root(parents,names,active_nodes)
#         print(parents)
#         print(names)
#         self.parents = parents
#         self.names = names
#         self.active_nodes = set(active_nodes) if active_nodes else set()
#         self.tree, self.root = self._build_tree()
#         self.required_nodes = self._find_required_nodes()
#     def managing_root(self,parents,names,active_nodes):
#         active_nodes = torch.tensor(active_nodes, dtype=torch.long)
#         active_nodes+=1
#         active_nodes = active_nodes.tolist()

#         parents = torch.tensor(parents, dtype=torch.long)
#         parents+=1
#         for i in self.roots_childs:
#             parents[i] = 0
#         parents = parents.tolist()
#         new_parents = [-1]
#         new_parents.extend(parents)
#         new_names = ["house"]
#         new_names.extend(names)
#         return new_parents,new_names,active_nodes
#     def _build_tree(self):
#         tree = defaultdict(list)
#         root = None
#         for idx, parent_idx in enumerate(self.parents):
#             if parent_idx == -1:
#                 root = idx
#             else:
#                 tree[parent_idx].append(idx)
#         return tree, root

#     def _find_required_nodes(self):
#         required = set(self.active_nodes)
#         for node in self.active_nodes:
#             while node != -1:
#                 required.add(node)
#                 node = self.parents[node]
#         return required

#     def _print_tree(self, node, prefix="", is_last=True):
#         connector = "└── " if is_last else "├── "
#         print(prefix + connector + self.names[node])

#         children = self.tree.get(node, [])
#         display_children = [
#             child for child in children
#             if child in self.required_nodes or self.parents[child] in self.active_nodes
#         ]

#         for i, child in enumerate(display_children):
#             is_last_child = (i == len(display_children) - 1)
#             new_prefix = prefix + ("    " if is_last else "│   ")
#             self._print_tree(child, new_prefix, is_last_child)

#     def visualize(self):
#         if self.root is None:
#             print("No root found.")
#             return

#         print(self.names[self.root])
#         top_level = [
#             child for child in self.tree[self.root]
#             if child in self.required_nodes or self.root in self.active_nodes
#         ]

#         for i, child in enumerate(top_level):
#             is_last = (i == len(top_level) - 1)
#             self._print_tree(child, "", is_last)

if __name__=="__main__":
    node_names = torch.load("node_classes.pt")
    for i in [0, 1, 2, 3, 4, 13, 15, 20, 60]:
        print(i,node_names[i])
    curr_graph = torch.tensor([ 0,  0,  0,  0,  0,  4,  4,  4,  6,  4,  4, 10,  4,  0, 13,  0, 15, 15,
            15, 15,  0, 10,  4, 13,  4, 23, 14, 19, 19,  9,  9,  2,  9,  9,  2, 22,
            9,  9, 22,  2,  2, 15,  8,  5, 20,  9, 18,  9,  2, 18,  9, 19,  2, 24,
            3,  9,  9,  2,  2,  3,  0,  4,  4, 13, 15,  4,  9, 64,  8, 64,  9, 68,
            10, 18, 18, 22, 18, 18, 61, 65,  8,  9,  4, 82, 65, 10, 18, 65,  3,  4,
            89,  9, 10, 10,  2, 10, 10, 22,  9, 20,  9, 23, 21,  3, 10, 15, 19, 11])
    parents = curr_graph.tolist()
    names = node_names
    sce = SceneContextExtractor(parents, names, active_nodes=[19])
    sce.visualize()
    print(sce.get_tree_string())