import torch

from collections import defaultdict

class SceneContextExtractor:
    def __init__(self, parents, names, active_nodes=None):
        ########!!!!!!! Hardcoded due to root conection connect 0->bathroom node. !!!!!!!!!!! ##################
        # torch.save(names,"test_names1.pt")
        # torch.save(parents,"test_parents1.pt")
        # torch.save(active_nodes,"test_active_nodes1.pt")

        self.roots_childs = [0,4,13,15,20] 
        active_nodes = set(active_nodes) if active_nodes else set()
        
        parents,names, active_nodes = self.managing_root(parents,names,active_nodes)
        # raise Exception("stop")
        self.active_nodes = set(active_nodes) if active_nodes else set()
        self.parents = parents
        self.names = names
        print("building tree")
        self.tree, self.root = self._build_tree()
        print("finding required nodes")
        self.required_nodes = self._find_required_nodes()

    def managing_root(self, parents, names, active_nodes):
        # Increment all indices by 1
        active_nodes = [x + 1 for x in active_nodes]
        parents = [p + 1 for p in parents]

        # Connect predefined children to new root (index 0)
        for i in self.roots_childs:
            parents[i] = 0

        # Add new root
        new_parents = [-1] + parents
        new_names = ["house"] + names

        return new_parents, new_names, active_nodes

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
        for i,node in enumerate(self.active_nodes):
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
    
    def get_ordered_leaf_active_paths(self):
        """
        Returns only deepest active paths (skip parent if child is active).
        """
        paths = []

        def dfs(node, path_so_far):
            path_so_far.append(self.names[node])
            children = self.tree.get(node, [])

            # Recurse first
            child_has_active = False
            for child in children:
                if self._subtree_has_active(child):
                    dfs(child, path_so_far[:])
                    child_has_active = True

            # If current is active and no active children: it's a leaf-active path
            if node in self.active_nodes and not child_has_active:
                paths.append(" > ".join(path_so_far))

        dfs(self.root, [])
        str_paths = ""
        for path in paths:
            str_paths+=path+"\n"
        return str_paths
        # return paths
    def _subtree_has_active(self, node):
        """
        Helper to check if the current node or any of its descendants is in active_nodes.
        """
        if node in self.active_nodes:
            return True
        return any(self._subtree_has_active(child) for child in self.tree.get(node, []))


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
    # node_names = torch.load("node_classes.pt")
    # for i in [0, 1, 2, 3, 4, 13, 15, 20, 60]:
    #     print(i,node_names[i])
    # curr_graph = torch.tensor([ 0,  0,  0,  0,  0,  4,  4,  4,  6,  4,  4, 10,  4,  0, 13,  0, 15, 15,
    #         15, 15,  0, 10,  4, 13,  4, 23, 14, 19, 19,  9,  9,  2,  9,  9,  2, 22,
    #         9,  9, 22,  2,  2, 15,  8,  5, 20,  9, 18,  9,  2, 18,  9, 19,  2, 24,
    #         3,  9,  9,  2,  2,  3,  0,  4,  4, 13, 15,  4,  9, 64,  8, 64,  9, 68,
    #         10, 18, 18, 22, 18, 18, 61, 65,  8,  9,  4, 82, 65, 10, 18, 65,  3,  4,
    #         89,  9, 10, 10,  2, 10, 10, 22,  9, 20,  9, 23, 21,  3, 10, 15, 19, 11])
    # parents = curr_graph.tolist()
    # names = node_names
    # sce = SceneContextExtractor(parents, names, active_nodes=[19])
    # sce.visualize()
    # print(sce.get_tree_string())
    #

    # Test1: 
    names = torch.load("test_names1.pt")
    parents = torch.load("test_parents1.pt")
    active_nodes = torch.load("test_active_nodes1.pt")
    print("parents[0]",parents[0])
    print(names)
    print(active_nodes)
    # names = names.tolist()
    # active_nodes = active_nodes.tolist()
    for act_nodes in active_nodes:
        print(names[act_nodes])
    sce = SceneContextExtractor(parents, names, active_nodes=active_nodes)
    # print("###### Required Nodes ######")
    # for req in sce.required_nodes:
    #     print("req:",names[req]," ",req)
        
    
    sce.visualize()
    pths = sce.get_ordered_leaf_active_paths()
    print(pths)
