import torch

class MovementTracker:
    def __init__(self, num_nodes=108):
        """
        Initializes the MovementTracker.
        Args:
            num_nodes (int): Number of nodes in the graph.
            mode (str): Mode of operation, old version is v1 helps support legacy stuff, v2 is the second version.
        """
        self.num_nodes = num_nodes
        self.movement_dict = {}
    
    def conver_to_unonehot(self, tsr: torch.Tensor):
        """
        Converts a one-hot encoded edges tensor to an unonehot representation.
        
        Args:
            tsr (Tensor): shape [1, num_nodes, num_nodes] or [num_nodes, num_nodes], one-hot encoded edges
        
        Returns:
            Tensor: shape [num_nodes], unonehot representation of edges
        """
        tsr = tsr.squeeze(0) if tsr.dim() == 3 else tsr  # Ensure it's 2D
        assert tsr.dim() == 2, "Input tensor must be 2D."
        assert tsr.shape[0] == self.num_nodes and tsr.shape[1]
        
        unonehot = torch.argmax(tsr, dim=1)  # Get the index of the max value in each row
        assert unonehot.shape == (self.num_nodes,), "Unonehot tensor must have shape [num_nodes]."
        return unonehot
        

    def detect(self, prev_edges_unonehot: torch.Tensor, edges_unonehot: torch.Tensor):
        """
        Detects which objects have changed parents between two graphs.

        Args:
            prev_edges_unonehot (Tensor): shape [num_nodes], current object's previous parent index
            edges_unonehot (Tensor): shape [num_nodes], current object's current parent index

        Returns:
            movement_detected (bool)
            movement_inds (List[int])
            movements (Tensor): shape [2, num_nodes]
        """
        assert prev_edges_unonehot.shape == (self.num_nodes,)
        assert edges_unonehot.shape == (self.num_nodes,)
        assert prev_edges_unonehot.dim() == 1
        assert edges_unonehot.dim() == 1

        diff = edges_unonehot - prev_edges_unonehot
        movement_inds = torch.nonzero(diff).squeeze(1).tolist()
        movement_detected = len(movement_inds) > 0
        movements = torch.stack((prev_edges_unonehot, edges_unonehot))  # shape: [2, num_nodes]
        condensed_movements = []
        for i in movement_inds:
            condensed_movements.append((i, prev_edges_unonehot[i].item(), edges_unonehot[i].item()))

        return movement_detected, movement_inds, movements, condensed_movements
    
    def update(self, routine_window):
        routine = routine_window[0] # only the first step is needed for movement tracking
        prev_edges = routine['edges']
        edges = routine['y_edges']
        prev_edges_unonehot = self.conver_to_unonehot(prev_edges)
        edges_unonehot = self.conver_to_unonehot(edges)
        movement_detected, movement_inds, movements, condensed_movements = self.detect(prev_edges_unonehot, edges_unonehot)
        if movement_detected:
            for node_idx, prev_parent, new_parent in condensed_movements:
                self.movement_dict[node_idx] = prev_parent
    
    def reset(self):
        """
        Resets the movement tracker.
        """
        self.movement_dict = {}
    
    
        
if __name__ == "__main__":
    # Example usage
    tracker = MovementTracker(num_nodes=5)
    prev_edges = torch.tensor([0, 1, 2, 3, 4])
    edges = torch.tensor([0, 1, 2, 4, 3])  # Node 3 and Node 4 changed parents

    movement_detected, movement_inds, movements, condesed_movs = tracker.detect(prev_edges, edges)
    print("Movement Detected:", movement_detected)
    print("Movement Indices:", movement_inds)
    print("Movements Tensor:\n", movements)
    print("Condensed Movements:", condesed_movs)