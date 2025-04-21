class ProactovityModule:
    def __init__(self, name):
        self.name = name

    def action_predictor(self, state):
        """
        Predicts the next action based on the current state.
        Args:
            state (dict): The current state of the environment.
        Returns:
            action (list): The predicted action. list of dictonaries. each dictionary mentions how an object is supposed to change (location or/and state).
            info_for_explainer (dict): Information for the explainer.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    def explainer(self, state):
        """
        Explains the predicted action based on the current state.
        Args:
            state (dict): The current state of the environment.
        Returns:
            explanation (str): The explanation for the predicted action.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    
class STOT(ProactovityModule):
    def __init__(self):
        name = "STOT"
        super().__init__(name)
        self.name = name
        

    def action_predictor(self, state):
        """
        Predicts the next action based on the current state.
        Args:
            state (dict): The current state of the environment.
        Returns:
            action (list): The predicted action. list of dictonaries. each dictionary mentions how an object is supposed to change (location or/and state).
            info_for_explainer (dict): Information for the explainer.
        """
        # Implement the logic for predicting actions based on the current state
        pass

    def explainer(self, state):
        """
        Explains the predicted action based on the current state.
        Args:
            state (dict): The current state of the environment.
        Returns:
            explanation (str): The explanation for the predicted action.
        """
        # Implement the logic for explaining the predicted action
        pass