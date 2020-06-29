class UniformPolicy:
    """
    Sample action uniformly independent of the state
    """

    def __init__(self, action_space):
        """
        """
        self.action_space = action_space

    def select_action(self, state):
        """
        """
        return self.action_space.sample()
