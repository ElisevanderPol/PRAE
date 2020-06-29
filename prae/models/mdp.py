class MDP:
    """
    Abstract MDP wrapper class
    """
    def __init__(self, states, transitions, rewards, gamma):
        """
        """
        self.states = states
        self.transitions = transitions
        self.rewards = rewards
        self.gamma = gamma
