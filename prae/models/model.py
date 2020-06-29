from torch import nn


class Model(nn.Module):
    """
    """
    def __init__(self, state_encoder, action_encoder, reward):
        """
        """
        super().__init__()
        self.state_encoder = state_encoder
        self.action_encoder = action_encoder
        self.reward = reward

    def forward(self, x):
        """
        """
        raise NotImplementedError("This model is a placeholder")

    def train(self, boolean):
        """
        """
        self.state_encoder.train = boolean
        self.action_encoder.train = boolean
        self.reward.train = boolean


    @property
    def device(self):
        """
        """
        return self.state_encoder.device
