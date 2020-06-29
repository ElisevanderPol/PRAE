import torch
from torch import nn
import torch.nn.functional as F


class EncoderMLP(nn.Module):
    """MLP encoder, maps observation to latent state."""
    def __init__(self, in_dim, out_dim, mid=64, mid2=32,
                 activation=F.relu):
        """
        Initialize MLP
        """
        super().__init__()

        self.fc1 = nn.Linear(in_dim, mid)
        self.fc2 = nn.Linear(mid, mid2)
        self.fc3 = nn.Linear(mid2, out_dim)

        self.act = activation

    def forward(self, obs):
        """
        """
        if len(obs.shape) == 3:
            obs = obs.squeeze(1)
        h = self.act(self.fc1(obs))
        h = self.act(self.fc2(h))
        z = self.fc3(h)
        return z

    @property
    def device(self):
        """
        """
        device = next(self.parameters()).device
        return device
