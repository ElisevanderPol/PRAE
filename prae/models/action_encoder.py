import torch
from torch import nn
import torch.nn.functional as F


class ActionEncoder(nn.Module):
    """
    """
    def __init__(self, n_dim, n_actions, hidden_dim=100, temp=1.):
        """
        """
        super().__init__()
        self.linear1 = nn.Linear(n_dim+n_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, n_dim)
        self.n_actions = n_actions

    def forward(self, z, a):
        """
        """
        if len(z.shape) == 3:
            z = z.squeeze(0)
            a = a.repeat(z.shape[0], 1)
        za = torch.cat([z, a], dim=1)
        # State-dependent action-embedding
        za = F.relu(self.linear1(za))
        zt = self.linear2(za)
        return zt
