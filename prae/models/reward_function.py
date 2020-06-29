import torch
from torch import nn
import torch.nn.functional as F


class RewardPredictor(nn.Module):
    """
    """
    def __init__(self, input_size, output_size=1, activation=F.relu):
        super().__init__()

        self.fc_in = nn.Linear(input_size, 64)
        self.fc_out = nn.Linear(64, output_size)

        self.act1 = activation

    def forward(self, z):
        """
        """
        h = self.act1(self.fc_in(z))
        r = self.fc_out(h)
        return r

    def get_rewards(self, nodes, reward_type='learned'):
        """
        """
        if reward_type == 'learned':
            rewards = torch.max(self.forward(nodes), dim=1)[1] - 1
        elif reward_type == 'goal':
            rewards = torch.zeros_like(nodes)[:, :, 0].unsqueeze(2)
            rewards[:, -1] = 1.
        return rewards
