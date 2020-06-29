import torch
from torch import nn
from prae.distances import square_dist, HingedSquaredEuclidean


class Loss(nn.Module):
    """
    """
    def __init__(self, hinge, neg=True, rew=True):
        """
        """
        super().__init__()
        self.reward_loss = square_dist
        # If False, no negative sampling
        self.neg = neg
        # If False, no reward loss
        self.rew = rew
        self.distance = HingedSquaredEuclidean(eps=hinge)

    def forward(self, z_c, z_l, z_n, z_f, r, r_e):
        """
        """
        # Transition loss
        transition_loss = self.distance.distance(z_n, z_l).mean()

        # Reward loss
        if self.rew:
            reward_loss = 0.5 * self.reward_loss(r, r_e).mean()
        else:
            reward_loss = torch.zeros_like(transition_loss)

        # Negative los
        if self.neg:
            z_n = tile(z_n, z_f)

            batch_size = z_c.shape[0]

            negative_loss = self.distance.negative_distance(z_n, z_f).sum()/batch_size
        else:
            negative_loss = torch.zeros_like(transition_loss)

        return transition_loss, reward_loss, negative_loss



def tile(embedding, example):
    """
    """
    n = example.shape[0]//embedding.shape[0]
    embedding = embedding.unsqueeze(1).repeat(1, n, 1)
    embedding = squeeze_embedding(embedding)
    return embedding


def squeeze_embedding(x):
    """
    """
    b, n, d = x.shape
    x = x.reshape(b*n, d)
    return x
