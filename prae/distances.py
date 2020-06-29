import torch
from torch import nn


def square_dist(x, y, dim=1):
    """
    """
    return (x-y).pow(2).sum(dim=dim)


class HingedSquaredEuclidean(nn.Module):
    """
    Euclidean distance with hinge on negatives
    """
    def __init__(self, eps=1.0):
        """
        """
        super().__init__()
        self.distance = square_dist
        self.eps = eps

    def forward(self, x, y, dim=1):
        """
        """
        return 0.5 * self.distance(x, y, dim)


    def negative_distance(self, x, y, dim=1):
        """
        """
        dist = self.forward(x, y, dim)
        neg_dist = torch.max(torch.zeros_like(dist), self.eps-dist)
        return neg_dist

    def pairwise_distance(self, x):
        dist_mat = []
        for r1 in x:
            dist = self.forward(x, r1)
            dist_mat.append(dist.unsqueeze(1))
        dist_mat = torch.cat(dist_mat, 1)
        return dist_mat

    def pairwise_negative(self, dist_mat):
        """
        """
        neg_dist = torch.max(torch.zeros_like(dist_mat), self.eps-dist_mat)
        return neg_dist
