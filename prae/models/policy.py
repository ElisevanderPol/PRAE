import numpy as np

import torch
import torch.nn.functional as F

from prae.helpers import state_prep_np, state_prep_torch, tile_z
from prae.distances import HingedSquaredEuclidean


class Policy:
    """
    """

    def __init__(self, network, q_values, mdp, eps):
        """
        """
        self.network = network
        self.q_values = torch.transpose(q_values, 1, 0).to(self.network.device)
        self.mdp = mdp
        self.distance = HingedSquaredEuclidean(eps=eps)

    def select_action(self, state, eta=1e-20):
        """
        """
        device = self.network.device

        abstract_states = state_prep_torch(self.mdp.abstract_states, device)
        state = state_prep_np(state, device)
        z = self.network.state_encoder(state)
        if len(z.shape) == 1:
            z = z.unsqueeze(0)

        weighted_q = self.interpolate(z, abstract_states, eta=eta)

        value, action = torch.max(weighted_q, dim=0)

        q_list = [q for q in weighted_q.squeeze()]

        action_list = [i for i, q in enumerate(q_list)
                       if 0.5*(q-value)**2 < 1e-8]
        action = np.random.choice(action_list)
        return action

    def interpolate(self, z, abstract_states, eta=1e-20):
        """
        """
        z = tile_z(z, abstract_states.shape[0])

        distances = self.distance(abstract_states, z)

        weights = F.softmax(-distances/eta, dim=0).unsqueeze(1)

        weighted_q = torch.mm(self.q_values, weights)
        return weighted_q
