import torch
from torch import nn


class ActionEquivariantTrainer(nn.Module):
    """
    """
    def __init__(self, loss_function, train=True):
        """
        """
        super().__init__()
        self.loss_function = loss_function
        if train:
            self.mode = "train"
        else:
            self.mode = "valid"

    def forward(self, model, obs, action, next_obs, reward, neg_obs):
        """
        """
        # Abstract state
        z_c = model.state_encoder(obs)

        # Abstract action
        action_embedding = model.action_encoder(z_c, action)

        # transition in latent space
        z_l = z_c + action_embedding
        neg_obs = squeeze_samples(neg_obs)

        # embedding of negative sample(s)
        z_f = model.state_encoder(neg_obs)
        z_n = model.state_encoder(next_obs)

        # Predicted reward
        r_e = model.reward(z_l)

        # Loss components
        trans_loss, reward_loss, neg_loss = self.loss_function(z_c, z_l, z_n,
                                                               z_f, reward,
                                                               r_e)
        loss = trans_loss + reward_loss + neg_loss
        return loss


def squeeze_samples(x):
    """
    Reshape negative samples into the batch dimension
    """
    if len(x.shape) == 5:
        b, n, c, w, h = x.shape
        x = x.reshape(b*n, c, w, h)
    elif len(x.shape) == 3:
        b, n, h = x.shape
        x = x.reshape(b*n, h)
    return x
