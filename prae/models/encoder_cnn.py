import torch
from torch import nn
import torch.nn.functional as F


class EncoderCNN(nn.Module):
    """CNN encoder, maps observation to latent state."""
    def __init__(self, channels, out_dim, h=32, w=32,
                 activation=F.relu):
        """
        Initialize CNN
        """
        super().__init__()

        self.cnn1 = nn.Conv2d(channels, 16, (3, 3))
        self.cnn2 = nn.Conv2d(16, 16, (3, 3))

        def conv2d_size_out(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(w))
        convh = conv2d_size_out(conv2d_size_out(h))
        self.linear_input_size = convw * convh * 16

        self.fc1 = nn.Linear(self.linear_input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, out_dim)

        self.act = activation

    def forward(self, obs):
        h = self.act(self.cnn1(obs))
        h = self.act(self.cnn2(h))
        h_flat = h.view(-1, self.linear_input_size)

        h = self.act(self.fc1(h_flat))
        h = self.act(self.fc2(h))
        z = self.fc3(h)
        return z

    @property
    def device(self):
        """
        """
        device = next(self.parameters()).device
        return device
