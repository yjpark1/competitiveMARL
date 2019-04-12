import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    def __init__(self):
        super(Conv, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU())

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU())

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU())

    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        return h


class ActorNetwork(nn.Module):
    """
    MLP network (can be used as critic or actor)
    """
    def __init__(self, conv, out_dim, model_own=False, model_adv=False):
        """
        Inputs:
            agent_dim (int) : Number of dimensions for agents count
            input_dim (int) : Number of dimensions in input  (agents, observation)
            out_dim (int)   : Number of dimensions in output
            hidden_dim (int) : Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(ActorNetwork, self).__init__()
        self.model_own = model_own
        self.model_adv = model_adv
        self.out_dim = out_dim
        self.nonlin = F.relu

        # layers: input (1, 32, 32) / output (32, 4, 4)
        self.conv = conv
        self.dense1 = nn.Linear(32 * 16, self.out_dim)

        # approximate model layer
        if self.model_own:
            self.layer_model_own = nn.ConvTranspose2d(32, 1, kernel_size=8, stride=8)
        if self.model_adv:
            self.layer_model_adv = nn.Linear(32 * 16, self.out_dim)

    def forward(self, obs):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): policy, next_state
        """
        h = self.conv(obs)
        policy = self.dense1(h.reshape(h.size(0), -1))

        # outputs
        out = [policy]
        if self.model_own:
            next_state = self.layer_model_own(h)
            out += [next_state]
        if self.model_adv:
            adv_action = self.layer_model_adv(h.reshape(h.size(0), -1))
            out += [adv_action]

        return out


class CriticNetwork(nn.Module):
    """
    MLP network (can be used as critic or actor)
    """
    def __init__(self, conv, out_dim=1, model_own=False, model_adv=False):
        """
        Inputs:
            agent_dim (int) : Number of dimensions for agents count
            input_dim (int): Number of dimensions in input  (agents, observation)
            out_dim (int): Number of dimensions in output

            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(CriticNetwork, self).__init__()
        self.model_own = model_own
        self.model_adv = model_adv
        self.out_dim = out_dim
        self.nonlin = F.relu

        # layers: input (1, 32, 32) / output (32, 4, 4)
        self.conv = conv
        self.dense1 = nn.Linear(32 * 16 + 4, self.out_dim)
        
        # approximate model layer
        if self.model_own:
            self.layer_model_own = nn.Linear(32 * 16 + 4, self.out_dim)
        if self.model_adv:
            self.layer_model_adv = nn.Linear(32 * 16 + 4, self.out_dim)

    def forward(self, obs, action, adv_action=None):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Q-function
            out (PyTorch Matrix): reward
        """
        h = self.conv(obs)
        h = h.reshape(h.size(0), -1)
        obs_act = torch.cat([h] + [action], dim=-1)
        Q = self.dense1(obs_act)

        # outputs
        out = [Q]
        if self.model_own:
            own_r = self.layer_model_own(obs_act)
            out += [own_r]
        if self.model_adv:
            adv_r = self.layer_model_own(obs_act)
            out += [adv_r]

        return out


if __name__ == '__main__':
    import numpy as np
    conv = Conv()
    actor = ActorNetwork(conv, out_dim=4, model_own=False, model_adv=False)
    critic = CriticNetwork(conv, out_dim=1, model_own=False, model_adv=False)
    actor.parameters()
    critic.parameters()
    wa = [x for x in actor.conv.parameters()]
    critic.conv.parameters()
    wc = [x for x in critic.conv.parameters()]

    x = np.random.uniform(size=(128, 3, 10))
    x = torch.tensor(x, dtype=torch.float32)
    y = actor.forward(x)
