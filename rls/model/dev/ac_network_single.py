import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorNetwork(nn.Module):
    """
    MLP network (can be used as critic or actor)
    """
    def __init__(self, input_dim, out_dim):
        """
        Inputs:
            agent_dim (int) : Number of dimensions for agents count
            input_dim (int) : Number of dimensions in input  (agents, observation)
            out_dim (int)   : Number of dimensions in output
            hidden_dim (int) : Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(ActorNetwork, self).__init__()
        self.nb_action = out_dim
        self.nonlin = F.relu
        self.dense1 = nn.Linear(input_dim, 64)
        # return sequence is not exist in pytorch. Instead, output will return with first dimension for sequences.
        self.dense2 = nn.Linear(64, 64)
        self.dense3 = nn.Linear(64, out_dim)

    def forward(self, obs):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        out = F.relu(self.dense1(obs))
        out = F.relu(self.dense2(out))
        out = self.dense3(out)
        out = nn.Softmax(dim=-1)(out)
        return out


class CriticNetwork(nn.Module):
    """
    MLP network (can be used as critic or actor)
    """
    def __init__(self, input_dim, out_dim):
        """
        Inputs:
            agent_dim (int) : Number of dimensions for agents count
            input_dim (int): Number of dimensions in input  (agents, observation)
            out_dim (int): Number of dimensions in output

            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(CriticNetwork, self).__init__()

        self.nonlin = F.relu
        self.dense1 = nn.Linear(input_dim, 64)
        # return sequence is not exist in pytorch. Instead, output will return with first dimension for sequences.
        self.dense2 = nn.Linear(64, 64)
        self.dense3 = nn.Linear(64, out_dim)

    def forward(self, obs, action):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        obs_act = torch.cat((obs, action), dim=-1)
        out = F.relu(self.dense1(obs_act))
        out = F.relu(self.dense2(out))
        out = self.dense3(out)
        return out
