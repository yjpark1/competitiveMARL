import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x, h0=None):
        if len(x.size()) == 3:
            # TimeDistributed + Dense
            # shape: (seq_len, batch, features)
            seq_len, b, n = x.size(0), x.size(1), x.size(2)
            # merge batch and seq dimensions
            x_reshape = x.contiguous().view(seq_len * b, n)

        elif len(x.size()) == 4:
            # TimeDistributed + RNN
            # shape: (seq_len, batch, agents, features)
            seq_len, b, n, d = x.size(0), x.size(1), x.size(2), x.size(3)
            # merge batch and seq dimensions
            x_reshape = x.contiguous().view(seq_len, b * n, d)

            # return
        if isinstance(self.module, nn.Linear):
            # TimeDistributed + Dense
            y = self.module(x_reshape)
            # We have to reshape Y
            # shape: (seq_len, batch, documents, features)
            y = y.contiguous().view(seq_len, b, n, y.size()[-1])
            return y

        elif isinstance(self.module, nn.LSTM):
            # TimeDistributed + RNN
            if h0 is not None:
                h0, c0 = h0
                h0 = h0.contiguous().view(h0.size()[0], b * n, h0.size()[-1])
                c0 = c0.contiguous().view(c0.size()[0], b * n, c0.size()[-1])
                h0 = (h0, c0)

            y, (h, c) = self.module(x_reshape, h0)
            # shape: (seq_len, batch x agents, features)
            y = y.contiguous().view(seq_len, b, n, y.size()[-1])
            # shape: (seq_len, batch, agents, features)

            # shape: (batch, agents, features)
            h = h.contiguous().view(h.size()[0], b, n, h.size()[-1])
            c = c.contiguous().view(c.size()[0], b, n, c.size()[-1])
            return y, (h, c)

        elif isinstance(self.module, nn.GRU):
            # TimeDistributed + RNN
            y, h = self.module(x_reshape, h0)
            # shape: (seq_len, batch x agents, features)
            y = y.contiguous().view(seq_len, b, n, y.size()[-1])
            # shape: (seq_len, batch, agents, features)

            # shape: (batch, agents, features)
            h = h.contiguous().view(1, b, n, h.size()[-1])
            return y, h

        else:
            raise ImportError('Not Supported Layers!')


class ActorNetwork(nn.Module):
    """
    MLP network (can be used as critic or actor)
    """
    def __init__(self, nb_agents, input_dim, out_dim):
        """
        Inputs:
            agent_dim (int) : Number of dimensions for agents count
            input_dim (int) : Number of dimensions in input  (agents, observation)
            out_dim (int)   : Number of dimensions in output
            hidden_dim (int) : Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(ActorNetwork, self).__init__()
        self.nb_agents = nb_agents
        self.nonlin = F.relu
        self.dense1 = nn.Linear(input_dim, 128)
        self.lstmTime = TimeDistributed(nn.LSTM(128, 128, num_layers=1, bidirectional=False))
        # return sequence is not exist in pytorch. Instead, output will return with first dimension for sequences.
        self.bilstmAgent = TimeDistributed(nn.LSTM(128, 64, num_layers=1, bidirectional=True))
        self.dense2 = nn.Linear(128, out_dim)
        self.dense3 = nn.Linear(128, input_dim)
        self.hState = None

    def forward(self, obs):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): policy, next_state
        """
        # obs: (time, batch, agent, feature)
        hid = F.relu(self.dense1(obs))
        hid, hcTime = self.lstmTime(hid, h0=self.hState)
        hid = F.relu(hid)
        hid = hid.permute(2, 1, 0, 3)
        hid, hcAgent = self.bilstmAgent(hid, h0=None)
        hid = F.relu(hid)
        hid = hid.permute(2, 1, 0, 3)
        policy = self.dense2(hid)
        policy = nn.Softmax(dim=-1)(policy)
        next_state = self.dense3(hid)
        return policy, next_state, hcTime

    def init_hidden(self, batch_size):
        if self.lstmTime.module.bidirectional:
            return torch.zeros(2, batch_size, self.nb_agents, 128, requires_grad=True)
        else:
            return torch.zeros(1, batch_size, self.nb_agents, 128, requires_grad=True)


class CriticNetwork(nn.Module):
    """
    MLP network (can be used as critic or actor)
    """
    def __init__(self, nb_agents, input_dim, out_dim):
        """
        Inputs:
            agent_dim (int) : Number of dimensions for agents count
            input_dim (int): Number of dimensions in input  (agents, observation)
            out_dim (int): Number of dimensions in output

            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(CriticNetwork, self).__init__()

        self.nb_agents = nb_agents
        self.nonlin = F.relu
        self.dense1 = nn.Linear(input_dim, 128)
        self.lstmTime = TimeDistributed(nn.LSTM(128, 128, num_layers=1, bidirectional=False))
        # return sequence is not exist in pytorch. Instead, output will return with first dimension for sequences.
        self.lstmAgent = TimeDistributed(nn.LSTM(128, 128, num_layers=1, bidirectional=False))
        self.dense2 = nn.Linear(128, out_dim)
        self.dense3 = nn.Linear(128, out_dim)
        self.hState = None

    def forward(self, obs, action):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Q-function
            out (PyTorch Matrix): reward
        """
        obs_act = torch.cat((obs, action), dim=-1)
        hid = F.relu(self.dense1(obs_act))
        hid, hcTime = self.lstmTime(hid, h0=self.hState)
        hid = F.relu(hid)
        hid, hcAgent = self.lstmAgent(hid, h0=None)
        hid = F.relu(hid[:, :, -1, :])
        Q = self.dense2(hid)
        r = self.dense3(hid)
        return Q, r, hcTime

    def init_hidden(self, batch_size):
        if self.lstmTime.module.bidirectional:
            return torch.zeros(2, batch_size, self.nb_agents, 128, requires_grad=True)
        else:
            return torch.zeros(1, batch_size, self.nb_agents, 128, requires_grad=True)


if __name__ == '__main__':
    actor = ActorNetwork(input_dim=10, out_dim=5)
    critic = CriticNetwork(input_dim=10 + 5, out_dim=1)

    s = torch.randn(15, 128, 3, 10)
    pred_actor = actor.forward(s)
    print(pred_actor[0].size())
    print(pred_actor[1].size())
    print(actor.hState)
    actor.hState = actor.init_hidden(batch_size=128)

    h, c = pred_actor[2]
    h.size()
    c.size()

    a = torch.randn(15, 128, 3, 5)
    pred_critic = critic.forward(s, a)
    print(critic.hState)
    critic.hState = critic.init_hidden(batch_size=128)

