import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        t, n = x.size(0), x.size(1)
        # merge batch and seq dimensions
        x_reshape = x.contiguous().view(t * n, x.size(2))
        y = self.module(x_reshape)
        # We have to reshape Y
        y = y.contiguous().view(t, n, y.size()[1])
        return y


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
        self.out_dim = out_dim
        self.nonlin = F.relu
        self.dense1 = TimeDistributed(nn.Linear(input_dim, 64))
        # return sequence is not exist in pytorch. Instead, output will return with first dimension for sequences.
        self.bilstm = nn.LSTM(64, 32, num_layers=1,
                              batch_first=True, bidirectional=True)
        if type(self.out_dim) is list:
            self.dense2_1 = TimeDistributed(nn.Linear(64, out_dim[0]))
            self.dense2_2 = TimeDistributed(nn.Linear(64, out_dim[1]))
        else:
            self.dense2 = TimeDistributed(nn.Linear(64, out_dim))

        self.dense3 = TimeDistributed(nn.Linear(64, input_dim))

    def forward(self, obs):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): policy, next_state
        """
        hid = F.relu(self.dense1(obs))
        hid, _ = self.bilstm(hid, None)
        hid = F.relu(hid)
        if type(self.out_dim) is list:
            policy = [self.dense2_1(hid), self.dense2_2(hid)]
        else:
            policy = self.dense2(hid)
        next_state = self.dense3(hid)
        return policy, next_state


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
        self.dense1 = TimeDistributed(nn.Linear(input_dim, 64))
        # return sequence is not exist in pytorch. Instead, output will return with first dimension for sequences.
        self.lstm = nn.LSTM(64, 64, num_layers=1,
                            batch_first=True, bidirectional=False)
        self.dense2 = nn.Linear(64, out_dim)
        self.dense3 = nn.Linear(64, out_dim)

    def attention_net(self, lstm_output, final_state):
        """
        Now we will incorporate Attention mechanism in our LSTM model. In this new model, we will use attention to compute soft alignment score corresponding
        between each of the hidden_state and the last hidden_state of the LSTM. We will be using torch.bmm for the batch matrix multiplication.

        Arguments
        ---------

        lstm_output : Final output of the LSTM which contains hidden layer outputs for each sequence.
        final_state : Final time-step hidden state (h_n) of the LSTM

        ---------

        Returns : It performs attention mechanism by first computing weights for each of the sequence present in lstm_output and and then finally computing the
                  new hidden state.

        Tensor Size :
                    hidden.size() = (batch_size, hidden_size)
                    attn_weights.size() = (batch_size, num_seq)
                    soft_attn_weights.size() = (batch_size, num_seq)
                    new_hidden_state.size() = (batch_size, hidden_size)

        """
        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

        return new_hidden_state

    def forward(self, obs, action):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Q-function
            out (PyTorch Matrix): reward
        """
        if type(action) is list:
            obs_act = torch.cat([obs] + action, dim=-1)
        else:
            obs_act = torch.cat([obs] + [action], dim=-1)
        out = F.relu(self.dense1(obs_act))

        output, (final_hidden_state, final_cell_state) = self.lstm(out, None)
        # final_hidden_state.size() = (1, batch_size, hidden_size)
        # output.size() = (batch_size, num_seq, hidden_size)
        attn_output = self.attention_net(output, final_hidden_state)
        Q = self.dense2(attn_output)
        r = self.dense3(attn_output)
        return Q, r
