# reference: https://github.com/vy007vikas/PyTorch-ActorCriticRL/blob/master/train.py
from __future__ import division
import torch
import numpy as np
import shutil
from rls import arglist
import copy
from rls.utils import to_categorical

GAMMA = 0.95


class Trainer:
    def __init__(self, actor, critic, memory_own, memory_adv,
                 model_own=False, model_adv=False, action_type='Discrete'):
        """
        DDPG for categorical action
        """
        self.device = torch.device('cuda:0')
        # self.device = torch.device('cpu')

        self.model_own = model_own
        self.model_adv = model_adv

        self.iter = 0
        self.actor = actor.to(self.device)
        self.target_actor = copy.deepcopy(actor).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), arglist.actor_learning_rate)

        self.critic = critic.to(self.device)
        self.target_critic = copy.deepcopy(critic).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), arglist.critic_learning_rate)

        self.memory_own = memory_own
        self.memory_adv = memory_adv
        self.nb_actions = actor.out_dim
        self.action_type = action_type  # MultiDiscrete

        self.target_actor.eval()
        self.target_critic.eval()

    def soft_update(self, target, source, tau):
        """
        Copies the parameters from source network (x) to target network (y) using the below update
        y = TAU*x + (1 - TAU)*y
        :param target: Target network (PyTorch)
        :param source: Source network (PyTorch)
        :return:
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )

    def hard_update(self, target, source):
        """
        Copies the parameters from source network to target network
        :param target: Target network (PyTorch)
        :param source: Source network (PyTorch)
        :return:
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def process_obs(self, obs):
        obs = np.array([np.stack(obs)], dtype='float32')
        return obs

    def process_action(self, actions):
        actions = np.expand_dims(actions, axis=0)
        actions = torch.from_numpy(actions)
        return actions

    def process_reward(self, rewards):
        rewards = np.array(rewards, dtype='float32')
        rewards = np.sum(rewards)
        rewards = torch.tensor(rewards)
        return rewards

    def process_done(self, done):
        done = np.all(done)
        done = np.array(done, dtype='float32')
        done = torch.from_numpy(done)
        return done

    def to_onehot(self, actions):
        actions = to_categorical(actions, num_classes=self.nb_actions)
        actions = actions.astype('float32')
        return actions

    def get_exploration_action(self, state, mode='train'):
        """
        gets the action from actor added with exploration noise
        :param state: state (Numpy array)
        :return: sampled action (Numpy array)
        """
        state = self.process_obs(state)
        state = torch.from_numpy(state)
        state = state.to(self.device)

        out = self.actor.forward(state)
        logits = out[0]
        logits = logits.detach()
        if mode == 'train':
            actions = self.gumbel_softmax(logits, hard=True)
            actions = actions.cpu().numpy()
        elif mode == 'test':
            actions = torch.argmax(logits, dim=-1)
            actions = actions.cpu().numpy()
            actions = self.to_onehot(actions)
        actions = actions[0]

        return actions

    def gumbel_softmax(self, x, hard=True):
        n, t = x.size(0), x.size(1)
        # merge batch and seq dimensions
        x_reshape = x.contiguous().view(n * t, x.size(2))
        y = torch.nn.functional.gumbel_softmax(x_reshape, hard=hard)
        # We have to reshape Y
        y = y.contiguous().view(n, t, y.size()[1])
        return y

    def set_index(self):
        index = self.memory_own.make_index(arglist.batch_size)
        return index

    def process_batch(self, memory, index):
        # collect replay sample from all agents
        s0, a0, r, s1, d = memory.sample_index(index)

        s0 = torch.tensor(s0, dtype=torch.float32)
        a0 = torch.tensor(a0, dtype=torch.float32)
        r = torch.tensor(r, dtype=torch.float32)
        s1 = torch.tensor(s1, dtype=torch.float32)
        d = torch.tensor(d, dtype=torch.float32)

        return s0, a0, r, s1, d

    def optimize(self):
        """
        Samples a random batch from replay memory and performs optimization
        critic outputs: [Q, r_own, r_adv]
        actor outputs: [policy, s_own, a_adv]
        """
        index = self.set_index()
        s0, a0, r, s1, d = self.process_batch(self.memory_own, index)
        _, a0_adv, r_adv, _, _ = self.process_batch(self.memory_adv, index)

        s0 = s0.to(self.device)
        a0 = a0.to(self.device)
        r = r.to(self.device)
        s1 = s1.to(self.device)
        d = d.to(self.device)

        a0_adv = a0_adv.to(self.device)
        r_adv = r_adv.to(self.device)

        # ---------------------- optimize critic ----------------------
        # Use target actor exploitation policy here for loss evaluation
        logits1 = self.target_actor.forward(s1)[0]
        a1 = self.gumbel_softmax(logits1)

        q_next = self.target_critic.forward(s1, a1)[0]
        q_next = q_next.detach()
        q_next = torch.squeeze(q_next)

        # Loss: TD error
        # y_exp = r + gamma*Q'( s1, pi'(s1))
        y_expected = r + GAMMA * q_next * (1. - d)
        # y_pred = Q( s0, a0)
        out = self.critic.forward(s0, a0)

        # Sum. loss of critic
        y_predicted = out[0]
        y_predicted = torch.squeeze(y_predicted)
        critic_TDLoss = torch.nn.SmoothL1Loss()(y_predicted, y_expected)

        loss_critic = critic_TDLoss
        if self.model_own:
            pred_r = out[1]
            pred_r = torch.squeeze(pred_r)
            critic_ModelLoss_own = torch.nn.L1Loss()(pred_r, r)
            loss_critic += critic_ModelLoss_own

        if self.model_adv:
            pred_r_adv = out[2]
            pred_r_adv = torch.squeeze(pred_r_adv)
            critic_ModelLoss_adv = torch.nn.L1Loss()(pred_r_adv, r_adv)
            loss_critic += critic_ModelLoss_adv

        # Update critic
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        # ---------------------- optimize actor ----------------------
        out = self.actor.forward(s0)
        pred_logits0 = out[0]
        pred_a0 = self.gumbel_softmax(pred_logits0)

        # Sum. loss of actor
        # Loss: max. Q
        Q = self.critic.forward(s0, pred_a0)[0]
        actor_maxQ = -1 * Q.mean()
        loss_actor = actor_maxQ

        if self.model_own:
            pred_s1 = out[1]
            pred_s1 = torch.squeeze(pred_s1)
            actor_ModelLoss_own = torch.nn.L1Loss()(pred_s1, s1)
            loss_actor += actor_ModelLoss_own

        if self.model_adv:
            pred_a0_adv = out[2]
            pred_a0_adv = torch.squeeze(pred_a0_adv)
            actor_ModelLoss_adv = torch.nn.CrossEntropyLoss()(pred_a0_adv.contiguous().view(-1, self.nb_actions),
                                                              torch.argmax(a0_adv.contiguous().view(-1, self.nb_actions), dim=-1))
            loss_actor += actor_ModelLoss_adv

        # Loss: regularization
        l2_reg = torch.cuda.FloatTensor(1)
        for W in self.actor.parameters():
            l2_reg = l2_reg + W.norm(2)
        loss_actor += torch.squeeze(l2_reg) * 1e-3

        # Update actor
        # run random noise to exploration
        self.actor.train()
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

        # Update target env
        self.soft_update(self.target_actor, self.actor, arglist.tau)
        self.soft_update(self.target_critic, self.critic, arglist.tau)

        return loss_actor, loss_critic

    def save_models(self, fname):
        """
        saves the target actor and critic models
        :param episode_count: the count of episodes iterated
        :return:
        """
        torch.save(self.target_actor.state_dict(), './Models/' + str(fname) + '_actor.pt')
        torch.save(self.target_critic.state_dict(), './Models/' + str(fname) + '_critic.pt')
        print('Models saved successfully')

    def load_models(self, fname):
        """
        loads the target actor and critic models, and copies them onto actor and critic models
        :param episode: the count of episodes iterated (used to find the file name)
        :return:
        """
        self.actor.load_state_dict(torch.load('./Models/' + str(fname) + '_actor.pt'))
        self.critic.load_state_dict(torch.load('./Models/' + str(fname) + '_critic.pt'))
        self.hard_update(self.target_actor, self.actor)
        self.hard_update(self.target_critic, self.critic)
        print('Models loaded succesfully')

    def save_training_checkpoint(self, state, is_best, episode_count):
        """
        Saves the models, with all training parameters intact
        :param state:
        :param is_best:
        :param filename:
        :return:
        """
        filename = str(episode_count) + 'checkpoint.path.rar'
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, 'model_best.pth.tar')