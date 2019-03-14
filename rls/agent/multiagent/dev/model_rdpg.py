# reference: https://github.com/vy007vikas/PyTorch-ActorCriticRL/blob/master/train.py
from __future__ import division
import torch
import torch.nn.functional as F
import numpy as np
import shutil
from rls import arglist
import copy
from rls.utils import to_categorical

GAMMA = 0.95
TAU = 0.001


class Trainer:
    def __init__(self, actor, critic, memory):
        """
        DDPG for categorical action
        """
        self.device = torch.device('cuda:0')

        self.iter = 0
        self.actor = actor.to(self.device)
        self.target_actor = copy.deepcopy(actor).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), arglist.learning_rate)

        self.critic = critic.to(self.device)
        self.target_critic = copy.deepcopy(critic).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), arglist.learning_rate)

        self.memory = memory
        self.nb_actions = 5

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
        obs = np.array(obs, dtype='float32')
        obs = np.expand_dims(obs, axis=0)  # add dim. batch
        obs = np.expand_dims(obs, axis=0)  # add dim. seq_len
        obs = torch.from_numpy(obs)
        return obs

    def process_action(self, actions):
        actions = np.argmax(actions, axis=-1)
        actions = actions.reshape(-1)
        return actions

    def process_reward(self, rewards):
        rewards = np.array(rewards, dtype='float32')
        rewards = torch.from_numpy(rewards)
        return rewards

    def process_done(self, done):
        done = np.array(done, dtype='float32')
        done = torch.from_numpy(done)
        return done

    def process_batch(self, experiences):
        s0 = []
        a0 = []
        r = []
        s1 = []
        d = []
        for e in experiences:
            s0.append(torch.cat([step.state0 for step in e], dim=0))
            a0.append(torch.stack([step.action for step in e], dim=0))
            r.append(torch.stack([step.reward for step in e], dim=0))
            s1.append(torch.cat([step.state1 for step in e], dim=0))
            d.append(torch.stack([step.terminal1 for step in e], dim=0))

        s0 = torch.cat(s0, dim=1)
        a0 = torch.stack(a0, dim=1)
        r = torch.stack(r, dim=1)
        s1 = torch.cat(s1, dim=1)
        d = torch.stack(d, dim=1)

        return s0, a0, r, s1, d

    def to_onehot(self, a1):
        a1 = to_categorical(a1, num_classes=self.nb_actions)
        a1 = a1.astype('float32')
        a1 = torch.from_numpy(a1)
        return a1

    def get_exploration_action(self, state):
        """
        gets the action from actor added with exploration noise
        :param state: state (Numpy array)
        :return: sampled action (Numpy array)
        """
        # state = np.expand_dims(state, axis=0)
        state = state.to(self.device)
        action, _, hcTime = self.actor.forward(state)
        self.actor.hState = (hcTime[0].detach(), hcTime[1].detach())  # memorize hidden state
        action = action.detach()
        new_action = action.data.cpu().numpy()  # + (self.noise.sample() * self.action_lim)
        return new_action

    def optimize(self):
        """
        Samples a random batch from replay memory and performs optimization
        :return:
        """
        experiences = self.memory.sample(arglist.batch_size)
        s0, a0, r, s1, d = self.process_batch(experiences)

        s0 = s0.to(self.device)
        a0 = a0.to(self.device)
        r = r.to(self.device)
        s1 = s1.to(self.device)
        d = d.to(self.device)

        # ---------------------- optimize critic ----------------------
        # Use target actor exploitation policy here for loss evaluation
        a1, _, _ = self.target_actor.forward(s1)
        a1 = a1.detach()
        q_next, _, _ = self.target_critic.forward(s1, a1)
        q_next = q_next.detach()
        q_next = torch.squeeze(q_next)
        # Loss: TD error
        # y_exp = r + gamma*Q'( s1, pi'(s1))
        y_expected = r + GAMMA * q_next * (1. - d)
        # y_pred = Q( s0, a0)
        y_predicted, pred_r, _ = self.critic.forward(s0, a0)
        y_predicted = torch.squeeze(y_predicted)
        pred_r = torch.squeeze(pred_r)

        # Sum. Loss
        critic_TDLoss = torch.nn.SmoothL1Loss()(y_predicted, y_expected)
        critic_ModelLoss = torch.nn.L1Loss()(pred_r, r)
        loss_critic = critic_TDLoss
        loss_critic += critic_ModelLoss

        # Update critic
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        # ---------------------- optimize actor ----------------------
        pred_a0, pred_s1, _ = self.actor.forward(s0)

        # Loss: entropy for exploration
        entropy = torch.sum(pred_a0 * torch.log(pred_a0), dim=-1).mean()
        # Loss: regularization
        l2_reg = torch.cuda.FloatTensor(1)
        for W in self.actor.parameters():
            l2_reg = l2_reg + W.norm(2)
        # Loss: max. Q
        Q, _, _ = self.critic.forward(s0, pred_a0)
        actor_maxQ = -1 * Q.mean()

        # Loss: model loss
        actor_ModelLoss = torch.nn.L1Loss()(pred_s1, s1)

        # Sum. Loss
        loss_actor = actor_maxQ
        loss_actor += entropy * 0.05
        loss_actor += torch.squeeze(l2_reg) * 0.001
        loss_actor += actor_ModelLoss

        # Update actor
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

        # Update target model
        self.soft_update(self.target_actor, self.actor, arglist.tau)
        self.soft_update(self.target_critic, self.critic, arglist.tau)

        return loss_actor, loss_critic, critic_TDLoss, critic_ModelLoss, actor_maxQ, actor_ModelLoss

    def save_models(self, episode_count):
        """
        saves the target actor and critic models
        :param episode_count: the count of episodes iterated
        :return:
        """
        torch.save(self.target_actor.state_dict(), './Models/' + str(episode_count) + '_actor.pt')
        torch.save(self.target_critic.state_dict(), './Models/' + str(episode_count) + '_critic.pt')
        print('Models saved successfully')

    def load_models(self, episode):
        """
        loads the target actor and critic models, and copies them onto actor and critic models
        :param episode: the count of episodes iterated (used to find the file name)
        :return:
        """
        self.actor.load_state_dict(torch.load('./Models/' + str(episode) + '_actor.pt'))
        self.critic.load_state_dict(torch.load('./Models/' + str(episode) + '_critic.pt'))
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