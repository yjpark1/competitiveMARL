# reference: https://github.com/vy007vikas/PyTorch-ActorCriticRL/blob/master/train.py
from __future__ import division
import torch
import torch.nn.functional as F
import numpy as np
import shutil
from rls import arglist
import copy
from rls.utils import to_categorical

arglist.batch_size = 128
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
        obs = np.expand_dims(obs, axis=0)
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

    def to_onehot(self, a1):
        a1 = to_categorical(a1, num_classes=self.nb_actions)
        a1 = a1.astype('float32')
        a1 = torch.from_numpy(a1)
        return a1

    def get_exploitation_action(self, state):
        """
        gets the action from target actor added with exploration noise
        :param state: state (Numpy array)
        :return: sampled action (Numpy array)
        """
        # state = torch.from_numpy(state).to(self.device)
        action, _ = self.target_actor.forward(state)
        action = action.detach()
        # action = action.data.numpy().to(self.device)
        return action

    def get_exploration_action(self, state):
        """
        gets the action from actor added with exploration noise
        :param state: state (Numpy array)
        :return: sampled action (Numpy array)
        """
        # state = np.expand_dims(state, axis=0)
        state = state.to(self.device)
        action, _ = self.actor.forward(state)
        action = action.detach()
        new_action = action.data.cpu().numpy()  # + (self.noise.sample() * self.action_lim)
        return new_action

    def optimize(self):
        """
        Samples a random batch from replay memory and performs optimization
        :return:
        """
        s1, a1, r1, s2, d = self.memory.sample(arglist.batch_size)

        s1 = s1.to(self.device)
        a1 = a1.to(self.device)
        r1 = r1.to(self.device)
        s2 = s2.to(self.device)
        d = d.to(self.device)

        # ---------------------- optimize critic ----------------------
        # Use target actor exploitation policy here for loss evaluation
        a2, _ = self.target_actor.forward(s2)
        a2 = a2.detach()
        q_next, _ = self.target_critic.forward(s2, a2)
        q_next = q_next.detach()
        q_next = torch.squeeze(q_next)
        # y_exp = r + gamma*Q'( s2, pi'(s2))
        y_expected = r1 + GAMMA * q_next * (1. - d)
        # y_pred = Q( s1, a1)
        y_predicted, pred_r1 = self.critic.forward(s1, a1)
        y_predicted = torch.squeeze(y_predicted)
        pred_r1 = torch.squeeze(pred_r1)

        # compute critic loss, and update the critic
        loss_critic = F.smooth_l1_loss(y_predicted, y_expected)
        loss_critic += F.smooth_l1_loss(pred_r1, r1)

        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        # ---------------------- optimize actor ----------------------
        pred_a1, pred_s2 = self.actor.forward(s1)
        entropy = torch.sum(pred_a1 * torch.log(pred_a1), dim=-1).mean()
        l2_reg = torch.cuda.FloatTensor(1)
        for W in self.actor.parameters():
            l2_reg = l2_reg + W.norm(2)

        Q, _ = self.critic.forward(s1, pred_a1)
        loss_actor = -1 * torch.sum(Q)
        loss_actor += entropy * 0.05
        loss_actor += torch.squeeze(l2_reg) * 0.001
        loss_actor += F.smooth_l1_loss(pred_s2, s2)

        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

        self.soft_update(self.target_actor, self.actor, arglist.tau)
        self.soft_update(self.target_critic, self.critic, arglist.tau)

        return loss_actor, loss_critic

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