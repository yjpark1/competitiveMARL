# reference: https://github.com/vy007vikas/PyTorch-ActorCriticRL/blob/master/train.py
from __future__ import division
import torch
import torch.nn.functional as F
import numpy as np
import shutil
from rls import arglist
import copy
from rls.utils import to_categorical

arglist.batch_size = 256
GAMMA = 0.99
TAU = 0.001


class Trainer:
    def __init__(self, actor, critic, memory):
        """
        DDPG for categorical action
        """
        self.device = torch.device('cuda:0')

        self.iter = 0
        self.eps = 1

        self.actor = actor.to(self.device)
        self.target_actor = copy.deepcopy(actor).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), arglist.actor_learning_rate)

        self.critic = critic.to(self.device)
        self.target_critic = copy.deepcopy(critic).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), arglist.critic_learning_rate)

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
        # actions = np.argmax(actions, axis=-1)
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

    def get_exploration_action(self, state):
        """
        gets the action from actor added with exploration noise
        :param state: state (Numpy array)
        :return: sampled action (Numpy array)
        """
        # state = np.expand_dims(state, axis=0)
        state = state.to(self.device)
        policy = self.actor.forward(state).detach()
        action = policy.data.cpu().numpy()  # + (self.noise.sample() * self.action_lim)
        action = np.squeeze(action)
        # stochastic exploration
        new_action = np.random.choice(self.nb_actions, p=action, replace=False)
        # new_action = np.argmax(action)

        # e-greedy exploration
        # if np.random.uniform() < self.eps:
        #     self.eps += -0.0005
        #     new_action = np.random.choice([0, 1], p=[0.5, 0.5])
        # else:
        #     new_action = np.argmax(new_action[0], axis=-1)

        return new_action, policy[0]

    def optimize(self):
        """
        Samples a random batch from replay memory and performs optimization
        :return:
        """
        s0, a0, r, s1, d = self.memory.sample(arglist.batch_size)

        s0 = s0.to(self.device)
        a0 = a0.to(self.device)
        r = r.to(self.device)
        s1 = s1.to(self.device)
        d = d.to(self.device)

        # ---------------------- optimize critic ----------------------
        # Use target actor exploitation policy here for loss evaluation
        a1 = self.target_actor.forward(s1)
        a1 = a1.detach()

        # target critic
        q_next = self.target_critic.forward(s1, a1)
        q_next = q_next.detach()
        q_next = torch.squeeze(q_next)

        # Loss: TD error
        # y_exp = r + gamma*Q'( s1, pi'(s1))
        y_expected = r + GAMMA * q_next * (1. - d)
        # y_pred = Q( s0, a0)
        y_predicted = self.critic.forward(s0, a0)
        y_predicted = torch.squeeze(y_predicted)

        # Sum. Loss
        critic_TDLoss = torch.nn.SmoothL1Loss()(y_predicted, y_expected)
        loss_critic = critic_TDLoss

        # Update critic
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.)
        self.critic_optimizer.step()

        # ---------------------- optimize actor ----------------------
        pred_a0 = self.actor.forward(s0)

        # Loss: entropy for exploration
        entropy = torch.sum(pred_a0 * torch.log(pred_a0 + 1e-10), dim=-1)
        entropy = entropy.mean()

        # Loss: regularization
        l2_reg = torch.cuda.FloatTensor(1)
        for W in self.actor.parameters():
            l2_reg = l2_reg + W.norm(2)

        # Loss: max. Q
        Q = self.critic.forward(s0, pred_a0)
        actor_maxQ = - Q.mean()

        # Loss: env loss
        # Sum. Loss
        loss_actor = actor_maxQ
        loss_actor += entropy * 0.01  # <replace Gaussian noise>
        loss_actor += torch.squeeze(l2_reg) * 0.001

        # Update actor
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.)
        self.actor_optimizer.step()

        self.soft_update(self.target_actor, self.actor, arglist.tau)
        self.soft_update(self.target_critic, self.critic, arglist.tau)

        self.iter += 1

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