import numpy as np
import torch
import time
import pickle
from PIL import Image
import matplotlib.pyplot as plt

from rls import arglist
from rls.replay_buffer import ReplayBuffer
from rls.agent.multiagent.model_ddpg_competitive_mgym import Trainer

import gym
import mgym
import os
'''
NOT WORKING
from mgym.envs.snake_env import GRID_HEIGHT, GRID_WIDTH, GRID_SIZE
GRID_HEIGHT = 32
GRID_WIDTH = 32
GRID_SIZE = 16
'''
N = 2


def onehot2D(s):
    s = (np.arange(4) == s[..., None] - 1).astype('float32')
    s = np.transpose(s, [2, 0, 1])
    return s


def run(env, actor_own, critic_own, actor_adv, critic_adv, dir_model=None,
        own_model_own=False, own_model_adv=False, adv_model_own=False, adv_model_adv=False,
        flag_train=True, scenario_name=None, action_type='Discrete', cnt=0):
    """
    function of learning agent
    """
    if flag_train:
        exploration_mode = 'train'
    else:
        exploration_mode = 'test'

    torch.set_default_tensor_type('torch.FloatTensor')
    print('observation shape: ', env.observation_space)
    print('action shape: ', env.action_space)

    # <create actor-critic networks>
    memory_own = ReplayBuffer(size=1e+6)
    memory_adv = ReplayBuffer(size=1e+6)
    learner_own = Trainer(actor_own, critic_own, memory_own, memory_adv,
                          model_own=own_model_own, model_adv=own_model_adv, action_type=action_type)
    learner_adv = Trainer(actor_adv, critic_adv, memory_adv, memory_own,
                          model_own=adv_model_own, model_adv=adv_model_adv, action_type=action_type)

    def gumbel_softmax(self, x, hard=True):
        y = torch.nn.functional.gumbel_softmax(x, hard=hard)
        return y

    learner_own.gumbel_softmax = gumbel_softmax.__get__(learner_own, Trainer)
    learner_adv.gumbel_softmax = gumbel_softmax.__get__(learner_own, Trainer)

    # own
    if (own_model_own == False) & (own_model_adv == False):
        type_own = 'no_model'
    elif (own_model_own == True) & (own_model_adv == False):
        type_own = 'model_own'
    elif (own_model_own == True) & (own_model_adv == True):
        type_own = 'model_adv'
    else:
        raise NotImplementedError
    
    # opponent
    if (adv_model_own == False) & (adv_model_adv == False):
        type_adv = 'no_model'
    elif (adv_model_own == True) & (adv_model_adv == False):
        type_adv = 'model_own'
    elif (adv_model_own == True) & (adv_model_adv == True):
        type_adv = 'model_adv'
    else:
        raise NotImplementedError
    
    dir_model = '/mgym/' + type_own + ' vs ' + type_adv

    if not flag_train:
        learner_own.load_models('/mgym/' + type_own + '/' + scenario_name + 'own_fin_' + str(cnt))
        learner_adv.load_models('/mgym/' + type_adv + '/' + scenario_name + 'adv_fin_' + str(cnt))
    else:
        if not os.path.exists('Models/' + dir_model):
            os.makedirs('Models/' + dir_model)

    episode_rewards_own = [0.0]  # sum of rewards for our agents
    episode_rewards_adv = [0.0]  # sum of rewards for adversary agents
    agent_rewards = [[0.0] for _ in range(N)]  # individual agent reward
    final_ep_rewards_own = []  # sum of rewards for training curve
    final_ep_rewards_adv = []  # sum of rewards for training curve
    final_ep_ag_rewards = []  # agent rewards for training curve
    agent_info = [[[]]]  # placeholder for benchmarking info
    env.reset(N)
    # obs_n = np.expand_dims(onehot2D(env.grid), 0)
    obs_n = onehot2D(env.grid)

    episode_step = 0
    train_step = 0
    t_start = time.time()

    print('Starting iterations...')
    while True:
        # get action
        action_n_own = learner_own.get_exploration_action(obs_n, mode=exploration_mode)
        action_n_adv = learner_adv.get_exploration_action(obs_n, mode=exploration_mode)

        # environment step
        action_n_env = (action_n_own.argmax(), action_n_adv.argmax())
        new_obs_n, rew_n, done_n, info_n = env.step(action_n_env)
        # new_obs_n = np.expand_dims(onehot2D(new_obs_n), 0)
        new_obs_n = onehot2D(new_obs_n)

        # make shared reward
        rew_own = rew_n[0]
        rew_adv = rew_n[1]

        episode_step += 1
        done = np.all(done_n)
        terminal = done or (episode_step >= arglist.max_episode_len)
        # collect experience
        learner_own.memory_own.add(obs_n, action_n_own, rew_own, new_obs_n, float(done))
        learner_adv.memory_own.add(obs_n, action_n_adv, rew_adv, new_obs_n, float(done))
        obs_n = new_obs_n

        for i, rew in enumerate(rew_n):
            if i == 0:
                episode_rewards_own[-1] += rew
            else:
                episode_rewards_adv[-1] += rew
            agent_rewards[i][-1] += rew

        if terminal:
            env.reset(N)
            # obs_n = np.expand_dims(onehot2D(env.grid), 0)
            obs_n = onehot2D(env.grid)
            episode_step = 0
            episode_rewards_own.append(0)
            episode_rewards_adv.append(0)
            for a in agent_rewards:
                a.append(0)
            agent_info.append([[]])

        # increment global step counter
        train_step += 1

        # for displaying learned policies
        if arglist.display:
            time.sleep(0.01)
            plt.imshow(env.grid)
            plt.show()
            # frame = env.render(mode='rgb_array', close=False)
            # Image.fromarray(frame)
            # continue

        # update all trainers, if not in display or benchmark mode
        # <learning agent>
        do_learn = (train_step > arglist.warmup_steps) and (
                train_step % arglist.update_rate == 0) and arglist.is_training
        if flag_train and do_learn:
            loss_own = learner_own.optimize()
            loss_adv = learner_adv.optimize()

        if flag_train:
            # save model, display training output
            if terminal and (len(episode_rewards_own) % arglist.save_rate == 0):
                # print statement depends on whether or not there are adversaries
                print("steps: {}, episodes: {}, reward (our): {}, reward (adv): {}, time: {}".format(
                    train_step, len(episode_rewards_own),
                    round(np.mean(episode_rewards_own[-arglist.save_rate:]), 3),
                    round(np.mean(episode_rewards_adv[-arglist.save_rate:]), 3),
                    round(time.time() - t_start, 3)))
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards_own.append(np.mean(episode_rewards_own[-arglist.save_rate:]))
                final_ep_rewards_adv.append(np.mean(episode_rewards_adv[-arglist.save_rate:]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

            # saves final episode reward for plotting training curve later
            if len(episode_rewards_own) > arglist.num_episodes:
                hist = {'reward_episodes_own': episode_rewards_own,
                        'reward_episodes_adv': episode_rewards_adv,
                        'reward_episodes_by_agents': agent_rewards}
                file_name = 'Models/' + dir_model + '/history_' + scenario_name + '_' + str(cnt) + '.pkl'
                with open(file_name, 'wb') as fp:
                    pickle.dump(hist, fp)
                print('...Finished total of {} episodes.'.format(len(episode_rewards_own)))
                # save model
                learner_own.save_models(dir_model + '/' + scenario_name + 'own_fin_' + str(cnt))
                learner_adv.save_models(dir_model + '/' + scenario_name + 'adv_fin_' + str(cnt))
                break
        else:
            # save model, display testing output
            if terminal and (len(episode_rewards_own) % 10 == 0):
                # print statement depends on whether or not there are adversaries
                print("steps: {}, episodes: {}, reward (our): {}, reward (adv): {}, time: {}".format(
                    train_step, len(episode_rewards_own),
                    round(np.mean(episode_rewards_own[-10:]), 3),
                    round(np.mean(episode_rewards_adv[-10:]), 3),
                    round(time.time() - t_start, 3)))

                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards_own.append(np.mean(episode_rewards_own[-10:]))
                final_ep_rewards_adv.append(np.mean(episode_rewards_adv[-10:]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-10:]))

            # saves final episode reward for plotting training curve later
            if len(episode_rewards_own) > arglist.num_episodes:
                hist = {'reward_episodes_own': episode_rewards_own,
                        'reward_episodes_adv': episode_rewards_adv,
                        'reward_episodes_by_agents': agent_rewards}
                file_name = 'Models/test_results/' + dir_model + '/test_rewards_' + scenario_name + '_' + str(cnt) + '.pkl'
                with open(file_name, 'wb') as fp:
                    pickle.dump(hist, fp)
                print('...Finished total of {} episodes.'.format(len(episode_rewards_own)))
                env.close()
                break


if __name__ == '__main__':
    import torch
    import numpy as np
    from rls import arglist
    import matplotlib.pyplot as plt
    import gym
    import mgym

    env = gym.make('Snake-v0')
    env.reset(2)
    s0 = env.grid

    print(env.snakes[0].id)
    print(env.snakes[1].id)

    plt.imshow(s0)
    plt.show()

    while True:
        a = env.action_space.sample()
        s1, r, done, info = env.step(a)
        # print('action:', a)
        print('reward:', r)
        plt.imshow(env.grid)
        plt.show()

        if done:
            print(done)
            break


