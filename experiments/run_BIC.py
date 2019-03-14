import numpy as np
import torch
import time
import pickle
from copy import deepcopy

from rls import arglist
from rls.replay_buffer import ReplayBuffer


def run(env, actor, critic, Trainer, scenario_name=None, action_type='Discrete', cnt=0):
    """function of learning agent
    """
    torch.set_default_tensor_type('torch.FloatTensor')
    print('observation shape: ', env.observation_space)
    print('action shape: ', env.action_space)

    # <create actor-critic networks>
    memory = ReplayBuffer(size=1e+6)
    learner = Trainer(actor, critic, memory, action_type=action_type)

    episode_rewards = [0.0]  # sum of rewards for all agents
    agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
    final_ep_rewards = []  # sum of rewards for training curve
    final_ep_ag_rewards = []  # agent rewards for training curve
    agent_info = [[[]]]  # placeholder for benchmarking info
    obs_n = env.reset()
    episode_step = 0
    train_step = 0
    t_start = time.time()

    print('Starting iterations...')
    while True:
        # get action
        if action_type == 'Discrete':
            action_n = learner.get_exploration_action(obs_n)[0]
            action_n_env = [np.array(x) for x in action_n.tolist()]
        elif action_type == 'MultiDiscrete':
            action_n = learner.get_exploration_action(obs_n)
            action_n_env = [np.concatenate([x, y], axis=-1) for x, y in zip(action_n[0][0], action_n[1][0])]

        # environment step
        new_obs_n, rew_n, done_n, info_n = env.step(action_n_env)

        episode_step += 1
        done_n = [float(d) for d in done_n]
        done = all(done_n)
        terminal = (episode_step >= arglist.max_episode_len)
        # collect experience
        learner.memory.add(obs_n, action_n_env, rew_n, new_obs_n, done_n)
        obs_n = new_obs_n

        for i, rew in enumerate(rew_n):
            episode_rewards[-1] += rew
            agent_rewards[i][-1] += rew

        if done or terminal:
            obs_n = env.reset()
            episode_step = 0
            episode_rewards.append(0)
            for a in agent_rewards:
                a.append(0)
            agent_info.append([[]])

        # increment global step counter
        train_step += 1

        # for displaying learned policies
        if arglist.display:
            time.sleep(0.1)
            env.render()
            continue

        # update all trainers, if not in display or benchmark mode
        # <learning agent>
        do_learn = (train_step > arglist.warmup_steps) and (
                train_step % arglist.update_rate == 0) and arglist.is_training
        if do_learn:
            loss = learner.optimize()

        # save model, display training output
        if terminal and (len(episode_rewards) % arglist.save_rate == 0):
            # print statement depends on whether or not there are adversaries
            print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                round(time.time() - t_start, 3)))
            t_start = time.time()
            # Keep track of final episode reward
            final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
            for rew in agent_rewards:
                final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

        # saves final episode reward for plotting training curve later
        if len(episode_rewards) > arglist.num_episodes:
            hist = {'reward_episodes': episode_rewards, 'reward_episodes_by_agents': agent_rewards}
            file_name = 'Models/history_' + scenario_name + '_' + str(cnt) + '.pkl'
            with open(file_name, 'wb') as fp:
                pickle.dump(hist, fp)
            print('...Finished total of {} episodes.'.format(len(episode_rewards)))
            learner.save_models(scenario_name + '_fin_' + str(cnt))  # save model
            break


def run_test(env, actor, critic, Trainer, scenario_name=None,
             action_type='Discrete', cnt=0):
    """function of learning agent
    """
    torch.set_default_tensor_type('torch.FloatTensor')
    print('observation shape: ', env.observation_space)
    print('action shape: ', env.action_space)

    # <create actor-critic networks>
    memory = ReplayBuffer(size=1e+6)
    learner = Trainer(actor, critic, memory, action_type=action_type)
    learner.load_models(arglist.appx + scenario_name + '_fin_' + str(cnt))

    episode_rewards = [0.0]  # sum of rewards for all agents
    agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
    final_ep_rewards = []  # sum of rewards for training curve
    final_ep_ag_rewards = []  # agent rewards for training curve
    agent_info = [[[]]]  # placeholder for benchmarking info
    obs_n = env.reset()
    episode_step = 0
    train_step = 0
    t_start = time.time()

    print('Starting iterations...')
    while True:
        # get action
        if action_type == 'Discrete':
            action_n = learner.get_exploration_action(obs_n)[0]
            action_n_env = [np.array(x) for x in action_n.tolist()]
        elif action_type == 'MultiDiscrete':
            action_n = learner.get_exploration_action(obs_n)
            action_n_env = [np.concatenate([x, y], axis=-1) for x, y in zip(action_n[0][0], action_n[1][0])]

        # environment step
        new_obs_n, rew_n, done_n, info_n = env.step(action_n_env)

        episode_step += 1
        done_n = [float(d) for d in done_n]
        done = all(done_n)
        terminal = (episode_step >= arglist.max_episode_len)
        # collect experience
        learner.memory.add(obs_n, action_n_env, rew_n, new_obs_n, done_n)
        obs_n = new_obs_n

        for i, rew in enumerate(rew_n):
            episode_rewards[-1] += rew
            agent_rewards[i][-1] += rew

        if done or terminal:
            obs_n = env.reset()
            episode_step = 0
            episode_rewards.append(0)
            for a in agent_rewards:
                a.append(0)
            agent_info.append([[]])

        # increment global step counter
        train_step += 1

        # for displaying learned policies
        if arglist.display:
            time.sleep(0.1)
            env.render()
            continue

        # save model, display training output
        if terminal and (len(episode_rewards) % 10 == 0):
            # print statement depends on whether or not there are adversaries
            print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                train_step, len(episode_rewards), np.mean(episode_rewards[-10:]),
                round(time.time() - t_start, 3)))
            t_start = time.time()
            # Keep track of final episode reward
            final_ep_rewards.append(np.mean(episode_rewards[-10:]))
            for rew in agent_rewards:
                final_ep_ag_rewards.append(np.mean(rew[-10:]))

        # saves final episode reward for plotting training curve later
        if len(episode_rewards) > arglist.num_episodes:
            hist = {'reward_episodes': episode_rewards,
                    'reward_episodes_by_agents': agent_rewards,
                    'memory': memory}
            file_name = 'Models/test_history_' + scenario_name + '_' + str(cnt) + '.pkl'
            with open(file_name, 'wb') as fp:
                pickle.dump(hist, fp)
            print('...Finished total of {} episodes.'.format(len(episode_rewards)))
            break


if __name__ == '__main__':
    from rls.model.ac_network_multi_gumbel_BIC import ActorNetwork, CriticNetwork
    from rls.agent.multiagent.BIC_gumbel_fix import Trainer
    from experiments.scenarios import make_env
    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    scenario_name = 'simple_spread'

    env = make_env(scenario_name, benchmark=False, discrete_action=True)
    cnt = 0
    seed = cnt + 12345678
    env.seed(seed)
    torch.cuda.empty_cache()
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    dim_obs = env.observation_space[0].shape[0]
    dim_action = env.action_space[0].n

    actor = ActorNetwork(input_dim=dim_obs, out_dim=dim_action)
    critic = CriticNetwork(input_dim=dim_obs + dim_action, out_dim=1)
    run(env, actor, critic, Trainer, scenario_name, cnt=cnt)



