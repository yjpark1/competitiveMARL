import numpy as np
import torch
import time
import pickle
from copy import deepcopy

from rls import arglist
from rls.replay_buffer import ReplayBuffer
from rls.agent.multiagent.model_ddpg_competitive import Trainer


def split_obs_n(env, obs_n):
    return

def combine_obs_n(env, obs_n_own, obs_n_adv):
    return

def combine_obs_n(env, action_n_env_own, action_n_env_adv):
    return


def run(env, actor_own, critic_own, actor_adv, critic_adv, 
        own_model_own=False, own_model_adv=False, adv_model_own=False, adv_model_adv=False,
        flag_train=True, scenario_name=None, action_type='Discrete', cnt=0):
    """
    function of learning agent
    """
    torch.set_default_tensor_type('torch.FloatTensor')
    print('observation shape: ', env.observation_space)
    print('action shape: ', env.action_space)

    # <create actor-critic networks>
    memory_own = ReplayBuffer(size=1e+6)
    memory_adv = ReplayBuffer(size=1e+6)
    learner_own = Trainer(actor_own, critic_own, memory_own,
                          model_own=own_model_own, model_adv=own_model_adv, action_type=action_type)
    learner_adv = Trainer(actor_adv, critic_adv, memory_adv,
                          model_own=adv_model_own, model_adv=adv_model_adv, action_type=action_type)

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
        action_n_own = learner_own.get_exploration_action(obs_n)[0]
        action_n_env_own = [np.array(x) for x in action_n_own.tolist()]

        action_n_adv = learner_own.get_exploration_action(obs_n)[0]
        action_n_env_adv = [np.array(x) for x in action_n_adv.tolist()]

        # environment step
        new_obs_n, rew_n, done_n, info_n = env.step(action_n_env)
        # make shared reward
        rew_shared = np.sum(rew_n)

        episode_step += 1
        done = all(done_n)
        terminal = (episode_step >= arglist.max_episode_len)
        # collect experience
        learner_own.memory.add(obs_n, action_n_env, rew_shared, new_obs_n, float(done))
        learner_adv.memory.add(obs_n, action_n_env, rew_shared, new_obs_n, float(done))
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
        if flag_train and do_learn:
            loss_own = learner_own.optimize()
            loss_adv = learner_adv.optimize()

        if flag_train:
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
        else:
            # save model, display testing output
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
    from rls.model.ac_network_model_multi_gumbel import ActorNetwork, CriticNetwork
    from rls.agent.multiagent.model_ddpg_gumbel_fix import Trainer
    from experiments.scenarios import make_env
    import os

    arglist.actor_learning_rate = 1e-2
    arglist.critic_learning_rate = 1e-2

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    cnt = 11
    # scenario_name = 'simple_spread'
    scenario_name = 'simple_speaker_listener'
    env = make_env(scenario_name, benchmark=False, discrete_action=True)
    seed = cnt + 12345678
    env.seed(seed)
    torch.cuda.empty_cache()
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    dim_obs = env.observation_space[0].shape[0]
    if hasattr(env.action_space[0], 'high'):
        dim_action = env.action_space[0].high + 1
        dim_action = dim_action.tolist()
        action_type = 'MultiDiscrete'
    else:
        dim_action = env.action_space[0].n
        action_type = 'Discrete'

    actor = ActorNetwork(input_dim=dim_obs, out_dim=dim_action)
    critic = CriticNetwork(input_dim=dim_obs + sum(dim_action), out_dim=1)
    run(env, actor, critic, Trainer, scenario_name, action_type, cnt=cnt)



