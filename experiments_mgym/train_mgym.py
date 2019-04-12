import torch
import numpy as np
from rls import arglist

# proposed
from rls.model.ac_networks_competitive_cnn import Conv, ActorNetwork, CriticNetwork
from experiments_mgym.run_mgym import run

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import gym
import mgym

TEST_ONLY = False
if TEST_ONLY:
    arglist.is_training = False

arglist.actor_learning_rate = 1e-4
arglist.critic_learning_rate = 1e-4

# 'no_model',
eval_list = ['no_model', 'model_own', 'model_adv']
for type in eval_list:
    if type == 'no_model':
        own_model_own = False
        own_model_adv = False
        adv_model_own = False
        adv_model_adv = False
    elif type == 'model_own':
        own_model_own = True
        own_model_adv = False
        adv_model_own = True
        adv_model_adv = False
    elif type == 'model_adv':
        own_model_own = True
        own_model_adv = True
        adv_model_own = True
        adv_model_adv = True

    # define agents
    cnn_own = Conv()
    actor_own = ActorNetwork(conv=cnn_own, out_dim=4, model_own=own_model_own, model_adv=own_model_adv)
    critic_own = CriticNetwork(conv=cnn_own, out_dim=1, model_own=own_model_own, model_adv=own_model_adv)
    cnn_adv = Conv()
    actor_adv = ActorNetwork(conv=cnn_adv, out_dim=4, model_own=adv_model_own, model_adv=adv_model_adv)
    critic_adv = CriticNetwork(conv=cnn_adv, out_dim=1, model_own=adv_model_own, model_adv=adv_model_adv)

    # run
    if TEST_ONLY:
        arglist.num_episodes = 100
        flag_train = False
    else:
        arglist.num_episodes = 40000
        flag_train = True

    scenario_name = 'Snake-v0'

    for cnt in range(1):
        env = gym.make(scenario_name)

        seed = 12345678 + cnt
        torch.cuda.empty_cache()
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        run(env, actor_own, critic_own, actor_adv, critic_adv, dir_model=None,
            own_model_own=own_model_own, own_model_adv=own_model_adv,
            adv_model_own=adv_model_own, adv_model_adv=adv_model_adv,
            flag_train=flag_train, scenario_name=scenario_name,
            action_type='Discrete', cnt=cnt)
