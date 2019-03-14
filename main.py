import torch
import numpy as np
from experiments.scenarios import make_env
from rls import arglist

# proposed
from rls.model.ac_networks_competitive import ActorNetwork, CriticNetwork
from experiments.run_competitive import run

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

scenarios = ['simple_adversary', 'simple_crypto', 'simple_push',
             'simple_tag', 'simple_world_comm']

TEST_ONLY = False

for scenario_name in scenarios:
    arglist.actor_learning_rate = 1e-2
    arglist.critic_learning_rate = 1e-2

    for cnt in range(10):
        # scenario_name = 'simple_spread'
        env = make_env(scenario_name, discrete_action=True)
        seed = cnt + 12345678

        env.seed(seed)
        torch.cuda.empty_cache()
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        dim_obs_own = env.observation_space[-1].shape[0]
        dim_obs_adv = env.observation_space[0].shape[0]
        dim_action = env.action_space[0].n
        action_type = 'Discrete'
        # own
        actor_own = ActorNetwork(input_dim=dim_obs_own, out_dim=dim_action,
                                 model_own=False, model_adv=False)
        critic_own = CriticNetwork(input_dim=dim_obs_own + np.sum(dim_action), out_dim=1,
                                   model_own=False, model_adv=False)
        # opponent
        actor_adv = ActorNetwork(input_dim=dim_obs_adv, out_dim=dim_action,
                                 model_own=False, model_adv=False)
        critic_adv = CriticNetwork(input_dim=dim_obs_adv + np.sum(dim_action), out_dim=1,
                                   model_own=False, model_adv=False)

        if TEST_ONLY:
            arglist.num_episodes = 100
            flag_train = False
        else:
            arglist.num_episodes = 40000
            flag_train = True

        run(env, actor_own, critic_own, actor_adv, critic_adv,
            own_model_own=True, own_model_adv=True, adv_model_own=True, adv_model_adv=True,
            flag_train=flag_train, scenario_name=scenario_name,
            action_type='Discrete', cnt=0)
