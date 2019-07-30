import torch
import numpy as np
from experiments.scenarios import make_env
from rls import arglist

# proposed
from rls.model.ac_networks_competitive_new import ActorNetwork, CriticNetwork
from experiments.run_competitive import run

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

scenarios = ['simple_world_comm_add']

TEST_ONLY = False
if TEST_ONLY:
    arglist.is_training = False

eval_list = ['model_adv']

for type_own, type_adv in zip(eval_list, eval_list):
    # own
    if type_own == 'no_model':
        own_model_own = False
        own_model_adv = False
    elif type_own == 'model_own':
        own_model_own = True
        own_model_adv = False
    elif type_own == 'model_adv':
        own_model_own = True
        own_model_adv = True

    # opponent
    if type_adv == 'no_model':
        adv_model_own = False
        adv_model_adv = False
    elif type_adv == 'model_own':
        adv_model_own = True
        adv_model_adv = False
    elif type_adv == 'model_adv':
        adv_model_own = True
        adv_model_adv = True

    for scenario_name in scenarios:
        arglist.actor_learning_rate = 1e-2
        arglist.critic_learning_rate = 1e-2

        for cnt in range(3):
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
            dim_action_own = env.action_space[-1].n
            dim_action_adv = env.action_space[0].n
            action_type = 'Discrete'

            # num_adv & adv action dims
            num_own = len([x for x in env.agents if not x.adversary])
            num_adv = len([x for x in env.agents if x.adversary])
            # own
            actor_own = ActorNetwork(input_dim=dim_obs_own, out_dim=dim_action_own,
                                     model_own=own_model_own, model_adv=own_model_adv, num_adv=num_adv, adv_out_dim=dim_action_adv)
            critic_own = CriticNetwork(input_dim=dim_obs_own + dim_action_own, out_dim=1,
                                       model_own=own_model_own, model_adv=own_model_adv)
            # opponent
            actor_adv = ActorNetwork(input_dim=dim_obs_adv, out_dim=dim_action_adv,
                                     model_own=adv_model_own, model_adv=adv_model_adv, num_adv=num_own, adv_out_dim=dim_action_own)
            critic_adv = CriticNetwork(input_dim=dim_obs_adv + dim_action_adv, out_dim=1,
                                       model_own=adv_model_own, model_adv=adv_model_adv)

            if TEST_ONLY:
                arglist.num_episodes = 100
                flag_train = False
            else:
                arglist.num_episodes = 40000
                flag_train = True

            run(env, actor_own, critic_own, actor_adv, critic_adv,
                own_model_own=own_model_own, own_model_adv=own_model_adv,
                adv_model_own=adv_model_own, adv_model_adv=adv_model_adv,
                flag_train=flag_train, scenario_name=scenario_name,
                action_type='Discrete', cnt=cnt)
