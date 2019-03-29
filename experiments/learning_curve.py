# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 13:51:40 2019

@author: yj-wn
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import seaborn as sns
import pickle
sns.set(palette="Set2")


root = 'Models/model_own vs model_own'

N = 40000
scenarios = ['simple_adversary', 'simple_crypto', 'simple_push',
             'simple_tag', 'simple_world_comm']

loc_dir = {
            'w/o Model': 'Models/no_model vs no_model',
            'Model_own': 'Models/model_own vs model_own',
           }


def readResult(root, scenario):
    # root = 'Models/model_own vs model_own'
    # scenario = scenarios[0]
    files = os.listdir(root)
    files = [x for x in files if 'history' in x]
    sc_files = [x for x in files if scenario in x]
    rewards_own = []
    rewards_adv = []
    for f in sc_files:
        path = os.path.join(root, f)
        with open(path, 'rb') as ff:
            reward = pickle.load(ff)
        r_own = reward['reward_episodes_own'][:N]
        r_adv = reward['reward_episodes_adv'][:N]

        rewards_own.append(r_own)
        rewards_adv.append(r_adv)

    rewards_own = np.array(rewards_own)
    rewards_adv = np.array(rewards_adv)
    ###
    scaler = MinMaxScaler()
    scaler.fit(rewards_own.flatten().reshape(-1, 1))
    rewards_own = scaler.transform(rewards_own)

    scaler = MinMaxScaler()
    scaler.fit(rewards_adv.flatten().reshape(-1, 1))
    rewards_adv = scaler.transform(rewards_adv)
    ###
    rewards_own = np.squeeze(rewards_own)
    rewards_adv = np.squeeze(rewards_adv)

    return rewards_own, rewards_adv


def Plot(sc_index):
    result = {
              'w/o Model (own)': 0,
              'w/o Model (adv)': 0,
              'Model_own (own)': 0,
              'Model_own (adv)': 0,
             }
    for mth, root in loc_dir.items():
         r_own, r_adv = readResult(root, scenario=scenarios[sc_index])
         result[mth + ' (own)'] = r_own
         result[mth + ' (adv)'] = r_adv

    # Smoothing CI plot
    data_smt = {'Method': [],
                'Episode': [],
                'seed': [],
                'Reward': [],}

    for k, re in result.items():
        for seed, reward in enumerate(re):
            reward = reward.tolist()
            reward = pd.DataFrame(reward).rolling(window=50).mean()
            reward = reward.dropna().values
            reward = reward.flatten()

            N = len(reward)
            episode = np.arange(N).tolist()
            method = np.repeat(k, N).tolist()
            seed = np.repeat(seed, N).tolist()
            reward = reward.tolist()

            data_smt['Method'] += method
            data_smt['seed'] += seed
            data_smt['Episode'] += episode
            data_smt['Reward'] += reward

    a_smt = pd.DataFrame(data_smt)
    plt.subplots(figsize=(5,3.75))
    sns.lineplot(x="Episode", y="Reward",
                 hue="Method", data=a_smt, n_boot=10,
                 ci=95)
    plt.savefig(scenarios[sc_index] + '_normalize.png',
                dpi=250, bbox_inches='tight')
    plt.show()


#######
for sc_index in range(5):
    Plot(sc_index)