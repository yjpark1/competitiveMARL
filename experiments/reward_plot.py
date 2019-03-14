# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 19:55:33 2018

@author: yj-wn
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(palette="Set2")

loc_dir = {
            'MADDPG': 'result_maddpg_simple_spread_origin_60k',
            'BiCNet': 'bicnet',
            'Proposed': 'result_our_simple_spread_origin_60k',
            'Proposed+model': 'model_based_partial',
            'Proposed+model(4frame)': '4frame_our',
            'RDPG(64)': 'rdpg64',
            'RDPG(128)': 'rdpg128'
           }

result = {
          'MADDPG': 0,
          'BiCNet': 0,
          'Proposed': 0,
          'Proposed+model': 0,
          'Proposed+model(4frame)': 0,
          'RDPG(64)': 0,
          'RDPG(128)': 0
         }

N = 60000

def readResult(root):
    rewards = []
    files = os.listdir(root)
    for f in files:
        path = os.path.join(root, f)
        reward = np.load(path)
        reward = reward[:N]
        rewards.append(reward)

    return np.stack(rewards)


for mth, root in loc_dir.items():
    result[mth] = readResult(root)

##############################
##############################
##############################
# CI plot
#data = {'method': [],
#        'episode': [],
#        'seed': [],
#        'reward': [],}
#
#for k, re in result.items():
#    for seed, reward in enumerate(re):
#        episode = np.arange(N).tolist()
#        method = np.repeat(k, N).tolist()
#        seed = np.repeat(seed, N).tolist()
#        reward = reward.tolist()
#
#        data['method'] += method
#        data['seed'] += seed
#        data['episode'] += episode
#        data['reward'] += reward
#
#a = pd.DataFrame(data)
#
#plt.subplots(figsize=(10,7))
#sns.lineplot(x="episode", y="reward",
#             hue="method", data=a, n_boot=10)

##############################
##############################
##############################
# Smoothing CI plot
data_smt = {'Method': [],
            'Episode': [],
            'seed': [],
            'Reward': [],}

for k, re in result.items():
    for seed, reward in enumerate(re):
        reward = reward.tolist()
        reward = pd.DataFrame(reward).rolling(window=400).mean()
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
plt.subplots(figsize=(8,6))
sns.lineplot(x="Episode", y="Reward",
             hue="Method", data=a_smt, n_boot=10)

