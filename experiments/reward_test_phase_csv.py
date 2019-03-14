# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 16:11:42 2019

@author: yj-wn
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(palette="Set2")

scenarios = ['simple_spread', 'simple_reference', 'simple_speaker_listener',
             'fullobs_collect_treasure', 'multi_speaker_listener']

index = 'env_origin'

loc_dir = {
            'MADDPG': 'Models/' + index + '/maddpg',
            'BiCNet': 'Models/' + index + '/BICnet',
            'MAAC': 'Models/' + index + '/MAAC',
            'MADR': 'Models/' + index + '/proposed+gumbel',
            'MADR+AML': 'Models/' + index + '/proposed+gumbel+model',
           }

N = 100

def readResult(root, scenario):
    # root = 'Models/env_partial/MAAC'
    # scenario = scenarios[0]
    files = os.listdir(root)
    files = [x for x in files if 'history' in x]
    files = [x for x in files if 'test' in x]
    sc_files = [x for x in files if scenario in x]
    rewards = []
    for f in sc_files:
        path = os.path.join(root, f)
        with open(path, 'rb') as ff:
            reward = pickle.load(ff)
        reward = reward['reward_episodes']
        reward = reward[:N]
        rewards.append(reward)
    rewards = np.stack(rewards)
    return rewards


def Plot(sc_index):
    result = {
              'MADDPG': 0,
              'BiCNet': 0,
              'MAAC': 0,
              'MADR': 0,
              'MADR+AML': 0,
             }

    for mth, root in loc_dir.items():
        result[mth] = readResult(root, scenario=scenarios[sc_index])

    out = []
    for mth, _ in result.items():
        out.append(result[mth])
    out = np.concatenate(out)
    # np.savetxt(scenarios[sc_index] + '_' + index + '.csv', out, delimiter=',')
    out_s = np.array([np.mean(out, axis=-1), np.std(out, axis=-1),
                      np.amin(out, axis=-1), np.amax(out, axis=-1), np.median(out, axis=-1)])
    np.savetxt(scenarios[sc_index] + '_' + index + '_statistic.csv', out_s, delimiter=',')


#######
for sc_index in range(5):
    Plot(sc_index)