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

N = 100
scenarios = ['simple_adversary', 'simple_crypto', 'simple_push',
             'simple_tag', 'simple_world_comm']

scenarios = ['simple_adversary', 'simple_crypto']

loc_dir = {
            'no_model vs no_model': 'Models/test_results/no_model vs no_model',
            'model_own vs no_model': 'Models/test_results/model_own vs no_model',
            'no_model vs model_own': 'Models/test_results/no_model vs model_own',
            'model_own vs model_own': 'Models/test_results/model_own vs model_own',
           }

loc_dir = {
            'no_model vs model_adv': 'Models/test_results/no_model vs model_adv',
            'model_adv vs no_model': 'Models/test_results/model_adv vs no_model',
            'model_own vs model_adv': 'Models/test_results/model_own vs model_adv',
            'model_adv vs model_own': 'Models/test_results/model_adv vs model_own',
            'model_adv vs model_adv': 'Models/test_results/model_adv vs model_adv',
           }


def readResult(root, scenario):
    # root = 'Models/model_own vs model_own'
    # scenario = scenarios[0]
    files = os.listdir(root)
    files = [x for x in files if 'test_rewards' in x]
    sc_files = [x for x in files if scenario in x]
    rewards_own = []
    rewards_adv = []
    for f in sc_files:
        path = os.path.join(root, f)
        with open(path, 'rb') as ff:
            reward = pickle.load(ff)
        r_own = reward['reward_episodes_own'][:N]
        r_adv = reward['reward_episodes_adv'][:N]
        '''
        scaler = MinMaxScaler()
        r_own = scaler.fit_transform(np.array(r_own).reshape(-1, 1))
        scaler = MinMaxScaler()
        r_adv = scaler.fit_transform(np.array(r_adv).reshape(-1, 1))
        '''
        rewards_own.append(r_own)
        rewards_adv.append(r_adv)

    rewards_own = np.array(rewards_own)
    rewards_adv = np.array(rewards_adv)

    rewards_own = np.squeeze(rewards_own)
    rewards_adv = np.squeeze(rewards_adv)

    return rewards_own, rewards_adv


def statistics(out):
    out_s = np.array([np.mean(out, axis=-1), np.std(out, axis=-1),
                      np.amin(out, axis=-1), np.amax(out, axis=-1), np.median(out, axis=-1)])
    return out_s


def wrt_results(sc_index):
    for case, root in loc_dir.items():
        r_own, r_adv = readResult(root, scenario=scenarios[sc_index])
        own = statistics(r_own)
        adv = statistics(r_adv)
        out_s = np.concatenate([own, adv], axis=0)
        np.savetxt(case + '_' + scenarios[sc_index] + '_statistic_add.csv', out_s, delimiter=',',
                   encoding='utf8')


for sc_index in range(2):
    wrt_results(sc_index)