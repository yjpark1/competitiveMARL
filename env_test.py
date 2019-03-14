import numpy as np
import multiagent.scenarios as scenarios
from multiagent.environment import MultiAgentEnv


def make_env(scenario_name, discrete_action=True):
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents

    simple_spread
    simple_reference
    simple_speaker_listener
    collect_treasure
    multi_speaker_listener

    '''
    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()

    # create world
    world = scenario.make_world()
    world.collaborative = False  # to get individual reward

    # create multiagent environment
    if hasattr(scenario, 'post_step'):
        post_step = scenario.post_step
    else:
        post_step = None

    env = MultiAgentEnv(world, reset_callback=scenario.reset_world,
                        reward_callback=scenario.reward,
                        observation_callback=scenario.observation,
                        post_step_callback=post_step,
                        discrete_action=discrete_action)
    env.force_discrete_action = True
    return env


scs = ['simple_adversary', 'simple_crypto', 'simple_push',
       'simple_tag', 'simple_world_comm']

sc = scs[4]
env = make_env(scenario_name=sc, discrete_action=True)
print('observation shape: ', env.observation_space)
print('action shape: ', env.action_space)

import mujoco_py
import gym
import time
env = gym.make('Hopper-v2')
print(env.observation_space)
print(env.action_space)
env.reset()
for _ in range(100):
    time.sleep(0.2)
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()