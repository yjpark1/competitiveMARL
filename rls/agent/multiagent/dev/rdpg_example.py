# Inspired by: https://github.com/yanpanlau/DDPG-Keras-Torcs/blob/master/ddpg.py

from ReplayBuffer import ReplayBuffer
from Noise import OUNoise
from Critic import Critic
from Actor import Actor
import tensorflow as tf
import numpy as np
import logging

from imp import reload
reload(logging)

# Runs on Gym.
class RDPGAgent:
    def __init__(self, env, batchSize = 10, bufferSize = 100,
                 gamma = 0.98, actorLR = 1e-4, criticLR = 1e-3,
                 maxSteps = 200, targetUpdate = 1e-3, epsilon = 1,
                 decay = 0.99, rewardScale = 1e-3, logFile = 'run.log'):
        self.env = env
        self.gamma = gamma
        self.batchSize = batchSize
        self.bufferSize = bufferSize
        self.maxSteps = maxSteps + 1
        self.rewardScale = rewardScale
        self.epsilon = epsilon
        self.decay = decay

        # Useful helpers.
        self.actionDim = self.env.action_space.shape[0]
        self.stateDim = self.env.observation_space.shape[0]
        self.featureDim = self.actionDim + self.stateDim
        self.minAction = self.env.action_space.low
        self.maxAction = self.env.action_space.high

        # For scaling output action values.
        self.actionBiasZeroOne = self.minAction
        self.actionScaleZeroOne = self.maxAction - self.minAction
        self.actionBiasTanH = (self.maxAction + self.minAction) / 2.0
        self.actionScaleTanH = self.maxAction - self.actionBiasTanH 

        # Initialize noise process.
        self.noise = OUNoise(self.actionDim)

        # Initialize replay buffer.
        self.buffer = ReplayBuffer(self.bufferSize)

        # Initialize logging.
        logging.basicConfig(filename = logFile,
                            level = logging.INFO,
                            format = '[%(asctime)s] %(message)s',
                            datefmt = '%m/%d/%Y %I:%M:%S %p')
        logging.info('Initializing DRPG agent with passed settings.')

        # Tensorflow GPU optimization.
        config = tf.ConfigProto() # GPU fix?
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config = config)
        from keras import backend as K
        K.set_session(self.sess)

        # Make actor network (creates target model internally).
        self.actor = Actor(self.sess, self.maxSteps, self.featureDim,
                           self.actionDim, self.batchSize, targetUpdate,
                           actorLR, self.actionScaleTanH, self.actionBiasTanH)

        # Make critic network (creates target model internally).
        self.critic = Critic(self.sess, self.maxSteps, self.featureDim,
                             self.actionDim, self.batchSize, targetUpdate,
                             actorLR)

    # Train or run for some number of episodes.
    def run(self, numEpisodes, training = False, warmUp = 30):
        for i in range(numEpisodes):
            sequence = []
            totalReward = 0
            totalSteps = 0
            o = self.env.reset()

            # Stores (O1, A1, O2, A2, etc) for prediction.
            history = np.zeros((self.maxSteps * self.featureDim))
            history[:self.stateDim] = o
            for j in range(self.maxSteps - 1):
                # We do this reshaping to get history into (BatchSize, TimeSteps, Dims).
                batchedHistory = np.reshape(history, (self.maxSteps, self.featureDim))
                batchedHistory = np.expand_dims(batchedHistory, axis = 0)

                # Predict action or use random with e-greedy.
                # if (np.random.random_sample() < self.epsilon and training):
                #     a = np.random.random((self.actionDim))
                #     a = a * self.actionScaleZeroOne
                #     a = a + self.actionBiasZeroOne
                # else:
                #     a = self.actor.model.predict(batchedHistory)[0]

                # Predict an action and add noise to it for exploration purposes.
                a = self.actor.model.predict(batchedHistory)[0] + self.epsilon * self.noise.noise()
                a = np.clip(a, self.minAction, self.maxAction)

                # Take a single step.
                oPrime, r, d, _ = self.env.step(a)
                r *= self.rewardScale

                newTimeStart = (j + 1) * self.featureDim
                # Update agent state and ongoing agent history data. History is
                # passed to our actor for prediction, and sequence is for later.
                history[j * self.featureDim + self.stateDim:newTimeStart] = a
                history[newTimeStart:(j + 1) * self.featureDim + self.stateDim] = oPrime
                sequence.append({'o': o, 'a': a, 'r': r, 'd': d})
                totalReward += r
                totalSteps += 1
                o = oPrime

                # Quit early.
                if d: break

            # Anneal epsilon.
            if i > warmUp:
                self.epsilon *= self.decay

            # Print some episode debugging and reward information.
            print('Episode: %03d / Steps: %d / Reward: %f' % (i + 1, totalSteps, totalReward / self.rewardScale))
            logging.info('Episode: %03d / Steps: %d / Reward: %f' % (i + 1, totalSteps, totalReward / self.rewardScale))

            # Simulation only.
            if not training:
                continue

            # Add sequence to buffer.
            self.buffer.add(sequence)

            # Resample sequences from the buffer
            samples = self.buffer.getBatch(self.batchSize)
            numSamples = len(samples)

            # Do not train until we have
            # seen self.warmUp episodes.
            if self.buffer.getCount() < warmUp:
                continue

            # Some more debug info.
            print('Training on sampled sequences from all episodes.')
            logging.info('Training on sampled sequences from all episodes.')

            # All of these do not include time step t = T.
            # Used to store H[i, t] for each episode i and step t.
            sampleHistories = np.zeros((numSamples, self.maxSteps - 1,
                                        self.maxSteps * self.featureDim))
            # Used to store H[i, t + 1] for each episode i and step t.
            sampleHistoriesWithNext = np.zeros((numSamples, self.maxSteps - 1,
                                                self.maxSteps * self.featureDim))
            # Used to store R[i, t] for each episode i and step t.
            sampleRewards = np.zeros((numSamples, self.maxSteps - 1))
            # Used to store NotDone[i, t] for each episode i and step t.
            sampleNotDoneMasks = np.zeros((numSamples, self.maxSteps - 1))
            # Used to store action[i, t] taken for each episode i and step t.
            sampleActions = np.zeros((numSamples, self.maxSteps - 1, self.actionDim))

            # Compute info for each episode i.
            for i in range(numSamples):
                sample = samples[i]
                historySoFar = np.zeros((self.maxSteps * self.featureDim))
                # Iteratively build up historySoFar for each timestep t.
                for t in range(len(sample) - 1):
                    step, nextStep = sample[i], sample[i + 1]
                    # This is (oT, aT), which we are adding to running history.
                    history = np.concatenate([step['o'], step['a']], axis = 0)
                    historySoFar[t * self.featureDim:(t + 1) * self.featureDim] = history

                    # This is (o1, a1, o2, a2 ... ot).
                    sampleHistoryEnd = (t + 1) * self.featureDim - self.actionDim
                    sampleHistories[i, t, :sampleHistoryEnd] = historySoFar[:sampleHistoryEnd]

                    # This is (o1, a1, o2, a2 ... ot, at, ot+1).
                    sampleNextEnd = (t + 1) * self.featureDim
                    sampleHistoriesWithNext[i, t, :sampleNextEnd] = historySoFar[:sampleNextEnd]
                    sampleHistoriesWithNext[i, t, sampleNextEnd:sampleNextEnd + self.stateDim] = nextStep['o']

                    # Set rewards and not done masks.
                    sampleRewards[i, t] = step['r']
                    sampleActions[i, t] = step['a']
                    sampleNotDoneMasks[i, t] = 0 if step['d'] else 1

            # Separate out self.maxSteps since it is the timestep dimension for RNN.
            sampleHistories = np.reshape(sampleHistories, (numSamples, self.maxSteps - 1,
                                                           self.maxSteps, self.featureDim))

            # Separate out self.maxSteps since it is the timestep dimension for RNN.
            sampleHistoriesWithNext = np.reshape(sampleHistoriesWithNext, (numSamples, self.maxSteps - 1,
                                                                           self.maxSteps, self.featureDim))

            # Update models using samples, rewards, and masks.
            self.update(numSamples, sampleHistories, sampleHistoriesWithNext,
                        sampleRewards, sampleActions, sampleNotDoneMasks)

    # Given a bunch of experienced histories, update our models.
    def update(self, numSamples, histories, historiesNext, rewards, chosenActions, notDoneMasks):
        # Reshape [i, t] pairs to a single dimension, which will be the RNN batch dimension.
        historiesBatch = np.reshape(histories, (-1, self.maxSteps, self.featureDim))
        historiesNextBatch = np.reshape(historiesNext, (-1, self.maxSteps, self.featureDim))

        # Compute QSample targets [y] for updating the critic Q[S][A] outputs.
        targetActions = self.actor.target.predict(historiesNextBatch) # (B * (T - 1), F).
        targetQ = self.critic.target.predict([historiesNextBatch, targetActions]) # (B * (T - 1), 1).
        targetQ = np.reshape(targetQ, (numSamples, self.maxSteps - 1)) # (B, T - 1).
        y = rewards + notDoneMasks * (self.gamma * targetQ) # (B, T - 1)
        y = np.reshape(y, (numSamples * (self.maxSteps - 1), 1)) # (B * (T - 1), 1)

        # Train the critic model, passing in both the history and chosen actions.
        chosenActionsFlat = np.reshape(chosenActions, (numSamples * (self.maxSteps - 1), self.actionDim))
        # print (chosenActionsFlat.shape, historiesBatch.shape, historiesNextBatch.shape)
        self.critic.model.train_on_batch([historiesBatch, chosenActionsFlat], y)

        # Compute the gradient of the critic output WRT to its action input.
        # We cannot use chosenActions here since those were noisy predictions.
        currentActionsForGrad = self.actor.model.predict(historiesBatch)
        currentActionsGrad = self.critic.modelActionGradients(historiesBatch, currentActionsForGrad)

        # Train the actor model using the critic gradient WRT action input.
        self.actor.trainModel(historiesBatch, currentActionsGrad)

        # Update target models.
        self.actor.trainTarget()
        self.critic.trainTarget()