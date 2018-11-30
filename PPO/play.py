import tensorflow as tf
import numpy as np
from ppo import PPO
from policy_network import PolicyNet
import sagym
import gym
import logging
import roboschool
import sys
sys.path.append("..")
from supporting.utility import get_log_path

logging.basicConfig(level=logging.INFO)

""" Main file that initializes an environment,
    sets up the policy networks, training algorithm, etc.
"""

ITERATIONS = 100000000

CHKPT_PATH = './model/'

C_1 = 0.1
C_2 = 0.0 #0 = no entropy
RESTORE = True
CURIOSITY = False
ETA = 0.2
LAMBDA=1e-1
BETA = 2e-1
GAE_T = 512
N_ACTORS = 32

#ENV = 'MountainCar-v0'
#ENV = 'MountainCarContinuous-v0'
#ENV = 'Carnot-v1'
#ENV = 'CartPole-v0'
#ENV = 'Pendulum-v0'
ENV = "RoboschoolInvertedPendulum-v1"
ENV = "RoboschoolInvertedPendulumSwingup-v1"
#ENV = 'RoboschoolHopper-v1'
#ENV = "SAContinuous-v0"
env = gym.make(ENV)

import pdb

#env.observation_space = gym.spaces.Box(low=-1, high=1, shape=(16,16,50+1))
#SPIN_ARRAY_SHAPE = (16,16,50)

def observation_process(obs):
    """Function to take the (raw) observation from gym environment
    and get it to a form that we can feed into network, save in buffer,
    etc"""
    O, B, M, E = obs
    O = np.array(O) # (reps, spins)
    O = np.reshape(np.transpose(O), SPIN_ARRAY_SHAPE) #(Lx, Ly, reps)
    O = np.expand_dims(O, axis=0) # (1, Lx, Ly, reps)


    aux = np.zeros_like(O[:,:,:,0])  # (1, Lx, Ly)
    aux[0,0,0] = B
    aux[0,1,0] = 1/B
    aux = np.expand_dims(aux, axis=-1)  # (1, Lx, Ly, 1)


    return np.concatenate((O,aux), axis=-1) # (1, Lx, Ly, ch+1)



observation_process = lambda x : np.expand_dims(x, axis=0)

if __name__=='__main__':
    obs_space = env.observation_space

    ppo = PPO(env=env, gamma=0.99,
                epsilon=0.2, c_1=C_1, c_2=C_2,
                eta=ETA,
                llambda=LAMBDA, beta=BETA, restore=RESTORE,
                output_path='./', flags={})
    ppo.set_render_mode(mode='human')

    with tf.Session() as sess:
        ppo.attach_session(sess)
        obs = observation_process(env.reset())
        #pdb.set_trace()
        #pdb.set_trace()
        reward_E = 0
        success_num = 0

        while True:
            while True:
                action, v_pred = ppo.policy.act(observation=obs, stochastic=False)
                action = action[0]
                _obs, reward_E, done, info = env.step(action)
                env.render()
                next_obs = observation_process(_obs)

                if done:
                    obs = observation_process(env.reset())
                    break

                else:
                    obs = next_obs

