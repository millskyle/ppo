import tensorflow as tf
import numpy as np
from dqn import DQN
import gym
import cleangym
import roboschool
import logging
logging.basicConfig(level=logging.INFO)

""" Main file that initializes an environment,
    sets up the policy networks, training algorithm, etc.
"""

EPISODES = 100000000
CHKPT_PATH = './model/'
RESTORE = True



STATE_SEQ_LENGTH = 4  # each state will be made up of this many "observations"

#env = gym.make('MountainCar-v0')
env = gym.make('CartPole-v0')

#env = gym.make('RoboschoolPong-v1')

if __name__=='__main__':

    dqn = DQN(env=env, restore=RESTORE, state_sequence_length=STATE_SEQ_LENGTH,
              )

    with tf.Session() as sess:
        dqn.attach_session(sess)

        for ep in range(EPISODES):
            obs = env.reset()

            dqn._start_of_episode()
            while True:
                dqn._before_env_step()
                #add observation to the sequence buffer
                dqn._sequence_buffer.add(obs)
                obs_seq, _ = dqn._sequence_buffer.dump()
                #request the index of the action
                action = dqn.get_action(obs_seq, epsilon=0.0)

                #take the action
                next_obs, reward, done, info = env.step(action)
                dqn._replay_buffer.add((obs, action, reward, done, next_obs))
                obs = next_obs
                dqn._after_env_step()

                if done:
                    break
            dqn._end_of_episode()
