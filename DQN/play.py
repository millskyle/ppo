import tensorflow as tf
import numpy as np
from dqn import DQN
import gym
import cleangym
import roboschool
import kmgym
import logging
import sys
sys.path.append("..")
from supporting.utility import LinearSchedule
import colored_traceback.auto

logging.basicConfig(level=logging.DEBUG)


""" Main file that initializes an environment,
    sets up the policy networks, training algorithm, etc.
"""
from main import STATE_SEQ_LENGTH, CHKPT_PATH
from main import env


RESTORE = True

"""Number of episodes to play, with an env.reset() between"""
EPISODES = 1

if __name__=='__main__':

    dqn = DQN(env=env, restore=RESTORE, state_sequence_length=STATE_SEQ_LENGTH,
              checkpoint_path=CHKPT_PATH)

    with tf.Session() as sess:
        dqn.attach_session(sess)

        for ep in range(EPISODES):
            obs = env.reset()
            #add observation to the sequence buffer
            dqn._sequence_buffer.add(obs)

            dqn._start_of_episode()
            while True:
                obs_seq, _ = dqn._sequence_buffer.dump()
                #request the index of the action
                action = dqn.get_action(obs_seq, epsilon=0.0)

                #take the action
                next_obs, reward, done, info = env.step(action)
                env.render()
                dqn._sequence_buffer.add(next_obs)
                obs_seq_tp1, _ = dqn._sequence_buffer.dump()

                if done:
                    break
