import tensorflow as tf
import numpy as np
from dqn import DQN
import gym
import cleangym
import roboschool
import logging
import sys
sys.path.append("..")
from supporting.utility import LinearSchedule
import colored_traceback.auto

logging.basicConfig(level=logging.DEBUG)

""" Main file that initializes an environment,
    sets up the policy networks, training algorithm, etc.
"""

EPISODES = 10000
CHKPT_PATH = './model/'
RESTORE = True
BATCH_SIZE=50
Q_SYNC_FREQ = 1000

epsilon = LinearSchedule(start=1.0, end=0.01, steps=EPISODES)

STATE_SEQ_LENGTH = 4  # each state will be made up of this many "observations"

#env = gym.make('MountainCar-v0')
env = gym.make('CartPole-v0')

#env = gym.make('RoboschoolPong-v1')

if __name__=='__main__':

    dqn = DQN(env=env, restore=RESTORE, state_sequence_length=STATE_SEQ_LENGTH,)

    with tf.Session() as sess:
        dqn.attach_session(sess)

        for ep in range(EPISODES):
            obs = env.reset()
            #add observation to the sequence buffer
            dqn._sequence_buffer.add(obs)

            dqn._start_of_episode()
            while True:
                dqn._before_env_step()
                obs_seq, _ = dqn._sequence_buffer.dump()
                #request the index of the action
                action = dqn.get_action(obs_seq, epsilon=epsilon.val(ep))

                #take the action
                next_obs, reward, done, info = env.step(action)
                dqn._sequence_buffer.add(next_obs)
                obs_seq_tp1, _ = dqn._sequence_buffer.dump()
                dqn._sequence_buffer.add(next_obs)
                dqn._replay_buffer.add((obs_seq, action, reward, done, obs_seq_tp1))
                dqn._after_env_step()

                if dqn._replay_buffer.size > BATCH_SIZE:
                    dqn.train(BATCH_SIZE)

                if dqn._total_step_counter.eval() % Q_SYNC_FREQ==0:
                    sess.run(dqn._sync_scopes_ops)

                if done:
                    break
            dqn._end_of_episode()
