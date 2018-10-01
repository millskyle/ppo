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

EPISODES = 10000
CHKPT_PATH = './models/'
RESTORE = False
BATCH_SIZE=128
Q_SYNC_FREQ = 2  #number of *episodes* between syncronization of Q functions
TRAINING_FREQ = 4 #Train after this many total steps

epsilon = LinearSchedule(start=1.0, end=0.01, steps=int(1e5))

STATE_SEQ_LENGTH = 1  # each state will be made up of this many "observations"

#env = gym.make('MountainCar-v0')
#env = gym.make('Carnot-v1')
#env = gym.make('Debug-v0')
env = gym.make('CartPole-v0')
#env = gym.make('KBlocker-v0')

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
                #logging.debug("Buffer size: {}".format(dqn._replay_buffer.size))
                dqn._before_env_step()
                obs_seq, _ = dqn._sequence_buffer.dump()
                #request the index of the action
                action = dqn.get_action(obs_seq, epsilon=epsilon.val(dqn._total_step_counter.eval()))

                #take the action
                next_obs, reward, done, info = env.step(action)
                dqn._after_env_step(reward=reward)
                dqn._sequence_buffer.add(next_obs)
                obs_seq_tp1, _ = dqn._sequence_buffer.dump()

                dqn._replay_buffer.add((obs_seq, action, reward, done, obs_seq_tp1), add_until_full=False)

                if dqn._total_step_counter.eval()%TRAINING_FREQ==0:
                    if dqn._replay_buffer.is_full:
                        dqn.train(BATCH_SIZE, epsilon=epsilon.val(dqn._total_step_counter.eval()))


                if done:
                    break
            dqn._end_of_episode()
            if dqn._episode_counter.eval() % Q_SYNC_FREQ==0:
                logging.debug("Syncing Qtarg <-- Q")
                sess.run(dqn._sync_scopes_ops)
