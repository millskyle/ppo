import tensorflow as tf
import numpy as np
from dqn import DQN
import gym
import cleangym
import logging
import sys
import progressbar
sys.path.append("..")
from supporting.utility import LinearSchedule, ParameterOverrideFile, DutyCycle
import colored_traceback.auto

logging.basicConfig(level=logging.INFO)

""" Main file that initializes an environment,
    sets up the policy networks, training algorithm, etc.
s
"""

TOTAL_STEPS = 200*1000
CHKPT_PATH = './models/'
RESTORE = False
BATCH_SIZE=256
Q_SYNC_FREQ = 64  #number of *steps* between syncronization of Q functions
TRAINING_FREQ = 4 #Train after this many total steps


epsilon = LinearSchedule(start=1.0, end=0.01, steps=int(0.75*TOTAL_STEPS))
epsilon_from_file = ParameterOverrideFile(name='epsilon', refresh_frequency=0.1)

FLAGS = {'prioritized_buffer': False,
         'double_q_learning': False,
         'multi_steps_n': 1,
         'name_prefix': 'base_',
         'learning_rate': 1e-4,

         'state_seq_length': 1
        }

env = gym.make('CartPole-v0')


bar = progressbar.ProgressBar(max_value=TOTAL_STEPS)

if __name__=='__main__':

    dqn = DQN(env=env, restore=RESTORE, checkpoint_path=CHKPT_PATH, flags=FLAGS)

    with tf.Session() as sess:
        dqn.attach_session(sess)

        while dqn._total_step_counter.eval() < TOTAL_STEPS:
            bar.update(dqn._total_step_counter.eval())
            obs = env.reset()
            #add observation to the sequence buffer
            dqn._sequence_buffer.add(obs)

            dqn._start_of_episode()
            while True:
                dqn._before_env_step()
                obs_seq, _ = dqn._sequence_buffer.dump()
                this_epsilon = epsilon_from_file.get(fallback=epsilon.val(dqn._total_step_counter.eval()))
                #request the index of the action.  DQN takes care of e-greedy, etc.
                action = dqn.get_action(obs_seq, epsilon=this_epsilon)
                #take the action
                next_obs, reward, done, info = env.step(action)
                dqn._after_env_step(reward=reward)

                #The "sequence buffer" holds the last t frames,
                #so that the observation can be comprised of multiple
                #states (e.g. history) as was done in the original dqn
                #paper.  Setting
                dqn._sequence_buffer.add(next_obs)
                obs_seq_tp1, _ = dqn._sequence_buffer.dump()

                dqn._multi_steps_buffer.add((obs_seq, action, reward, done, obs_seq_tp1), add_until_full=False)
                if dqn._multi_steps_buffer.is_full:
                    _ds, _ps = dqn._multi_steps_buffer.dump()  # get current contents of buffer, don't modify
                    _rewards = [_ds[i][2] for i in range(len(_ds))]  #extract just the rewards (the third column)
                    _returns = dqn.discount_rewards(_rewards) #discount the rewards
                    _return = np.sum(_returns)

                    _d, _p = dqn._multi_steps_buffer.popleft(1)
                    _o, _a, _r, _d, _otp1 = _d[0]
                    dqn._replay_buffer.add((_o, _a, _return, _d, _otp1), add_until_full=False)

                if dqn._total_step_counter.eval()%TRAINING_FREQ==0:
                    if dqn._replay_buffer.is_full:
                        logging.debug("{} in buffer. Training...".format(dqn._replay_buffer.size))
                        dqn.train(BATCH_SIZE, epsilon=this_epsilon)
                    else:
                        logging.debug("Filling replay buffer with experiences - size: {}".format(dqn._replay_buffer.size))

                if dqn._total_step_counter.eval() % Q_SYNC_FREQ==0:
                    logging.debug("Syncing Qtarg <-- Q")
                    sess.run(dqn._sync_scopes_ops)

                if done:
                    break
            dqn._end_of_episode()
