import tensorflow as tf
import numpy as np
from policy_network import NeuralNet
import gym

""" Play the trained game (no training)
"""

from main import env, CHKPT_PATH
env._max_episode_steps = 5000

EPISODES = 10


obs_space = env.observation_space

policy = NeuralNet(env=env, label='policy')


saver = tf.train.Saver()


with tf.Session() as sess:
    saver.restore(sess=sess, save_path=tf.train.latest_checkpoint(CHKPT_PATH))
    policy.attach_session(sess)

    for episodes in range(EPISODES):

        obs = env.reset()
        step=-1
        while True:
            obs = np.stack([obs]).astype(dtype=np.float32)
            step+=1
            act, v_pred = policy.act(observation=obs, stochastic=False)
            act = np.asscalar(act)
            v_pred = np.asscalar(v_pred)

            next_obs, reward, done, info = env.step(act)

            print("{2:5d}\t{0:10.2f}\t{1:10.2f}".format(v_pred, reward, step))
            env.render()

            if done:
                obs = env.reset()
                break

            else:
                obs = next_obs
