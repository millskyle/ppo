import tensorflow as tf
import numpy as np
from ppo import Algorithm
from policy_network import NeuralNet
import gym
import roboschool
import logging
logging.basicConfig(level=logging.INFO)

""" Main file that initializes an environment,
    sets up the policy networks, training algorithm, etc.
"""

ITERATIONS = 100000000

""" Continue training until there are SOLVED_THRESHOLD_CONSECUTIVE_ITERATIONS
consecutive episodes with reward greater than SOLVED_THRESHOLD """
SOLVED_THRESHOLD = 595.
SOLVED_THRESHOLD_CONSECUTIVE_ITERATIONS = 100

CHKPT_PATH = './model/'

RESTORE = True


env = gym.make('Acrobot-v1')
#env = gym.make('CartPole-v0')
#env = gym.make('RoboschoolPong-v1')
env.seed(0)
env._max_episode_steps = 200.


if __name__=='__main__':
    obs_space = env.observation_space

    policy = NeuralNet(env=env, label='policy')
    old_policy = NeuralNet(env=env, label='old_policy')

    ppo = Algorithm(policy=policy, old_policy=old_policy, gamma=0.95, epsilon=0.2, c_1=1.0, c_2=0.1)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        if RESTORE:
            try:
                saver.restore(sess, tf.train.latest_checkpoint(CHKPT_PATH))
            except:
                sess.run(tf.global_variables_initializer())

        policy.attach_session(sess)
        old_policy.attach_session(sess)
        ppo.attach_session(sess)

        obs = env.reset()
        reward = 0
        success_num = 0


        for iteration in range(ITERATIONS):
            observations =[]
            actions = []
            v_preds = []
            rewards = []
            run_policy_steps = 0


            while True:
                run_policy_steps += 1
                obs = np.stack([obs]).astype(dtype=np.float32)

                act, v_pred = policy.act(observation=obs, stochastic=True)
                act = np.asscalar(act)
                v_pred = np.asscalar(v_pred)
                observations.append(obs)
                actions.append(act)
                v_preds.append(v_pred)
                rewards.append(reward)

                next_obs, reward, done, info = env.step(act)
                if iteration%250==0:
                    env.render()
                    saver.save(sess, CHKPT_PATH + 'model.chkpt')

                if done:
                    v_preds_next = np.zeros(len(v_preds))
                    v_preds_next[0:-1] = v_preds[1:]  #the last should be zero
                    obs = env.reset()
                    reward = -1
                    break

                else:
                    obs = next_obs

            print (sum(rewards))
            if sum(rewards) >= SOLVED_THRESHOLD:
                success_num += 1
                if success_num >= SOLVED_THRESHOLD_CONSECUTIVE_ITERATIONS:
                    saver.save(sess, CHKPT_PATH + 'final.chkpt')
                    print ('Model saved.')
                    break
            else:
                success_num = 0

            advantage_estimate = ppo.estimate_advantage(rewards=rewards,
                                                        v_preds=v_preds,
                                                        v_preds_next=v_preds_next)

            observations = np.reshape(observations, newshape=[-1] + list(obs_space.shape))
            actions = np.array(actions).astype(np.int32)
            rewards = np.array(rewards).astype(np.float32)
            v_preds_next = np.array(v_preds_next).astype(np.float32)
            advantage_estimate = np.array(advantage_estimate).astype(np.float32)
            advantage_estimate = (advantage_estimate - advantage_estimate.mean()) / advantage_estimate.std()

            ppo.assign_new_to_old()


            if iteration > 0 and iteration % 10 == 0:


                data = [observations, actions, rewards, v_preds_next, advantage_estimate]

                for batch in range(10):
                    sample_indices = np.random.randint(low=0, high=observations.shape[0], size=16)
                    data_sample = []
                    for i_ in sample_indices:
                        data_sample.append([observations[i_], actions[i_], rewards[i_], v_preds_next[i_], advantage_estimate[i_]])
                    #data_sample = [np.take(a=ii, indices=sample_indices, axis=0) for ii in data]

                    ppo.train(observations=data_sample[0],
                              actions=data_sample[1],
                              rewards=data_sample[2],
                              v_preds_next=data_sample[3],
                              advantage_estimate=data_sample[4]
                             )
