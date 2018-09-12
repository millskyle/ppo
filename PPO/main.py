import tensorflow as tf
import numpy as np
from ppo import PPO
from policy_network import PolicyNetwork
import gym


""" Main file that initializes an environment,
    sets up the policy networks, training algorithm, etc.
"""

ITERATIONS = 1000

env = gym.make('CartPole-v0')
env.seed(0)

obs_space = env.observation_space

policy = PolicyNetwork(env=env, label='policy')
old_policy = PolicyNetwork(env=env, label='old_policy')

ppo = PPO(policy=policy, old_policy=old_policy, gamma=0.95, epsilon=0.2, c_1=1, c_2=0.01)

saver = tf.train.Saver()

with tf.Session() as sess:
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

            if done:
                v_preds_next = v_preds[1:] + [0]
                obs = env.reset()
                reward = -1
                break

            else:
                obs = next_obs


        if sum(rewards) >= 195:
            success_num += 1
            if success_num >= 100:
                saver.save(sess, './model/model.chkpt')
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


        data = [observations, actions, rewards, v_preds_next, advantage_estimate]

        for epoch in range(4):
            sample_indices = np.random.randint(low=0, high=observations.shape[0], size=64)
            data_sample = data[sample_indices, ...]
            ppo.train(observations=data_sample[:,0],
                      actions=data_sample[:,1],
                      rewards=data_sample[:,2],
                      v_preds_next=data_sample[:,3],
                      advantage_estimate=data_sample[:,4]
                     )
