import tensorflow as tf
import numpy as np
from ppo import Algorithm
from policy_network import NeuralNet
import gym
import cleangym
import roboschool
import logging
logging.basicConfig(level=logging.INFO)

""" Main file that initializes an environment,
    sets up the policy networks, training algorithm, etc.
"""

ITERATIONS = 100000000

""" Continue training until there are SOLVED_THRESHOLD_CONSECUTIVE_ITERATIONS
consecutive episodes with reward greater than SOLVED_THRESHOLD """
SOLVED_THRESHOLD = 198.
SOLVED_THRESHOLD_CONSECUTIVE_ITERATIONS = 100

CHKPT_PATH = './model/'

RESTORE = True
CURIOSITY = True
ETA = 1e-5
LAMBDA=1e-5
BETA = 2e-5

env = gym.make('MountainCar-v0')
#env = gym.make('Stirling-v0')
#env = gym.make('CartPole-v0')
#env = gym.make('RoboschoolPong-v1')

if __name__=='__main__':
    obs_space = env.observation_space

    policy = NeuralNet(env=env, label='policy')
    old_policy = NeuralNet(env=env, label='old_policy')

    ppo = Algorithm(policy=policy, old_policy=old_policy, gamma=0.95,
                    epsilon=0.2, c_1=0.1, c_2=100.0,
                    use_curiosity=CURIOSITY, eta=ETA,
                    llambda=LAMBDA, beta=BETA)

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
        reward_E = 0
        success_num = 0


        for iteration in range(ITERATIONS):
            observations = []
            observations_tp1 = []
            actions = []
            v_preds = []
            rewards_extrinsic = []
            rewards_intrinsic = []
            run_policy_steps = 0

            ppo._start_of_episode()


            while True:
                run_policy_steps += 1
                obs = np.stack([obs]).astype(dtype=np.float32)
                action, v_pred = policy.act(observation=obs, stochastic=True)
                action = np.asscalar(action)
                v_pred = np.asscalar(v_pred)
                observations.append(obs)
                actions.append(action)
                v_preds.append(v_pred)
                rewards_extrinsic.append(reward_E)
                next_obs, reward_E, done, info = env.step(action)

                observations_tp1.append(next_obs)

                # for curiosity, apply the intrinsic reward
                if CURIOSITY:
                    reward_I = ppo.evaluate_intrinsic_reward(obs=obs, obs_tp1=next_obs)
                    rewards_intrinsic.append(reward_I)
                else:
                    rewards_intrinsic.append(0.0)

                if ppo._render:
                    env.render()

                if iteration%1000==0 and iteration > -1:
                    saver.save(sess, CHKPT_PATH + 'model.chkpt')

                if done:
                    v_preds_next = v_preds[1:] + [0]  # next state of terminate state has 0 state value
                    obs = env.reset()
                    reward_E = 0
                    ppo._end_of_episode()
                    break

                else:
                    obs = next_obs


            logging.info("EPISODE {0:5d}".format(iteration))
            if sum(rewards_extrinsic) >= SOLVED_THRESHOLD:
                success_num += 1
                if success_num >= SOLVED_THRESHOLD_CONSECUTIVE_ITERATIONS:
                    saver.save(sess, CHKPT_PATH + 'final.chkpt')
                    print ('Model saved.')
                    ppo._end_of_episode()
                    break
            else:
                success_num = 0


            rewards = [i+e for i,e in zip(rewards_extrinsic, rewards_intrinsic)]

            advantage_estimate = ppo.estimate_advantage(rewards=rewards,
                                                        v_preds=v_preds,
                                                        v_preds_next=v_preds_next)
            with tf.variable_scope("Rewards"):
                ppo._summary_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='rewards_I', simple_value=sum(rewards_intrinsic))]), iteration)
                ppo._summary_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='rewards_E', simple_value=sum(rewards_extrinsic))]), iteration)


            observations = np.reshape(observations, newshape=[-1] + list(obs_space.shape))
            observations_tp1 = np.reshape(observations_tp1, newshape=[-1] + list(obs_space.shape))
            actions = np.array(actions).astype(np.int32)
            rewards = np.array(rewards).astype(np.float32)

            v_preds_next = np.array(v_preds_next).astype(np.float32)
            advantage_estimate = np.array(advantage_estimate).astype(np.float32)
            advantage_estimate = (advantage_estimate - advantage_estimate.mean()) / advantage_estimate.std()

            ppo.assign_new_to_old()

            if iteration > 0 and iteration % 1 == 0:

                data = [observations, actions, rewards, v_preds_next, advantage_estimate, observations_tp1]

                for batch in range(4):
                    sample_indices = np.random.randint(low=0, high=observations.shape[0], size=32)
                    data_sample = [np.take(a=ii, indices=sample_indices, axis=0) for ii in data]

                    ppo.train(observations=data_sample[0],
                              actions=data_sample[1],
                              rewards=data_sample[2],
                              v_preds_next=data_sample[3],
                              advantage_estimate=data_sample[4],
                              observations_tp1=data_sample[5],
                             )
