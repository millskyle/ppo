import tensorflow as tf
import numpy as np
from ppo import PPO
from policy_network import PolicyNet
import gym
import logging
import sys
sys.path.append("..")
from supporting.utility import get_log_path

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

C_1 = 0.1
C_2 = 0.1
RESTORE = True
CURIOSITY = False
ETA = 0.2
LAMBDA=1e-1
BETA = 2e-1

#env = gym.make('MountainCar-v0')
#env = gym.make('MountainCarContinuous-v0')
#env = gym.make('Carnot-v1')
env = gym.make('CartPole-v0')
#nv = gym.make('Pendulum-v0')
#env = gym.make('RoboschoolPendulum-v1')

if __name__=='__main__':
    obs_space = env.observation_space


    ppo = PPO(env=env, gamma=0.95,
                epsilon=0.2, c_1=C_1, c_2=C_2,
                use_curiosity=CURIOSITY, eta=ETA,
                llambda=LAMBDA, beta=BETA, restore=RESTORE,
                output_path='./', flags={})

    with tf.Session() as sess:
        ppo.attach_session(sess)

        obs = env.reset()
        reward_E = 0
        success_num = 0

        for iteration in range(ITERATIONS):
            ppo.start_of_episode()
            ppo._buffer.empty();
            run_policy_steps = 0

            # buffer structure
            # [ obs, a, r_total, d, obs_tp1, v_pred ]

            while True:
                run_policy_steps += 1

                obs = np.stack([obs]).astype(dtype=np.float32)
                action, v_pred = ppo.policy.act(observation=obs, stochastic=True)

                ppo.before_env_step()
                next_obs, reward_E, done, info = env.step(action[0])
                ppo.after_env_step()

                # for curiosity, apply the intrinsic reward
                if CURIOSITY:
                    reward_I = ppo.evaluate_intrinsic_reward(obs=obs, obs_tp1=next_obs)
                else:
                    reward_I = 0.0

                ppo._buffer.add([obs, np.asscalar(action), reward_E+reward_I, done, next_obs, np.asscalar(v_pred) ],
                                add_until_full=False )


                if done:
                    obs = env.reset()
                    ppo.end_of_episode()
                    break

                else:
                    obs = next_obs



            ppo._GAE_T = 25  #TODO move this elsewhere
            #Get a buffer's worth of advantage estimates.
            A_t = ppo.truncated_general_advantage_estimate(T=ppo._GAE_T,
                                                           from_buffer=ppo._buffer,
                                                           V=5, r=2)

            assert len(A_t)==ppo._buffer.size, f"A_t of size {len(A_t)} should be the same size as buffer of size{ppo._buffer.size}"


            V_t, _ = ppo._buffer.dump_column(col=5)
            V_tp1 = V_t[1:] + [0,]

            episode_reward, _ = ppo._buffer.dump_column(col=2)
            with tf.variable_scope("Rewards"):
                ppo._summary_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='reward', simple_value=sum(episode_reward))]), iteration)

            ppo.assign_new_to_old()

            if iteration > 0 and iteration % 1 == 0:

                for batch in range(4):
                    data_ = ppo._buffer.sample(32)
                    ind_ = ppo._buffer.get_last_returned_indices()
                    o = np.array([d[0] for d in data_]).reshape([-1] + list(obs_space.shape))
                    otp1 = np.array([d[4] for d in data_]).reshape([-1] + list(obs_space.shape))
                    a = np.array([d[1] for d in data_])
                    r = np.array([d[2] for d in data_])

                    these_A_t = [A_t[i] for i in ind_]
                    v_pred = [V_t[i] for i in ind_]
                    v_preds_next = [V_t[i] for i in ind_]

#recall the buffer stucture is...
#[obs, np.asscalar(action), reward_E+reward_I, done, next_obs, np.asscalar(v_pred) ],

                    ppo.train(observations=o,
                              actions=a,
                              rewards=r,
                              v_preds_next=[V_tp1[i] for i in ind_],
                              advantage_estimate=these_A_t,
                              observations_tp1=otp1,
                             )
