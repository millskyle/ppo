import tensorflow as tf
import numpy as np
from ppo import PPO
from policy_network import PolicyNet
#import sagym
import gym
import logging
import roboschool
import sys
sys.path.append("..")
from supporting.utility import get_log_path

logging.basicConfig(level=logging.INFO)

""" Main file that initializes an environment,
    sets up the policy networks, training algorithm, etc.
"""

ITERATIONS = 100000000

CHKPT_PATH = './model/'

C_1 = 0.1
C_2 = 0.0001 #0 = no entropy
RESTORE = True
CURIOSITY = False
ETA = 0.2
LAMBDA=1e-1
BETA = 2e-1
GAE_T = 512
N_ACTORS = 32

#ENV = 'MountainCar-v0'
#ENV = 'MountainCarContinuous-v0'
#ENV = 'Carnot-v1'
#ENV = 'CartPole-v0'
#ENV = 'Pendulum-v0'
#ENV = "RoboschoolInvertedPendulum-v1"
#ENV = "RoboschoolInvertedPendulumSwingup-v1"
ENV = 'RoboschoolHopper-v1'
#ENV = "SAContinuous-v0"
env = gym.make(ENV)

import pdb

#env.observation_space = gym.spaces.Box(low=-1, high=1, shape=(16,16,50+1))
#SPIN_ARRAY_SHAPE = (16,16,50)

def observation_process(obs):
    """Function to take the (raw) observation from gym environment
    and get it to a form that we can feed into network, save in buffer,
    etc"""
    O, B, M, E = obs
    O = np.array(O) # (reps, spins)
    O = np.reshape(np.transpose(O), SPIN_ARRAY_SHAPE) #(Lx, Ly, reps)
    O = np.expand_dims(O, axis=0) # (1, Lx, Ly, reps)


    aux = np.zeros_like(O[:,:,:,0])  # (1, Lx, Ly)
    aux[0,0,0] = B
    aux[0,1,0] = 1/B
    aux = np.expand_dims(aux, axis=-1)  # (1, Lx, Ly, 1)


    return np.concatenate((O,aux), axis=-1) # (1, Lx, Ly, ch+1)


class DynamicNormalizer(object):
    def __init__(self, columns, N):
        logging.info("Creating dynamic normalizer with {} columns normalizing to the last {} examples".format(columns, N))
        self.data = np.empty((N, columns))
        self.data[:] = np.nan
    
    def normalize(self, ex):
        assert len(ex) == self.data.shape[1]
        self.data = np.roll(self.data, 1, axis=1)
        self.data[0,:] = ex
        mean, std = self.compute_moments()
        return (ex - mean) / (std + 0.0001)

    def compute_moments(self):
        mean = np.nanmean(self.data, axis=0)
        std = np.nanstd(self.data, axis=0)
        return mean, std


obs_norm = DynamicNormalizer(columns=env.observation_space.shape[0], N=1000)

def observation_process(obs):
    print(obs)
    obs = obs_norm.normalize(obs)
    return np.expand_dims(obs, axis=0)

#observation_process = lambda x : np.expand_dims(x, axis=0)

if __name__=='__main__':
    obs_space = env.observation_space

    ppo = PPO(env=env, gamma=0.99,
                epsilon=0.2, c_1=C_1, c_2=C_2,
                eta=ETA,
                llambda=LAMBDA, beta=BETA, restore=RESTORE,
                output_path='./', flags={})
    ppo.set_render_mode(mode='human')

    with tf.Session() as sess:
        ppo.attach_session(sess)
        obs = observation_process(env.reset())
        #pdb.set_trace()
        #pdb.set_trace()
        reward_E = 0
        success_num = 0

        for iteration in range(ITERATIONS):
            ppo.start_of_episode()
            ppo._buffer.empty();
            run_policy_steps = 0
            ep_reward = 0

            # buffer structure
            # [ obs, a, r_total, d, obs_tp1, v_pred ]

            while True:
                run_policy_steps += 1

                #obs = np.expand_dims(np.array(obs),i axis=0)
                logging.debug(":main.py: obs.shape={obs.shape}")
                action, v_pred = ppo.policy.act(observation=obs, stochastic=True)
                action = action[0]

                ppo.before_env_step()
                _obs, reward_E, done, info = env.step(action)
                next_obs = observation_process(_obs)
                ppo.after_env_step()

                ppo._buffer.add([obs, action, reward_E, done, next_obs, np.asscalar(v_pred) ],
                                add_until_full=False )
                
                ep_reward += reward_E

                if done:
                    with tf.variable_scope("Rewards"):
                        ppo._summary_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='reward',
                                                                                           simple_value=ep_reward
                                                                                          )]), iteration)
                    obs = observation_process(env.reset())
                    ppo.end_of_episode()
                    break

                else:
                    obs = next_obs

            ppo._GAE_T = GAE_T  #TODO move this elsewhere
            #Get a buffer's worth of advantage estimates.
            A_t = ppo.truncated_general_advantage_estimate(T=ppo._GAE_T,
                                                           from_buffer=ppo._buffer,
                                                           V=5, r=2)

            assert len(A_t)==ppo._buffer.size, f"A_t of size {len(A_t)} should be the same size as buffer of size{ppo._buffer.size}"


            V_t, _ = ppo._buffer.dump_column(col=5)
            V_tp1 = V_t[1:] + [0,]

#            episode_reward, _ = ppo._buffer.dump_column(col=2)

            ppo.assign_new_to_old()

            if iteration > 0 and iteration % N_ACTORS == 0:

                for batch in range(16):
                    data_ = ppo._buffer.sample(1024)
                    ind_ = ppo._buffer.get_last_returned_indices()
                    o = np.array([d[0] for d in data_]).reshape([-1] + list(obs_space.shape))
                    otp1 = np.array([d[4] for d in data_]).reshape([-1] + list(obs_space.shape))
                    a = np.array([d[1] for d in data_])
                    r = np.array([d[2] for d in data_])

                    these_A_t = [A_t[i] for i in ind_]
                    v_pred = [V_t[i] for i in ind_]
                    v_preds_next = [V_t[i] for i in ind_]

#recall the buffer stucture is...
#[obs, np.asscalar(action), reward_E, done, next_obs, np.asscalar(v_pred) ],

                    ppo.train(observations=o,
                              actions=a,
                              rewards=r,
                              v_preds_next=[V_tp1[i] for i in ind_],
                              advantage_estimate=these_A_t,
                              observations_tp1=otp1,
                             )
