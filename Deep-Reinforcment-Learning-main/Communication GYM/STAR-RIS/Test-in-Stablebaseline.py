import copy
import math

import stable_baselines3.common.utils
from torch.utils import tensorboard

from My_Env_expose_position import My_Env
from STAR_RIS_ES import STAR_RIS_Env
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from torch.utils.tensorboard import SummaryWriter

# env = gym.make('CartPole-v1')
# env = gym.make('custom_env-v0')
# env = STAR_RIS_Env()
env = My_Env()

model = PPO(MlpPolicy, env, verbose=0, tensorboard_log="MyEnv_log/")

def evaluate(model, num_episodes=100):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last num_episodes
    """
    # This function will only work for a single Environment
    # env = model.get_env()
    all_episode_rewards = []
    for i in range(num_episodes):
        episode_rewards = []
        done = False
        obs = env.reset()
        while not done:
            # _states are only useful when using LSTM policies
            action, _states = model.predict(obs)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs, reward, done, info = env.step(action)
            episode_rewards.append(reward)

        print('episode=',i,'reward=',sum(episode_rewards))
        all_episode_rewards.append(sum(episode_rewards))

    mean_episode_reward = np.mean(all_episode_rewards)
    print("Mean reward:", mean_episode_reward, "Num episodes:", num_episodes)

    return mean_episode_reward


# Random Agent, before training
# mean_reward_before_train = evaluate(model, num_episodes=100)
# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
# print(f"mean_reward befroe training:{mean_reward:.2f} +/- {std_reward:.2f}")

# Train the agent for 10000 steps
model.learn(total_timesteps=15000000)
# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
model.save('models/MyEnv_PPO/test_env_expose_position_addition_1')
print(f"mean_reward after training:{mean_reward:.2f} +/- {std_reward:.2f}")


# #TODO predict enviroment, decode action
# env_tmp = STAR_RIS_Env()
# action = model.predict(env_tmp.get_state())
#
# print(action[0])
# print(type(action))
# print()
# action = action[0]
# action = action.reshape(-1)
# w_theta = action[3*10:3*10+4*6] * math.pi
# w_beta = (action[3*10+4*6:3*10+2*4*6] + 1)/2 * 100
# w_array = np.cos(w_theta) * w_beta + np.sin(w_theta) * w_beta * 1j
# w_array = np.reshape(w_array, (4, 6))
# print("\nBS beamforming for each user")
# for i in range(4):
#     for j in range(6):
#         print("%.2f+%.2fj   " % (w_array[i][j].real, w_array[i][j].imag), end='')
#     print()
# print()
#
# # calculate R coefficient
# New_theta_pi_R = action[0:10] * math.pi # theta in radian system
# New_beta_R = (action[10:2*10] +1)/2
# New_theta_R = np.cos(New_theta_pi_R) * New_beta_R + np.sin(New_theta_pi_R) * New_beta_R * 1j # calculate R coefficient
#
# theta_R = New_theta_R
# Theta_eye_R = np.eye(10) * theta_R # transfer R coefficient into matrix
# print("\ntransfer coefficient for reflection")
# for i in range(10):
#     for j in range(10):
#         print("%.2f+%.2fj   " % (Theta_eye_R[i][j].real, Theta_eye_R[i][j].imag), end='')
#     print()
# print()
#
# # determine the T coefficient based on R coefficient
# action_T = copy.deepcopy(action[2*10:3*10])
# action_T[action_T >= 0] = 1 * math.pi/2
# action_T[action_T >= 0] = 1 * math.pi/2
# New_theta_pi_T = New_theta_pi_R + action_T
# New_beta_T = np.sqrt(1-New_beta_R**2)
# theta_T = np.cos(New_theta_pi_T) * New_beta_T + np.sin(New_theta_pi_T) * New_beta_T * 1j
# Theta_eye_T = np.eye(10) * theta_T
# print("\ntransfer coefficient for transmission")
# for i in range(10):
#     for j in range(10):
#         print("%.2f+%.2fj   " % (Theta_eye_T[i][j].real, Theta_eye_T[i][j].imag), end='')
#     print()