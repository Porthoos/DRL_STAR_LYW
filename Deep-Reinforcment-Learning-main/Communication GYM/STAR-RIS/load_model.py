import gym
import numpy as np

from stable_baselines3 import TD3
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from My_Env_expose_position import My_Env
from STAR_RIS_ES import STAR_RIS_Env

model = PPO.load("models/MyEnv_PPO/test_env_expose_position_addition_1.zip")
# model = PPO.load("models/LYW_model/test_1.zip")

env = My_Env()
# env = STAR_RIS_Env()
env.reset()
pos = [0,0,10]
for i in range(100):
    print(i)

    print(env.P_K_list)

    action, states = model.predict(env.get_state())
    obs, rewards, done, info = env.step(action)

    print(env.type)
    print(env.STAR_position)
    # print(np.array(env.STAR_position) - np.array(pos))
    pos = env.STAR_position
    print(env.link_position)

    # print(env.data_rate_list_R)
    # print(env.data_rate_list_T)

    print(rewards)
    print()

    if done:
        env.reset()

env.reset()
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
print(mean_reward, std_reward)
