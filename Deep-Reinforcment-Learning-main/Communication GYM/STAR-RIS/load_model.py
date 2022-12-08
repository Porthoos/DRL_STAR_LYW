import gym
import numpy as np

from stable_baselines3 import TD3
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from My_Env_v1 import My_Env

model = PPO.load("models/MyEnv_PPO/test_env_3.zip")

env = My_Env()
env.reset()
for i in range(100):
    print(i)

    print(env.P_K_list)
    # env.P_K_list -= 10
    action, states = model.predict(env.get_state())
    obs, rewards, done, info = env.step(action)
    print(env.type)
    print(env.STAR_position)
    print(env.link_position)
    print(env.data_rate_list)
    print(rewards)
    print()

    if done:
        env.reset()

env.reset()
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
print(mean_reward, std_reward)
