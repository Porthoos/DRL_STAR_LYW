import copy
import math

import stable_baselines3.common.utils
from torch.utils import tensorboard

from My_Env_v1 import My_Env
from STAR_RIS_ES import STAR_RIS_Env
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from torch.utils.tensorboard import SummaryWriter

# env = gym.make('CartPole-v1')
# env = gym.make('custom_env-v0')
# env = STAR_RIS_Env()
env = My_Env()
# logger = stable_baselines3.common.utils.configure_logger(verbose=1, tensorboard_log="StableBseline/PPO")
model = TD3("MlpPolicy", env, verbose=0, tensorboard_log="TD3/")

model.learn(total_timesteps=10000000)
model.save("model/TD3")

