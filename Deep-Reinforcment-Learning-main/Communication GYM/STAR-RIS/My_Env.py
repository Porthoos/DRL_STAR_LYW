from gym import spaces
# import Paras
import copy
import math
import os
import glob
import time
from datetime import datetime

import torch
import numpy as np
import gym

from gym import spaces
import copy
import math
import os
import glob
import time
from datetime import datetime

# TODO discrete deployment?
class My_Env(gym.Env):
    def __init__(self):
        super(My_Env, self).__init__()
        self.KR = 3 # reflection users
        self.KT = 3 # transmission users

