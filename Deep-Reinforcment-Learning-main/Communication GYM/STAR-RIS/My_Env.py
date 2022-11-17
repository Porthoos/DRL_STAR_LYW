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

#TODO discrete deployment?
class My_Env(gym.Env):
    def __init__(self):
        super(My_Env, self).__init__()
        self.KR = 3                 #reflection users
        self.KT = 3                 #transmission users
        self.K = self.KT+self.KR    #total users

        self.M = 4                  #antenna number
        self.N = 10                 #STAR-RIS element number
        self.N_h = 2                #horizontal element number
        self.N_v = self.N/self.N_h  #vertical element number

        self.power_unit = 100       #TODO for each user power unit?
        self.B = 1                  #MHz? to calculate data rate by multiple SINR
        self.noise_power = 3*10**(-13)            #noise power

        self.T = 50                 #DRL max steps
        self.sum_rate = 0           #reward as sum-rate

        #ndarray for saving CSI at each fading block t, randomly initialized, BS to KR, BS to KT, STAR-RIS to KR, STAR-RIS to KT, and BS to STAR-RIS
        self.H_B_KR = np.random.normal(scale=1, size=(self.M, self.KR)) + np.random.normal(scale=1, size=(self.M, self.KR)) * 1j
        self.H_B_KT = np.random.normal(scale=1, size=(self.M, self.KT)) + np.random.normal(scale=1, size=(self.M, self.KT)) * 1j
        self.H_R_KR = np.random.normal(scale=1, size=(self.N, self.KR)) + np.random.normal(scale=1, size=(self.N, self.KR)) * 1j
        self.H_R_KT = np.random.normal(scale=1, size=(self.N, self.KT)) + np.random.normal(scale=1, size=(self.N, self.KT)) * 1j
        self.H_B_R = np.random.normal(scale=1, size=(self.N, self.M)) + np.random.normal(scale=1, size=(self.N, self.M)) * 1j

        # random fading matrix for all time slots
        self.G_B_KR = np.random.normal(scale=1, size=(self.M, self.KR, self.T)) + np.random.normal(scale=1, size=(self.M, self.KR, self.T)) * 1j
        self.G_B_KT = np.random.normal(scale=1, size=(self.M, self.KT, self.T)) + np.random.normal(scale=1, size=(self.M, self.KT, self.T)) * 1j
        self.G_R_KR = np.random.normal(scale=1, size=(self.N, self.KR, self.T)) + np.random.normal(scale=1, size=(self.N, self.KR, self.T)) * 1j
        self.G_R_KT = np.random.normal(scale=1, size=(self.N, self.KT, self.T)) + np.random.normal(scale=1, size=(self.N, self.KT, self.T)) * 1j
        self.G_B_R = np.random.normal(scale=1, size=(self.N, self.M, self.T)) + np.random.normal(scale=1, size=(self.N, self.M, self.T)) * 1j