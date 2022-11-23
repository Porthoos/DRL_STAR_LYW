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

        self.K = 6    #total users

        self.M = 4                  #antenna number
        self.N = 10                 #STAR-RIS element number
        self.N_h = 2                #horizontal element number
        self.N_v = self.N/self.N_h  #vertical element number

        self.power_unit = 100       #TODO for each user power unit?
        self.B = 1                  #MHz? to calculate data rate by multiple SINR
        self.noise_power = 3*10**(-13)            #noise power

        self.T = 50                 #DRL max steps
        self.sum_rate = 0           #reward as sum-rate

        self.W_list = np.ones(shape=(self.M, self.K)) + 0 * 1j

        #ndarray for saving CSI at each fading block t, randomly initialized, BS to KR, BS to KT, STAR-RIS to KR, STAR-RIS to KT, and BS to STAR-RIS
        # self.H_B_KR = np.random.normal(scale=1, size=(self.M, self.KR)) + np.random.normal(scale=1, size=(self.M, self.KR)) * 1j
        # self.H_B_KT = np.random.normal(scale=1, size=(self.M, self.KT)) + np.random.normal(scale=1, size=(self.M, self.KT)) * 1j
        # self.H_R_KR = np.random.normal(scale=1, size=(self.N, self.KR)) + np.random.normal(scale=1, size=(self.N, self.KR)) * 1j
        # self.H_R_KT = np.random.normal(scale=1, size=(self.N, self.KT)) + np.random.normal(scale=1, size=(self.N, self.KT)) * 1j
        # self.H_B_R = np.random.normal(scale=1, size=(self.N, self.M)) + np.random.normal(scale=1, size=(self.N, self.M)) * 1j

        self.CSI_B_K = np.random.normal(scale=1, size=(self.M, self.K)) + np.random.normal(scale=1, size=(self.M, self.K)) * 1j
        self.CSI_R_K = np.random.normal(scale=1, size=(self.N, self.K)) + np.random.normal(scale=1, size=(self.N, self.K)) * 1j
        self.CSI_B_R = np.random.normal(scale=1, size=(self.M, self.N)) + np.random.normal(scale=1, size=(self.M, self.N)) * 1j

        # random fading matrix for all time slots
        # self.G_B_KR = np.random.normal(scale=1, size=(self.M, self.KR, self.T)) + np.random.normal(scale=1, size=(self.M, self.KR, self.T)) * 1j
        # self.G_B_KT = np.random.normal(scale=1, size=(self.M, self.KT, self.T)) + np.random.normal(scale=1, size=(self.M, self.KT, self.T)) * 1j
        # self.G_R_KR = np.random.normal(scale=1, size=(self.N, self.KR, self.T)) + np.random.normal(scale=1, size=(self.N, self.KR, self.T)) * 1j
        # self.G_R_KT = np.random.normal(scale=1, size=(self.N, self.KT, self.T)) + np.random.normal(scale=1, size=(self.N, self.KT, self.T)) * 1j
        # self.G_B_R = np.random.normal(scale=1, size=(self.N, self.M, self.T)) + np.random.normal(scale=1, size=(self.N, self.M, self.T)) * 1j

        self.FD_B_K = np.random.normal(scale=1, size=(self.M, self.K, self.T)) + np.random.normal(scale=1, size=(self.M, self.K, self.T)) * 1j
        self.FD_R_K = np.random.normal(scale=1, size=(self.N, self.K, self.T)) + np.random.normal(scale=1, size=(self.N, self.K, self.T)) * 1j
        self.FD_B_R = np.random.normal(scale=1, size=(self.M, self.N, self.T)) + np.random.normal(scale=1, size=(self.M, self.N, self.T)) * 1j

        # fading intensity
        self.fading_scale_BS = 0.1
        self.fading_scale_RIS = 0.2
        # self.G_B_KR = self.fading_scale_BS * self.G_B_KR
        # self.G_B_KT = self.fading_scale_BS * self.G_B_KT
        # self.G_R_KR = self.fading_scale_RIS  * self.G_R_KR
        # self.G_R_KT = self.fading_scale_RIS  * self.G_R_KT
        # self.G_B_R = self.fading_scale_RIS  * self.G_B_R
        self.FD_B_K = self.fading_scale_BS * self.FD_B_K
        self.FD_R_K = self.fading_scale_RIS * self.FD_R_K
        self.FD_B_R = self.fading_scale_RIS * self.FD_B_R #TODO should be RIS fading scale?
        self.Rice = 5 # Rician factor

        # positions
        self.BS_position = [2000, 2000, 5]
        self.STAR_position = [0, 0, 200]
        self.link_position = [0, 0, 0]
        self.type = np.zeros(shape=(1, self.K))
        # self.P_KR_list = np.random.normal(scale=3, size=(3, self.KR)) + 3
        # self.P_KT_list = np.random.normal(scale=3, size=(3, self.KT)) - 3
        # self.P_KR_list[2, :] = 0 #R user hight
        # self.P_KT_list[2, :] = 0 #T user hight
        # self.P_KR_list_initial = copy.deepcopy(self.P_KR_list)
        # self.P_KT_list_initial = copy.deepcopy(self.P_KT_list)
        self.P_K_list = np.random.normal(scale=3, size=(3, self.K)) + 3
        self.P_K_list[3, :] = 0
        self.P_K_list_initial = copy.deepcopy(self.P_K_list)
        self.t = 0

        # create vectors for storing the R/T coefficients
        # theta represents phase shift
        self.theta_R = np.random.normal(scale=1, size=(self.N)) + np.random.normal(scale=1, size=(self.N)) * 1j
        self.theta_T = np.random.normal(scale=1, size=(self.N)) + np.random.normal(scale=1, size=(self.N)) * 1j
        # Theta_eye represents the matrix to save phase shift
        self.Theta_eye_R = np.eye(self.N)
        self.Theta_eye_T = np.eye(self.N)
        # save data rate for each user
        self.data_rate_list = np.zeros(self.K)
        # self.data_rate_list_R = np.zeros(self.KR)
        # self.data_rate_list_T = np.zeros(self.KT)

        # N reflection phase shift, N reflection amplitude, N transmission phase shift, M*K phase shift, M*K power,
        # x y z position for STAR_RIS, number of linked user, move up / down
        self.action_dim = 3 * self.N + 2 * self.M * self.K + 2 + 1 + 1
        self.action_space = spaces.Box(low=0, high=1, shape=(self.action_dim,), dtype=np.float32)
        self.num_states = 2*self.M * self.K + 2*self.N * self.K + 2*self.M*self.N
        self.observation_space = spaces.Box(low=0, high=100, shape=(self.num_states,), dtype=np.float32)




    #TODO calculate CSI information
    def calculate_CSI(self):

        return

    #TODO calculate data rate for each user
    def calculate_datarate(self):

        return

    #TODO get the observation environment
    def get_state(self):

        return

    #TODO user random move
    def user_move(self):
        # perform random movement for users
        self.P_K_list[0, :] = self.P_K_list[0, :] + np.random.normal(scale=0.5, size=(1, self.K))
        self.P_K_list[1, :] = self.P_K_list[1, :] + np.random.normal(scale=0.5, size=(1, self.K))

    # TODO divide users into reflection and transmission
    # 1 represents reflection
    # -1 represents transmission
    def divide(self):
        for i in range(self.K):
            if ((self.P_K_list[i][0] - self.STAR_position[0]) / (self.link_position[0] - self.STAR_position[0]) -
                (self.P_K_list[i][1] - self.STAR_position[1]) / (self.link_position[1] - self.STAR_position[1])) * \
                ((self.BS_position[0] - self.STAR_position[0]) / (self.link_position[0] - self.STAR_position[0]) -
                 (self.BS_position[1] - self.STAR_position[1]) / (self.link_position[1] - self.STAR_position[1])) >= 0:
                self.type[i] = 1
            else:
                self.type[i] = -1


    #TODO step function
    def step(self, action):
        action = action.reshape(-1)

        # reflection coefficient
        phaseshift_relfection = action[0:self.N] * math.pi
        amplitude_reflection = (action[self.N:2*self.N] + 1) / 2 # why +1/2
        self.theta_R = math.cos(phaseshift_relfection) * amplitude_reflection + math.sin(phaseshift_relfection) * amplitude_reflection * 1j
        self.Theta_eye_R = np.eye(self.N) * self.theta_R

        # transmission coefficient  >=0 pi/2, <0 -pi/2
        phaseshift_transmission = (action[2*self.N:3*self.N] >= 0) * 1 * math.pi/2 + (action[2*self.N:3*self.N] < 0) * -1 * math.pi/2
        amplitude_transmission = np.sqrt(1 - amplitude_reflection**2)
        self.theta_T = math.cos(phaseshift_transmission) * amplitude_transmission + math.sin(phaseshift_transmission) * amplitude_transmission * 1j
        self.Theta_eye_T = np.eye(self.N) * self.theta_T

        # BS beamforming
        phaseshift_BS = action[3*self.N:3*self.N+self.M*self.K] * math.pi
        power_BS = (action[3*self.N+self.M*self.K:3*self.N+2*self.M*self.K] + 1)/2 * self.power_unit
        w_array = np.cos(phaseshift_BS) * power_BS + np.sin(phaseshift_BS) * power_BS * 1j
        self.W_list = np.reshape(w_array, (self.M, self.K))

        # STAR-RIS position and face direction
        self.STAR_position = [action[3*self.N+2*self.M*self.K]*100, action[3*self.N+2*self.M*self.K+1]*100, 200]
        self.link_position = [self.P_K_list[0][6*action[3*self.N+2*self.M*self.K+2]] + action[3*self.N+2*self.M*self.K+3] * 10,
                              self.P_K_list[1][6*action[3*self.N+2*self.M*self.K+2]],
                              self.P_K_list[2][6*action[3*self.N+2*self.M*self.K+2]]]

        self.divide()
        self.calculate_datarate()
        self.sum_rate = sum(self.data_rate_list)

        self.user_move()
        self.calculate_CSI()
        next_state = self.get_state()
        self.t += 1
        if self.t >= self.T:
            done = True
        else:
            done = False

        return np.array([next_state]).astype(np.float32), self.sum_rate, done

    #TODO reset the environmrnt, user position, time, observation state, STAR position???
    def reset(self, *args, **kwargs):
        self.P_K_list = copy.deepcopy(self.P_K_list_initial)
        state = self.get_state()
        self.t = 0
        return np.array([state]).astype(np.float32)

    #TODO