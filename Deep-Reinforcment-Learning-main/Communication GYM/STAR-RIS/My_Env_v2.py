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

        self.CSI_B_K = np.random.normal(scale=1, size=(self.M, self.K)) + np.random.normal(scale=1, size=(self.M, self.K)) * 1j
        self.CSI_R_K = np.random.normal(scale=1, size=(self.N, self.K)) + np.random.normal(scale=1, size=(self.N, self.K)) * 1j
        self.CSI_B_R = np.random.normal(scale=1, size=(self.N, self.M)) + np.random.normal(scale=1, size=(self.N, self.M)) * 1j

        self.FD_B_K = np.random.normal(scale=1, size=(self.M, self.K, self.T)) + np.random.normal(scale=1, size=(self.M, self.K, self.T)) * 1j
        self.FD_R_K = np.random.normal(scale=1, size=(self.N, self.K, self.T)) + np.random.normal(scale=1, size=(self.N, self.K, self.T)) * 1j
        self.FD_B_R = np.random.normal(scale=1, size=(self.N, self.M, self.T)) + np.random.normal(scale=1, size=(self.N, self.M, self.T)) * 1j

        # fading intensity
        self.fading_scale_BS = 0.1
        self.fading_scale_RIS = 0.2
        self.FD_B_K = self.fading_scale_BS * self.FD_B_K
        self.FD_R_K = self.fading_scale_RIS * self.FD_R_K
        self.FD_B_R = self.fading_scale_RIS * self.FD_B_R #TODO should be RIS fading scale?
        self.Rice = 5 # Rician factor
        self.scale = 10000

        # positions
        self.BS_position = [2000, 2000, 5]
        self.STAR_position = [0, 0, 100]
        self.type = np.zeros(shape=(self.K, 1))
        self.P_K_list = np.random.normal(scale=3, size=(3, self.K))
        self.P_K_list[2, :] = 0
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
        self.action_dim = 3 * self.N + 2 * self.M * self.K + 2
        self.action_space = spaces.Box(low=0, high=1, shape=(self.action_dim,), dtype=np.float32)
        # BS to user CSI, STAR-RIS element to user CSI, BS to STAR-RIS element CSI
        self.num_states = 2*self.M * self.K + 2*self.N * self.K + 2*self.M*self.N
        self.observation_space = spaces.Box(low=0, high=100, shape=(self.num_states,), dtype=np.float32)


    #TODO calculate CSI information
    def calculate_CSI(self):
        # calculate pathloss from BS to STAR-RIS
        distance_B_R = np.sqrt((self.BS_position[0]-self.STAR_position[0])**2 + (self.BS_position[1]-self.STAR_position[1])**2 + (self.BS_position[2]-self.STAR_position[2])**2)
        pathloss_B_R = 10 ** (-30 / 10) * (distance_B_R ** (-2.2))

        # calculate AOD \ DOD
        # projection of incident signal on x-y plane has Beta angle with x-axis
        # projection of incident signal on x-y plane has phi angle with incident signal
        # STAR-RIS has alpha angle with y-axis
        beta_sin = (self.STAR_position[1] - self.BS_position[1]) / np.sqrt((self.STAR_position[1] - self.BS_position[1])**2 + (self.STAR_position[0] - self.BS_position[0])**2)
        beta_cos = np.sqrt(1 - beta_sin**2)
        phi_sin = (self.STAR_position[2] - self.BS_position[2]) / np.sqrt((self.STAR_position[1] - self.BS_position[1])**2 + (self.STAR_position[0] - self.BS_position[0])**2 + (self.STAR_position[2] - self.BS_position[2])**2)
        phi_cos = np.sqrt(1 - phi_sin**2)
        alpha_sin = 0
        alpha_cos = 1
        a_B = np.ones(self.M) + 0*1j
        a_R = np.ones(self.N) + 0*1j

        for i in range(self.M):
            real = np.cos(2 * math.pi * i * beta_sin)
            imag = np.sin(2 * math.pi * i * beta_sin)
            a_B[i] = real + imag * 1j

        for i in range(self.N):
            # vertical + horizontal
            # real = np.cos(2 * math.pi * (int(i/self.N_h) * phi_sin + (i - int(i/self.N_h)*self.N_h) * phi_cos * beta_sin))
            # imag = np.sin(2 * math.pi * (int(i/self.N_h) * phi_sin + (i - int(i/self.N_h)*self.N_h) * phi_cos * beta_sin))
            real = np.cos(2 * math.pi * (int(i/self.N_h) * phi_sin + (i - int(i/self.N_h)*self.N_h) * phi_cos *
                                         (beta_sin * alpha_cos - beta_cos * alpha_sin)))
            imag = np.cos(2 * math.pi * (int(i/self.N_h) * phi_sin + (i - int(i/self.N_h)*self.N_h) * phi_cos *
                                         (beta_sin * alpha_cos - beta_cos * alpha_sin)))
            a_R[i] = real + imag * 1j

        a_R = np.mat(a_R).conj().T
        a_B = np.mat(a_B)
        B_R_LOS = a_R * a_B
        B_R_NLOS = self.FD_B_R[:, :, self.t]
        self.CSI_B_R = np.sqrt(pathloss_B_R) * (np.sqrt(self.Rice / (1 + self.Rice)) * B_R_LOS + np.sqrt(1 / (1 + self.Rice)) * B_R_NLOS)

        for i in range(self.K):
            distance_B_K = max(1, np.linalg.norm(self.P_K_list[:, i].T - self.BS_position))
            distance_R_K = max(1, np.linalg.norm(self.P_K_list[:, i].T - self.STAR_position))

            pathloss_B_K = 10**(-30/10)*(distance_B_K**(-3.5))
            pathloss_R_K = 10**(-30/10)*(distance_R_K**(-3.5))
            self.CSI_B_K[:, i] = np.sqrt(pathloss_B_K) * self.FD_B_K[:, i, self.t]
            self.CSI_R_K[:, i] = np.sqrt(pathloss_R_K) * self.FD_R_K[:, i, self.t]


    #TODO calculate data rate for each user
    def calculate_datarate(self):
        self.calculate_CSI()

        for i in range(self.K):
            CSI_R_K = self.CSI_R_K[:, i].conj().T
            if self.type[i] == 1:
                h_mid = np.dot(self.Theta_eye_R, CSI_R_K)
            else:
                h_mid = np.dot(self.Theta_eye_T, CSI_R_K)
            CSI_K = np.dot(h_mid, self.CSI_B_R) + self.CSI_B_K[:, i]
            SignalPower_K = abs(np.dot(CSI_K, self.W_list[:, i].T))**2
            InterferencePoser_K = 0 - SignalPower_K
            for j in range(self.K):
                InterferencePoser_K += abs(np.dot(CSI_K, self.W_list[:, j].T))**2

            SINR_K = SignalPower_K / (InterferencePoser_K + self.noise_power)
            datarate_K = self.B * math.log((1+SINR_K), 2)
            self.data_rate_list[i] = datarate_K

        return


    #TODO get the observation environment
    def get_state(self):
        CSI_B_K_state = self.CSI_B_K.ravel()
        CSI_R_K_state = self.CSI_R_K.ravel()
        CSI_B_R_state = self.CSI_B_R.ravel()
        CSI_B_R_state = np.array(CSI_B_R_state) # why need this
        CSI_B_K_info = np.append(np.real(CSI_B_K_state), np.imag(CSI_B_K_state))
        CSI_R_K_info = np.append(np.real(CSI_R_K_state), np.imag(CSI_R_K_state))
        CSI_B_R_info = np.append(np.real(CSI_B_R_state), np.imag(CSI_B_R_state))
        state = np.append(CSI_B_R_info*self.scale, CSI_B_K_info*self.scale)
        state = np.append(state, CSI_R_K_info*self.scale)
        return state
        # TODO need confirmation
        # return [np.array(CSI_B_R_info*self.scale), np.array(CSI_B_K_info*self.scale), np.array(CSI_R_K_info*self.scale)]


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
            if self.P_K_list[0][i] > self.STAR_position[0]:
                self.type[i] = 1
            else:
                self.type[i] = -1


    #TODO step function
    def step(self, action):
        action = action.reshape(-1)

        # reflection coefficient
        phaseshift_reflection = action[0:self.N] * math.pi
        amplitude_reflection = (action[self.N:2*self.N] + 1) / 2 # why +1/2
        self.theta_R = np.cos(phaseshift_reflection) * amplitude_reflection + np.sin(phaseshift_reflection) * amplitude_reflection * 1j
        self.Theta_eye_R = np.eye(self.N) * self.theta_R

        # transmission coefficient  >=0 pi/2, <0 -pi/2
        phaseshift_transmission = (action[2*self.N:3*self.N] >= 0) * 1 * math.pi/2 + (action[2*self.N:3*self.N] < 0) * -1 * math.pi/2
        amplitude_transmission = np.sqrt(1 - amplitude_reflection**2)
        self.theta_T = np.cos(phaseshift_transmission) * amplitude_transmission + np.sin(phaseshift_transmission) * amplitude_transmission * 1j
        self.Theta_eye_T = np.eye(self.N) * self.theta_T

        # BS beamforming
        phaseshift_BS = action[3*self.N:3*self.N+self.M*self.K] * math.pi
        power_BS = (action[3*self.N+self.M*self.K:3*self.N+2*self.M*self.K] + 1)/2 * self.power_unit
        w_array = np.cos(phaseshift_BS) * power_BS + np.sin(phaseshift_BS) * power_BS * 1j
        self.W_list = np.reshape(w_array, (self.M, self.K))

        # STAR-RIS position and face direction
        self.STAR_position = [action[3*self.N+2*self.M*self.K]*100, action[3*self.N+2*self.M*self.K+1]*100, 200]

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

        return np.array([next_state]).astype(np.float32), self.sum_rate, done, {}


    #TODO reset the environmrnt, user position, time, observation state, STAR position???
    def reset(self, *args, **kwargs):
        self.P_K_list = copy.deepcopy(self.P_K_list_initial)
        state = self.get_state()
        self.t = 0
        return np.array([state]).astype(np.float32)

    #TODO