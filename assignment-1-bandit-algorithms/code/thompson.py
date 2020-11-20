import numpy as np
import random

class Thompson:
    no_of_arms = None
    horizon = None
    bandit_instance = None
    epsilon = None
    random_seed = None
    
    random_beta_gen = None
    arm_list = []
    REW = 0 #reward for given horizon
    REW_100 = 0 #reward for t = 100
    REW_400 = 0 #reward for t = 400
    REW_1600 = 0 #reward for t = 1600
    REW_6400 = 0 #reward for t = 6400
    REW_25600 = 0 #reward for t = 25600
    REW_102400 = 0 #reward for t = 102400

    #per arm stats
    rewards_per_arm = [] #[[0,1,0,0,...],...]
    number_of_pulls_per_arm = []
    nummber_of_successes_per_arm = []
    number_of_failures_per_arm = []
    cumulative_rewards_per_arm = []
    beta_per_arm = []

    def __init__(self, _no_of_arms, _horizon, _bandit_instance, _epsilon, _random_seed):
        self.no_of_arms = _no_of_arms
        self.horizon = _horizon
        self.bandit_instance = _bandit_instance
        self.epsilon = _epsilon
        self.random_seed = _random_seed
        np.random.seed(_random_seed)
            
        self.arm_list = np.arange(_no_of_arms,dtype="int64")
        self.rewards_per_arm = [[] for _ in range(_no_of_arms)]
        self.number_of_pulls_per_arm = np.zeros(_no_of_arms,dtype="int64")
        self.nummber_of_successes_per_arm = np.zeros(_no_of_arms,dtype="int64")
        self.number_of_failures_per_arm = np.zeros(_no_of_arms,dtype="int64")
        self.cumulative_rewards_per_arm = np.zeros(_no_of_arms,dtype="int64")
        self.beta_per_arm = np.zeros(_no_of_arms,dtype="float64")

    def calc_beta(self):
        self.beta_per_arm = self.random_beta_gen.betavariate(self.nummber_of_successes_per_arm+1, self.number_of_failures_per_arm)
        pass

    def pull_arm_and_update_values(self, arm_to_pull):
        reward = self.bandit_instance.sample_bandit_arm(arm_to_pull)

        self.number_of_pulls_per_arm[arm_to_pull] += 1
        self.cumulative_rewards_per_arm[arm_to_pull] += reward
        self.rewards_per_arm[arm_to_pull].append(reward)
        if(reward == 1):
            self.nummber_of_successes_per_arm[arm_to_pull] += 1
        else:
            self.number_of_failures_per_arm[arm_to_pull] += 1

    def simulate(self):
        #ASSUMPTION: explore for initial pulls to establish initial empirical means

        #find arm with max empirical mean 
        #ASSUMPTION: first arm with max empirical mean is chosen 
       
        #EXPLORE-EXPLOIT
        for t in range(self.horizon):
            if(t == 100):  #TODO remove these outputs, keep only one
                self.REW_100 = sum(self.cumulative_rewards_per_arm)
            elif(t == 400):
                self.REW_400 = sum(self.cumulative_rewards_per_arm)
            elif(t == 1600):
                self.REW_1600 = sum(self.cumulative_rewards_per_arm)
            elif(t == 6400):
                self.REW_6400 = sum(self.cumulative_rewards_per_arm)
            elif(t == 25600):
                self.REW_25600 = sum(self.cumulative_rewards_per_arm)
            elif(t == 102400):
                self.REW_102400 = sum(self.cumulative_rewards_per_arm)

            if t < self.no_of_arms:
                arm_to_pull = t % self.no_of_arms #follow round robin
            else:
                self.beta_per_arm = np.random.beta(self.nummber_of_successes_per_arm+1, self.number_of_failures_per_arm+1)
                arm_to_pull = np.argmax(self.beta_per_arm)            
            
            self.pull_arm_and_update_values(arm_to_pull)
            

        self.REW = np.sum(self.cumulative_rewards_per_arm)
        