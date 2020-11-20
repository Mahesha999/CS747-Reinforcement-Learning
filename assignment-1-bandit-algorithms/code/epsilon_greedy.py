import numpy as np
import random

class EpsilonGreedy:
    no_of_arms = None
    horizon = None
    bandit_instance = None
    epsilon = None
    random_seed = None
    random_explore_exploit_gen = None
    random_arm_gen = None
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
    cumulative_rewards_per_arm = []
    empirical_mean_per_arm = []
    arm_with_highest_empirical_mean = 0

    def __init__(self, _no_of_arms, _horizon, _bandit_instance, _epsilon, _random_seed):
        self.no_of_arms = _no_of_arms
        self.horizon = _horizon
        self.bandit_instance = _bandit_instance
        self.epsilon = _epsilon
        self.random_seed = _random_seed
        self.random_explore_exploit_gen = random.Random(_random_seed)
        self.random_arm_gen = random.Random(_random_seed)
        
        self.arm_list = [i for i in range(_no_of_arms)]
        self.rewards_per_arm = [[] for _ in range(_no_of_arms)]
        self.number_of_pulls_per_arm = [0 for _ in range(_no_of_arms)]
        self.cumulative_rewards_per_arm = [0 for _ in range(_no_of_arms)]
        self.empirical_mean_per_arm = [0.0 for _ in range(_no_of_arms)]
    
    def pull_arm_and_update_values(self, arm_to_pull):
        reward = self.bandit_instance.sample_bandit_arm(arm_to_pull)

        self.number_of_pulls_per_arm[arm_to_pull] += 1
        self.cumulative_rewards_per_arm[arm_to_pull] += reward
        self.rewards_per_arm[arm_to_pull].append(reward)
        self.empirical_mean_per_arm[arm_to_pull] = (self.cumulative_rewards_per_arm[arm_to_pull] / self.number_of_pulls_per_arm[arm_to_pull])

    def simulate(self):
        #ASSUMPTION: explore for initial pulls to establish initial empirical means
    
        #find arm with max empirical mean 
        #ASSUMPTION: first arm with max empirical mean is chosen 
        self.arm_with_highest_empirical_mean = np.argmax(self.empirical_mean_per_arm)

        #EXPLORE-EXPLOIT
        for t in range(self.horizon):
            if(t == 100):
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

            if self.random_explore_exploit_gen.random() < self.epsilon:
                #Explore with epsilon on probability (sample arm uniormly at random)
                arm_to_pull = self.random_arm_gen.choices(self.arm_list)[0]
            else:
                #Exploit with (1-epsilon) probability (sample arm with highest empirical probability)
                arm_to_pull = self.arm_with_highest_empirical_mean
            
            self.pull_arm_and_update_values(arm_to_pull)
            self.arm_with_highest_empirical_mean = np.argmax(self.empirical_mean_per_arm)

        self.REW = np.sum(self.cumulative_rewards_per_arm)