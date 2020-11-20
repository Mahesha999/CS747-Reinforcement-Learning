import numpy as np
import random

class KlUcb:
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
    klucb_per_arm = []
    arm_with_highest_empirical_mean = 0

    def __init__(self, _no_of_arms, _horizon, _bandit_instance, _epsilon, _random_seed):
        self.no_of_arms = _no_of_arms
        self.horizon = _horizon
        self.bandit_instance = _bandit_instance
        self.epsilon = _epsilon
        self.random_seed = _random_seed
        self.random_explore_exploit_gen = random.Random(_random_seed)
        self.random_arm_gen = random.Random(_random_seed)
        
        self.arm_list = np.arange(_no_of_arms,dtype="int64")
        self.rewards_per_arm = [[] for _ in range(_no_of_arms)]
        self.number_of_pulls_per_arm = np.zeros(_no_of_arms,dtype="int64")
        self.cumulative_rewards_per_arm = np.zeros(_no_of_arms,dtype="int64")
        self.empirical_mean_per_arm = np.zeros(_no_of_arms, dtype="float64")
        self.klucb_per_arm = np.zeros(_no_of_arms, dtype="float64")

    def pull_arm_and_update_values(self, arm_to_pull):
        reward = self.bandit_instance.sample_bandit_arm(arm_to_pull)

        self.number_of_pulls_per_arm[arm_to_pull] += 1
        self.cumulative_rewards_per_arm[arm_to_pull] += reward
        self.rewards_per_arm[arm_to_pull].append(reward)
        self.empirical_mean_per_arm[arm_to_pull] = (self.cumulative_rewards_per_arm[arm_to_pull] / self.number_of_pulls_per_arm[arm_to_pull])

    def calc_kl(self, p1, p2):
        if p1 == 0:
            return (1-p1)*np.log((1-p1)/(1-p2))
        elif p1 == 1:
            return p1*np.log(p1/p2)
        return p1*np.log(p1/p2) + (1-p1)*np.log((1-p1)/(1-p2))
            
    # Exhaustive kl-ucb using numpy
    # TODO: Fix
    #  - Gives "ValueError: operands could not be broadcast together with shapes (90,) (2,)"
    # Unused. calc_klucb_bs() instead
    def calc_klucb(self, t):
        rhs_inequality = (np.log(t) + 3*np.log(np.log(t)))/self.number_of_pulls_per_arm

        for arm_index in range(self.no_of_arms):
            if self.empirical_mean_per_arm[arm_index] == 1:  #(highest possible pat) there is no interval of values between pat and klucbat, we have converged to 1
                self.klucb_per_arm[arm_index] = 1 
            q = np.arange(self.empirical_mean_per_arm[arm_index], 1, 0.01)
            lhs_inequality_list = []
            for qi in q:
                lhs_inequality_list.append(self.calc_kl(self.empirical_mean_per_arm[arm_index],qi))
            lhs_inequality = np.array(lhs_inequality_list)
            lhs_minus_rhs = lhs_inequality - rhs_inequality  #since lhs <= rhs, lhs_minus_rhs <= 0
            lhs_minus_rhs[lhs_minus_rhs <= 0] = np.inf #since lhs <= rhs, lhs_minus_rhs <= 0, set all such lower KLs to infinity to help find first KL satisfying inequality
            self.klucb_per_arm[arm_index] = q[lhs_minus_rhs.argmin()]
    
    # kl-ucb based on binary search
    def calc_klucb_bs(self, t):
        rhs_inequality = (np.log(t) + 3*np.log(np.log(t)))/self.number_of_pulls_per_arm

        for i in range(self.no_of_arms):
            if self.empirical_mean_per_arm[i] == 1:  #(highest possible pat) there is no interval of values between pat and klucbat, we have converged to 1
                self.klucb_per_arm[i] = 1 
            
            self.klucb_per_arm[i] = self.bin_search_q(i, self.empirical_mean_per_arm[i], 1, rhs_inequality[i], self.empirical_mean_per_arm[i])
            pass

    def bin_search_q(self, arm_index, q0, qn, rhs_inequality, prev_q):
        if(abs(q0-qn)<=0.000001):
            return prev_q
        q_mid = (q0+qn)/2 
        kl = self.calc_kl(self.empirical_mean_per_arm[arm_index],q_mid)

        if kl <= rhs_inequality:
            return self.bin_search_q(arm_index, q_mid, qn, rhs_inequality, q_mid)
        else:
            return self.bin_search_q(arm_index, q0, q_mid, rhs_inequality, prev_q)
        pass
    
    def simulate(self):
        
        #EXPLORE-EXPLOIT
        for t in range(self.horizon):
            #TODO comment in all algos
            if(t == 100):
                self.REW_100 = np.sum(self.cumulative_rewards_per_arm)
            elif(t == 400):
                self.REW_400 = np.sum(self.cumulative_rewards_per_arm)
            elif(t == 1600):
                self.REW_1600 = np.sum(self.cumulative_rewards_per_arm)
            elif(t == 6400):
                self.REW_6400 = np.sum(self.cumulative_rewards_per_arm)
            elif(t == 25600):
                self.REW_25600 = np.sum(self.cumulative_rewards_per_arm)
            elif(t == 102400):
                self.REW_102400 = np.sum(self.cumulative_rewards_per_arm)

            #ASSUMPTION for first no_of_arms pulls, follow round robin, this will initialize number_of_pulls_per_arm=1 
            # This will avoid divide by zero when we divide by number_of_pulls_per_arm
            
            # ASSUMPTION: round robin for a bit longer. Otherwise, there is high chance that some arm
            # will have empirical mean of 1 (when it gets pulled in the only round robin pull) and 
            # klucb will never be employed on it as q in [pat,1] if t < self.no_of_arms:  
            if t < 20:  
                arm_to_pull = t % self.no_of_arms #follow round robin
            else:
                self.calc_klucb_bs(t)
                arm_to_pull = np.argmax(self.klucb_per_arm)

            self.pull_arm_and_update_values(arm_to_pull) 

        self.REW = np.sum(self.cumulative_rewards_per_arm)


