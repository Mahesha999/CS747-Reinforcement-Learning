import random 

class BanditInstance:
    population = [0,1]
    weights = []
    noOfArms = None
    random_gen = None

    def __init__(self, _seed, _population, _weights):
        self.random_gen = random.Random(_seed)
        self.population = _population
        self.weights = _weights
        self.noOfArms = len(_weights)

    def sample_bandit_arm(self, arm_index):
        if self.random_gen.random() < self.weights[arm_index]:
            return 1
        return 0