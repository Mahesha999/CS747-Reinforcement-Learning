import sys, getopt
import random 
import numpy as np

from bandit_instance import BanditInstance
from epsilon_greedy import EpsilonGreedy
from ucb import Ucb
from klucb import KlUcb
from thompson import Thompson
from thompson_hint import ThompsonWithHint
import os

def main(argv):
    weights = []
    try:
        opts, args = getopt.getopt(argv,"hi:a:s:e:h:",["instance=","algorithm=","randomSeed=","epsilon=","horizon="])
    except getopt.GetoptError:
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('bandit.py --instance <instance-file-path> --algorithm <algorithm--name> --randomSeed <seed-to-random-no-generator> --epsilon <bandit-epsilon> --horizon <horizon>')
        elif opt == '--instance':
            instance_file = arg
            f3 = open(instance_file, "r")
            for x in f3:
                weights.append(float(x))
            f3.close()
            no_of_arms = len(weights)
        elif opt == '--algorithm':
            algorithm = arg
        elif opt == '--randomSeed':
            random_seed = int(arg)
            random.seed(random_seed)
        elif opt == '--epsilon':
            epsilon = float(arg)
        elif opt == '--horizon':
            horizon = int(arg)

    banditInstance = BanditInstance(random_seed, [0,1], weights)
    
    np.seterr(all='ignore')
    
    if algorithm == 'epsilon-greedy':
        epsilong_greedy_instance = EpsilonGreedy(no_of_arms, horizon, banditInstance, epsilon, random_seed)
        epsilong_greedy_instance.simulate()
        MAXREW = np.max(weights) * horizon
        REG = MAXREW - epsilong_greedy_instance.REW
        print(instance_file + ', ' + algorithm + ', ' + str(random_seed) + ', ' + str(epsilon) + ', ' + str(horizon) + ', ' + str(REG))

    elif algorithm == 'ucb':
        ucb_instance = Ucb(no_of_arms, horizon, banditInstance, epsilon, random_seed)
        ucb_instance.simulate()
        MAXREW = np.max(weights) * horizon
        REG = MAXREW - ucb_instance.REW
        print(instance_file + ', ' + algorithm + ', ' + str(random_seed) + ', ' + str(epsilon) + ', ' + str(horizon) + ', ' + str(REG))

    elif algorithm == 'kl-ucb':
        klucb_instance = KlUcb(no_of_arms, horizon, banditInstance, epsilon, random_seed)
        klucb_instance.simulate()
        MAXREW = np.max(weights) * horizon
        REG = MAXREW - klucb_instance.REW
        print(instance_file + ', ' + algorithm + ', ' + str(random_seed) + ', ' + str(epsilon) + ', ' + str(horizon) + ', ' + str(REG))

    elif algorithm == 'thompson-sampling':
        thompson_instance = Thompson(no_of_arms, horizon, banditInstance, epsilon, random_seed)
        thompson_instance.simulate()
        MAXREW = np.max(weights) * horizon
        REG = MAXREW - thompson_instance.REW
        print(instance_file + ', ' + algorithm + ', ' + str(random_seed) + ', ' + str(epsilon) + ', ' + str(horizon) + ', ' + str(REG))

    elif algorithm == 'thompson-sampling-with-hint':
        thompson_hint_instance = ThompsonWithHint(no_of_arms, horizon, banditInstance, epsilon, random_seed, np.sort(weights))
        thompson_hint_instance.simulate()
        MAXREW = np.max(weights) * horizon
        REG = MAXREW - thompson_hint_instance.REW
        print(instance_file + ', ' + algorithm + ', ' + str(random_seed) + ', ' + str(epsilon) + ', ' + str(horizon) + ', ' + str(REG))

if __name__ == "__main__":
   main(sys.argv[1:])

