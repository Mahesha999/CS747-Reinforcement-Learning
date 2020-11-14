import argparse
import os
import sys
from argparse import RawTextHelpFormatter
from environments import *
from sarsa import Sarsa
from expected_sarsa import ExpectedSarsa
from qlearning import Qlearning
import matplotlib.pyplot as plt

#TODO
# parser = argparse.ArgumentParser(exit_on_error=True)



def main(argv):
    linsep = os.linesep

    command_description =  linsep.join(('Simulates Windy Gridworld problem. The grid size is fixed to 7x10.',
                                        'The start state is fixed to [3,0] and end state to [3,7]',
                                        'The wind strength is fixed in follow pattern: [0,0,0,1,1,1,2,2,1,0]',
                                        'with leftmost 0 corresponding to index-0 column.',
                                        'You can specify which of three algorithms to run:', 
                                        '1. sarsa (sarsa)', 
                                        '2. expected sarsa (esarsa)',  
                                        '3. q-learning (ql)',                                    
                                        'You can also choose from one for four different types of world characteristics:',
                                        '1. windy + four moves (windy)',
                                        '2. windy + eight moves (windy-king)',
                                        '3. stochastic wind + eight moves (stoch-wind-king) - in this noise exists only in windy columns',
                                        '4. stochastic noise everywhere + wind + eight moves (stochallcol-wind-king) - in this noise exists on all columns'
                                    ))

    parser = argparse.ArgumentParser(description=command_description, formatter_class=RawTextHelpFormatter)

    parser.add_argument('-a','--alpha', default=0.5, help='Learning rate (default: %(default)s)', type=float)
    parser.add_argument('-e','--epsilon', default=0.1, help='Used for epsilon greedy policy (default: %(default)s)', type=float)
    parser.add_argument('-g','--gamma', default=0.5, help='Discount factor (default: %(default)s)', type=float)
    parser.add_argument('-s','--seed', default=42, help='Seed for random number generator (default: %(default)s)', type=int)
    parser.add_argument('-p','--episodes', default=170, help='Number of episodes to run (default: %(default)s)', type=int)
    parser.add_argument('-l','--algo', required=True, choices=['sarsa','ql','esarsa'], help='Algorithm to run')
    parser.add_argument('-w','--gridworld',  required=True, choices=['windy', 'windy-king', 'stoch-wind-king', 'stochallcol-wind-king'])
    parser.add_argument('-pf','--plot-file',  default='plot.png', help="Name of png file for storing plot (default: %(default)s)")
    parser.add_argument('-of','--output-data-file',  default='output.txt', help="Name of output file for storing simulation output (default: %(default)s)")

    args_dict = vars(parser.parse_args(argv))
    # args
    # print(vars(args))
    # print(args['--gridworld'])
    # print(args['--algo'])

    alpha = float(args_dict['alpha'])
    gamma = float(args_dict['gamma'])
    epsilon = float(args_dict['epsilon']) 
    seed = int(args_dict['seed'])
    num_episodes = int(args_dict['episodes'])
    algo_param = args_dict['algo']
    gridworld_param = args_dict['gridworld']
    plot_file = args_dict['plot_file'] 
    output_file = args_dict['output_data_file'] 
    # print(args_dict)

    output_file = open(output_file, 'w')
    #print(alpha, gamma, epsilon, seed, episodes, algo, gridworld)

    if gridworld_param == 'windy':
        gw = WindyGridworld()
        gw_name = "Windy Gridworld"
    elif gridworld_param == 'windy-king':
        gw = WindyGridWorldWithKingsMoves() 
        gw_name = "Windy Gridworld with Kings moves"
    elif gridworld_param == 'stoch-wind-king':
        gw = StochaticWindGridWorldWithKingsMoves()
        gw_name = "Stochastic Windy Gridworld with Kings moves"
    elif gridworld_param == 'stochallcol-wind-king':
        gw = WindyStochaticEverywhereGridWorldWithKingsMoves()
        gw_name = "Windy Gridworld with Kings moves and Stochastic noise in all columns"
      
    algo = None
    if algo_param == 'sarsa':
        algo = Sarsa(gw, gw.num_states, gw.num_actions, alpha, epsilon, gamma, gw.get_start_state(), gw.get_end_state(), seed, num_episodes)
        algo_name = 'Sarsa'
    elif algo_param == 'esarsa': 
        algo = ExpectedSarsa(gw, gw.num_states, gw.num_actions, alpha, epsilon, gamma, gw.get_start_state(), gw.get_end_state(), seed, num_episodes)
        algo_name = 'Expected Sarsa'
    elif algo_param == 'ql': 
        algo = Qlearning(gw, gw.num_states, gw.num_actions, alpha, epsilon, gamma, gw.get_start_state(), gw.get_end_state(), seed, num_episodes)
        algo_name = 'Q-Learning'

    algo.simulate()

    plt.figure(0)
    plt.plot(algo.cummulative_timesteps_for_episode, np.array(range(algo.num_episodes)), label=algo_name)
    plt.plot([], [], ' ', label='α=' + str(alpha) + ',ε=' + str(epsilon) + ',γ=' + str(gamma))
    plt.legend()
    plt.xlabel("Timesteps")
    plt.ylabel("Episodes")
    plt.title(gw_name)
    plt.savefig(plot_file, bbox_inches='tight', )

    print(algo.get_run_description(), file=output_file)
    output_file.flush()
    output_file.close()


if __name__ == "__main__":
   main(sys.argv[1:])