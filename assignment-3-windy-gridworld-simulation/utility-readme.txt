Examples of running utility and output is included in report.

Utility help can be obtained by running script with `-h` option:
# python gridworld.py -h                        
usage: gridworld.py [-h] [-a ALPHA] [-e EPSILON] [-g GAMMA] [-s SEED] [-p EPISODES] -l {sarsa,ql,esarsa} -w {windy,windy-king,stoch-wind-king,stochallcol-wind-king} [-pf PLOT_FILE] [-of OUTPUT_DATA_FILE]

Simulates Windy Gridworld problem. The grid size is fixed to 7x10.
The start state is fixed to [3,0] and end state to [3,7]
The wind strength is fixed in follow pattern: [0,0,0,1,1,1,2,2,1,0]
with leftmost 0 corresponding to index-0 column.
You can specify which of three algorithms to run:
1. sarsa (sarsa)
2. expected sarsa (esarsa)
3. q-learning (ql)
You can also choose from one for four different types of world characteristics:
1. windy + four moves (windy)
2. windy + eight moves (windy-king)
3. stochastic wind + eight moves (stoch-wind-king) - in this noise exists only in windy columns
4. stochastic noise everywhere + wind + eight moves (stochallcol-wind-king) - in this noise exists on all columns

optional arguments:
  -h, --help            show this help message and exit
  -a ALPHA, --alpha ALPHA
                        Learning rate (default: 0.5)
  -e EPSILON, --epsilon EPSILON
                        Used for epsilon greedy policy (default: 0.1)
  -g GAMMA, --gamma GAMMA
                        Discount factor (default: 0.5)
  -s SEED, --seed SEED  Seed for random number generator (default: 42)
  -p EPISODES, --episodes EPISODES
                        Number of episodes to run (default: 170)
  -l {sarsa,ql,esarsa}, --algo {sarsa,ql,esarsa}
                        Algorithm to run
  -w {windy,windy-king,stoch-wind-king,stochallcol-wind-king}, --gridworld {windy,windy-king,stoch-wind-king,stochallcol-wind-king}
  -pf PLOT_FILE, --plot-file PLOT_FILE
                        Name of png file for storing plot (default: plot.png)
  -of OUTPUT_DATA_FILE, --output-data-file OUTPUT_DATA_FILE
                        Name of output file for storing simulation output (default: output.txt)
