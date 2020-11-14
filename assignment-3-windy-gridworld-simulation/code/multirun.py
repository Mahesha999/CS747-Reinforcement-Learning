from sarsa import Sarsa
from qlearning import Qlearning
from expected_sarsa import ExpectedSarsa
from environments import *
import matplotlib.pyplot as plt

alpha = 0.5
epsilon = 0.1
gamma = 1 #discount

num_episodes = 170

seeds = [89, 14, 93, 75, 66, 91, 5, 30, 64, 34]
num_seeds = len(seeds)


# All algorihtms in Windy gridworld
wg = WindyGridworld()

sarsa_total_wg = np.zeros(num_episodes, dtype=int) 
qlearning_total_wg = np.zeros(num_episodes, dtype=int) 
expected_sarsa_total_wg = np.zeros(num_episodes, dtype=int) 
for seed in seeds:
    sarsa = Sarsa(wg, wg.num_states, wg.num_actions, alpha, epsilon, gamma, wg.get_start_state(), wg.get_end_state(), seed, num_episodes)
    sarsa.simulate()
    sarsa_total_wg += sarsa.cummulative_timesteps_for_episode

    qlearning = Qlearning(wg, wg.num_states, wg.num_actions, alpha, epsilon, gamma, wg.get_start_state(), wg.get_end_state(), seed, num_episodes)
    qlearning.simulate()
    qlearning_total_wg += qlearning.cummulative_timesteps_for_episode

    expected_sarsa = ExpectedSarsa(wg, wg.num_states, wg.num_actions, alpha, epsilon, gamma, wg.get_start_state(), wg.get_end_state(), seed, num_episodes)
    expected_sarsa.simulate()
    expected_sarsa_total_wg += expected_sarsa.cummulative_timesteps_for_episode

plt.figure(0)
plt.plot(sarsa_total_wg / num_seeds, np.array(range(sarsa.num_episodes)), label='Sarsa')
plt.plot(qlearning_total_wg  / num_seeds, np.array(range(qlearning.num_episodes)), label='Q-learning')
plt.plot(expected_sarsa_total_wg  / num_seeds, np.array(range(expected_sarsa.num_episodes)), label='Expected Sarsa')
plt.plot([], [], ' ', label='α=' + str(alpha) + ',ε=' + str(epsilon) + ',γ=' + str(gamma))

plt.legend()
plt.xlabel("Timesteps")
plt.ylabel("Episodes")
plt.title("Windy Gridworld")
plt.savefig('Windy Gridworld.png', bbox_inches='tight', )

# All algorihtms in Windy gridworld with Kings move
wgk = WindyGridWorldWithKingsMoves()

sarsa_total_wgk = np.zeros(num_episodes, dtype=int) 
qlearning_total_wgk = np.zeros(num_episodes, dtype=int) 
expected_sarsa_total_wgk = np.zeros(num_episodes, dtype=int) 
for seed in seeds:
    sarsa = Sarsa(wgk, wgk.num_states, wgk.num_actions, alpha, epsilon, gamma, wgk.get_start_state(), wgk.get_end_state(), seed, num_episodes)
    sarsa.simulate()
    sarsa_total_wgk += sarsa.cummulative_timesteps_for_episode

    qlearning = Qlearning(wgk, wgk.num_states, wgk.num_actions, alpha, epsilon, gamma, wgk.get_start_state(), wgk.get_end_state(), seed, num_episodes)
    qlearning.simulate()
    qlearning_total_wgk += qlearning.cummulative_timesteps_for_episode

    expected_sarsa = ExpectedSarsa(wgk, wgk.num_states, wgk.num_actions, alpha, epsilon, gamma, wgk.get_start_state(), wgk.get_end_state(), seed, num_episodes)
    expected_sarsa.simulate()
    expected_sarsa_total_wgk += expected_sarsa.cummulative_timesteps_for_episode

plt.figure(1)
plt.plot(sarsa_total_wgk / num_seeds, np.array(range(sarsa.num_episodes)), label='Sarsa')
plt.plot(qlearning_total_wgk  / num_seeds, np.array(range(qlearning.num_episodes)), label='Q-learning')
plt.plot(expected_sarsa_total_wgk  / num_seeds, np.array(range(expected_sarsa.num_episodes)), label='Expected Sarsa')
plt.plot([], [], ' ', label='α=' + str(alpha) + ',ε=' + str(epsilon) + ',γ=' + str(gamma))
plt.legend()
plt.xlabel("Timesteps")
plt.ylabel("Episodes")
plt.title("Windy Gridworld with Kings moves")
plt.savefig('Windy Gridworld with Kings move.png', bbox_inches='tight', )


# All algorihtms in Stochastic-everywhere windy gridworld with Kings move
wsgk = WindyStochaticEverywhereGridWorldWithKingsMoves()

sarsa_total_wsgk = np.zeros(num_episodes, dtype=int) 
qlearning_total_wsgk = np.zeros(num_episodes, dtype=int) 
expected_sarsa_total_wsgk = np.zeros(num_episodes, dtype=int) 
for seed in seeds:
    sarsa = Sarsa(wsgk, wsgk.num_states, wsgk.num_actions, alpha, epsilon, gamma, wsgk.get_start_state(), wsgk.get_end_state(), seed, num_episodes)
    sarsa.simulate()
    sarsa_total_wsgk += sarsa.cummulative_timesteps_for_episode

    qlearning = Qlearning(wsgk, wsgk.num_states, wsgk.num_actions, alpha, epsilon, gamma, wsgk.get_start_state(), wsgk.get_end_state(), seed, num_episodes)
    qlearning.simulate()
    qlearning_total_wsgk += qlearning.cummulative_timesteps_for_episode

    expected_sarsa = ExpectedSarsa(wsgk, wsgk.num_states, wsgk.num_actions, alpha, epsilon, gamma, wsgk.get_start_state(), wsgk.get_end_state(), seed, num_episodes)
    expected_sarsa.simulate()
    expected_sarsa_total_wsgk += expected_sarsa.cummulative_timesteps_for_episode

plt.figure(2)
plt.plot(sarsa_total_wsgk / num_seeds, np.array(range(sarsa.num_episodes)), label='Sarsa')
plt.plot(qlearning_total_wsgk  / num_seeds, np.array(range(qlearning.num_episodes)), label='Q-learning')
plt.plot(expected_sarsa_total_wsgk  / num_seeds, np.array(range(expected_sarsa.num_episodes)), label='Expected Sarsa')
plt.plot([], [], ' ', label='α=' + str(alpha) + ',ε=' + str(epsilon) + ',γ=' + str(gamma))
plt.legend()
plt.xlabel("Timesteps")
plt.ylabel("Episodes")
plt.title("Windy Stochastic-everywhere Gridworld with Kings moves")
plt.savefig('Windy Stochastic-everywehre Gridworld with Kings move.png', bbox_inches='tight', )

# All algorihtms in Stochastic windy gridworld with Kings move
swgk = StochaticWindGridWorldWithKingsMoves()

sarsa_total_swgk = np.zeros(num_episodes, dtype=int) 
qlearning_total_swgk = np.zeros(num_episodes, dtype=int) 
expected_sarsa_total_swgk = np.zeros(num_episodes, dtype=int) 
for seed in seeds:
    sarsa = Sarsa(swgk, swgk.num_states, swgk.num_actions, alpha, epsilon, gamma, swgk.get_start_state(), swgk.get_end_state(), seed, num_episodes)
    sarsa.simulate()
    sarsa_total_swgk += sarsa.cummulative_timesteps_for_episode

    qlearning = Qlearning(swgk, swgk.num_states, swgk.num_actions, alpha, epsilon, gamma, swgk.get_start_state(), swgk.get_end_state(), seed, num_episodes)
    qlearning.simulate()
    qlearning_total_swgk += qlearning.cummulative_timesteps_for_episode

    expected_sarsa = ExpectedSarsa(swgk, swgk.num_states, swgk.num_actions, alpha, epsilon, gamma, swgk.get_start_state(), swgk.get_end_state(), seed, num_episodes)
    expected_sarsa.simulate()
    expected_sarsa_total_swgk += expected_sarsa.cummulative_timesteps_for_episode

plt.figure(3)
plt.plot(sarsa_total_swgk / num_seeds, np.array(range(sarsa.num_episodes)), label='Sarsa')
plt.plot(qlearning_total_swgk  / num_seeds, np.array(range(qlearning.num_episodes)), label='Q-learning')
plt.plot(expected_sarsa_total_swgk  / num_seeds, np.array(range(expected_sarsa.num_episodes)), label='Expected Sarsa')
plt.plot([], [], ' ', label='α=' + str(alpha) + ',ε=' + str(epsilon) + ',γ=' + str(gamma))
plt.legend()
plt.xlabel("Timesteps")
plt.ylabel("Episodes")
plt.title("Stochastic Wind Gridworld with Kings moves")
plt.savefig('Stochastic Wind Gridworld with Kings move.png', bbox_inches='tight', )


# Sarsa in all worlds
plt.figure(4)
plt.plot(sarsa_total_wg / num_seeds, np.array(range(sarsa.num_episodes)), label='In Windy Gridworld')
plt.plot(sarsa_total_wgk / num_seeds, np.array(range(sarsa.num_episodes)), label='In Windy Gridworld with Kings move')
plt.plot(sarsa_total_swgk / num_seeds, np.array(range(sarsa.num_episodes)), label='In Stochastic Wind Gridworld with Kings move')
plt.plot(sarsa_total_wsgk / num_seeds, np.array(range(sarsa.num_episodes)), label='In Windy Stochastic-everywhere Gridworld with Kings move')
plt.plot([], [], ' ', label='α=' + str(alpha) + ',ε=' + str(epsilon) + ',γ=' + str(gamma))
plt.legend()
plt.xlabel("Timesteps")
plt.ylabel("Episodes")
plt.title("Sarsa")
plt.savefig('Sarsa.png', bbox_inches='tight', )

# Expected sarsa in all worlds
plt.figure(5)
plt.plot(expected_sarsa_total_wg / num_seeds, np.array(range(sarsa.num_episodes)), label='In Windy Gridworld')
plt.plot(expected_sarsa_total_wgk / num_seeds, np.array(range(sarsa.num_episodes)), label='In Windy Gridworld with Kings move')
plt.plot(expected_sarsa_total_swgk / num_seeds, np.array(range(sarsa.num_episodes)), label='In Stochastic Wind Gridworld with Kings move')
plt.plot(expected_sarsa_total_wsgk / num_seeds, np.array(range(sarsa.num_episodes)), label='In Windy Stochastic-everywhere Gridworld with Kings move')
plt.plot([], [], ' ', label='α=' + str(alpha) + ',ε=' + str(epsilon) + ',γ=' + str(gamma))
plt.legend()
plt.xlabel("Timesteps")
plt.ylabel("Episodes")
plt.title("Expected Sarsa")
plt.savefig('Expected Sarsa.png', bbox_inches='tight', )


# Sarsa in all worlds
plt.figure(6)
plt.plot(qlearning_total_wg / num_seeds, np.array(range(sarsa.num_episodes)), label='In Windy Gridworld')
plt.plot(qlearning_total_wgk / num_seeds, np.array(range(sarsa.num_episodes)), label='In Windy Gridworld with Kings move')
plt.plot(qlearning_total_swgk / num_seeds, np.array(range(sarsa.num_episodes)), label='In Stochastic Wind Gridworld with Kings move')
plt.plot(qlearning_total_wsgk / num_seeds, np.array(range(sarsa.num_episodes)), label='In Windy Stochastic-everywhere Gridworld with Kings move')
plt.plot([], [], ' ', label='α=' + str(alpha) + ',ε=' + str(epsilon) + ',γ=' + str(gamma))
plt.legend()
plt.xlabel("Timesteps")
plt.ylabel("Episodes")
plt.title("Q-Learning")
plt.savefig('Q-Learning.png', bbox_inches='tight', )
pass 

