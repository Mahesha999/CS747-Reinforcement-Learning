import numpy as np
import os

class ExpectedSarsa:

    def __init__(self, _gridworld, _num_states, _num_actions, _alpha, _epsilon, _gamma, _start_state, _end_state, _seed, _num_episodes=170):
        self.alpha = _alpha
        self.epsilon = _epsilon
        self.gamma = _gamma # discount

        self.num_states = _num_states
        self.num_actions = _num_actions

        self.start_state = _start_state
        self.end_state = _end_state

        self.Q = np.zeros((_num_states, _num_actions)) #actually a Qhat
        self.gridworld = _gridworld  #environment transition

        self.num_episodes = _num_episodes
        self.cummulative_timesteps_for_episode = np.zeros(_num_episodes, dtype=int)   

        np.random.seed = _seed 
        
    def simulate(self):
        '''
        Q'(s,a) = Q(s,a) + α { target - Q(s,a)}
        target = r + γ Σ (π(s',a) Q(s',a))
        '''
        cumulative_timesteps = 0
        for episode in range(self.num_episodes):
            #initialize and start each episode from start state
            curr_state = self.start_state
            curr_action = self.select_action_with_epislon_greedy(curr_state)

            #run episode till it reaches end state while updating Qhat and counting timesteps per episode
            while curr_state != self.end_state:
                next_state, reward = self.gridworld.get_next_state_reward(curr_state, curr_action)
                next_action = self.select_action_with_epislon_greedy(next_state)

                target = reward + self.gamma * ((1-self.epsilon) * np.max(self.Q[next_state]) + self.epsilon * (np.sum(self.Q[next_state])-np.max(self.Q[next_state])))
                self.Q[curr_state][curr_action] += self.alpha * (target - self.Q[curr_state][curr_action])
                curr_state = next_state
                curr_action = next_action
                cumulative_timesteps += 1

            self.cummulative_timesteps_for_episode[episode] = cumulative_timesteps
    
    def select_action_with_epislon_greedy(self, _state):
        decision = np.random.choice(['explore','greedy'], p=[self.epsilon, 1-self.epsilon])

        if decision == 'greedy':
            return np.argmax(self.Q[_state])
        else:
            return np.random.choice(range(self.num_actions))

    def get_run_description(self):
        separator = os.linesep
        non_cumulative_timesteps_per_episode = self.cummulative_timesteps_for_episode
        non_cumulative_timesteps_per_episode[1:] -= non_cumulative_timesteps_per_episode [:-1]
        episode_info = ['episode#' + str(i) + ': ' + str(e) for i,e in enumerate(non_cumulative_timesteps_per_episode)]
        episode_info_string = separator.join(episode_info)  

        description = separator.join(('Expected Sarsa on ' + self.gridworld.name,
                        'α=' + str(self.alpha) + ',ε=' + str(self.epsilon) + ',γ=' + str(self.gamma),
                        'Number of rows in grid-world: ' + str(self.gridworld.num_rows), 
                        'Number of columns in grid-world: ' + str(self.gridworld.num_columns), 
                        'Number of states: ' + str(self.num_states), 
                        'Number of actions: ' + str(self.num_actions),
                        'Number of episodes: ' + str(self.num_episodes), 
                        'Start state: ' + str(self.start_state), 
                        'End state: ' + str(self.end_state),
                        'Random seed: ' + str(np.random.seed),
                        'Timesteps in last episode: ' + str(self.cummulative_timesteps_for_episode[-1]),
                        'Avg timesteps in last ten episode: ' + str(np.mean(self.cummulative_timesteps_for_episode[-10:])),
                        'Time steps per episode:',
                        episode_info_string))  

        return description




