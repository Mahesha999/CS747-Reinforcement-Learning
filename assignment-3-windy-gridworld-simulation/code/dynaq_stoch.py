import numpy as np
import os

class DynaQStochastic:

    class Model:

        def __init__(self, _num_states, _num_actions):
            self.num_states = _num_states
            self.num_actions = _num_actions

            self.T = np.zeros((_num_states, _num_states, _num_states))
            self.R = np.zeros((_num_states, _num_states, _num_states)) 
            self.total_s_a_count = np.zeros((_num_states, _num_states)) #totalVisits

            self.total_s_a_sdash_count = np.zeros((_num_states, _num_states, _num_states)) #totalTransitions
            self.total_reward = np.zeros((_num_states, _num_states, _num_states)) #totalReward

            self.states_action_visited = {}            

        
        def update_model(self, s, a, r, s_dash):
            ''' Model learning
            '''
            self.total_s_a_sdash_count[s][a][s_dash] += 1
            self.total_s_a_count[s][a] += 1
            self.total_reward[s][a][s_dash] += r

            # TODO below seems incorrect, should update for all s_dash    
            #self.T[s][a][s_dash] = self.total_s_a_sdash_count[s][a][s_dash] / self.total_s_a_count[s][a]
            self.T[s][a][:] = self.total_s_a_sdash_count[s][a][:] / self.total_s_a_count[s][a]

            #TODO should reward be also calculated? or it should be constant?
            self.R[s][a][s_dash] = self.total_reward[s][a][s_dash] / self.total_s_a_sdash_count[s][a][s_dash]

            if s in self.states_action_visited:
                self.states_action_visited[s].append(a)
            else:
                self.states_action_visited[s] = [a]


        def get_random_state_action(self):
            ''' Search control
            '''
            random_prev_state = np.random.choice(list(self.states_action_visited.keys()))
            #random_prev_action = np.random.choice(self.states_action_visited[random_prev_state])
            random_prev_action = np.random.choice(self.states_action_visited[random_prev_state])
            return random_prev_state, random_prev_action

        def get_next_state_reward(self, s, a):
            ''' Model experience
            '''
            s_dash = np.random.choice(range(self.num_states),p=self.T[s][a])
            return s_dash, self.R[s][a][s_dash] 
            

    def __init__(self, _gridworld, _num_states, _num_actions, _alpha, _epsilon, _gamma, _n, _start_state, _end_state, _seed, _num_episodes=170):
        self.alpha = _alpha
        self.epsilon = _epsilon
        self.gamma = _gamma # discount

        self.num_states = _num_states
        self.num_actions = _num_actions

        self.start_state = _start_state
        self.end_state = _end_state

        self.Q = np.zeros((_num_states, _num_actions)) #actually a Qhat
        self.model = self.Model(_num_states, _num_actions)
        self.gridworld = _gridworld  #environment transition

        self.n = _n 
        self.num_episodes = _num_episodes
        self.cummulative_timesteps_for_episode = np.zeros(_num_episodes, dtype=int)   

        self.log = open('dynaq.log', 'a')

        np.random.seed = _seed 
        
    def simulate(self):
        '''
        Q'(s,a) = Q(s,a) + α { target - Q(s,a)}
        target = r + γ Q(s',a')
        '''
        cumulative_timesteps = 0
        for episode in range(self.num_episodes):
            #initialize and start each episode from start state
            curr_state = self.start_state

            print(str(np.random.seed) + '-' + str(self.n) + '-' + str(episode) + ':', file=self.log, flush=True)

            #run episode till it reaches end state while updating Qhat and counting timesteps per episode
            while curr_state != self.end_state:
                # Llearning from environment
                curr_action = self.select_action_with_epislon_greedy(curr_state)
                next_state, reward = self.gridworld.get_next_state_reward(curr_state,curr_action)
                # next_action = self.select_action_with_epislon_greedy(next_state)
                self.Q[curr_state][curr_action] += self.alpha * (reward + self.gamma * np.max(self.Q[next_state]) - self.Q[curr_state][curr_action])
                self.model.update_model(curr_state, curr_action, reward, next_state)
                curr_state = next_state #TODO check if this is correct
                cumulative_timesteps += 1
                
                #print(str(np.random.seed) + '-' + str(self.n) + '-' + str(episode) + '-' + str(curr_state) + ':', file=self.log, end=" ", flush=True)
                # self.log.flush()

                # Planning (learning from model samples)
                for i in range(self.n):
                    #print(str(episode) + '-' + str(curr_state) + '-' + str(i), file=self.log)
                    #print(str(i), file=self.log, end=", ", flush=True)
                    #self.log.flush()
                    # print(str(episode) + '-' + str(curr_state) + '-' + str(i), file='dynaq.log')
                    # rand_s = np.random.choice(self.num_states)
                    # rand_a = np.random.choice(self.num_actions)
                    rand_prev_state, rand_prev_action = self.model.get_random_state_action()
                    s_dash, r = self.model.get_next_state_reward(rand_prev_state, rand_prev_action)
                    self.Q[rand_prev_state][rand_prev_action] += self.alpha * (r + self.gamma * np.max(self.Q[s_dash]) - self.Q[rand_prev_state][rand_prev_action])

                #print("", file=self.log, flush=True)

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

        description = separator.join(('Q-Learning on ' + self.gridworld.name,
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





