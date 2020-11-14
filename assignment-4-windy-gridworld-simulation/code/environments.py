import numpy as np

class WindyGridworld:
    '''
    Simulates windy grid world with four moves or actions: right, left, up and down.
    '''

    DEFAULT = 'default'
    KINGS = 'kings'
    KINGS_STOCHASTIC = 'kings_stochastic'
    STOCHASTIC = 'stochastic'    


    #TODO decide whether to use mode, currently unused
    def __init__(self, mode = DEFAULT):
        self.num_rows = 7
        self.num_columns = 10
        self.num_actions = 4
        self.num_states = self.num_rows * self.num_columns

        self.start_position = (3,0)
        self.end_position = (3,7)
        
        self.wind_strength = np.array([0,0,0,1,1,1,2,2,1,0])

        self.NORTH = 0
        self.EAST = 1
        self.SOUTH = 2
        self.WEST = 3

        self.transitions = np.zeros((self.num_states, self.num_actions, 2), dtype=np.int) #2 for next state and reward
        self.name = 'Windy Gridworld'
        self.init_transitions()

        #TODO explicitly set next state and reward for final state?


    def get_next_state_reward(self, _state, _action):
        return self.transitions[_state, _action]


    def init_transitions(self):
        '''
        Initialise transitions in each position of grid world for each action considering the 
        wind strength at that position.
        '''
        for state in range(self.num_states):
            current_position = np.unravel_index(state, (self.num_rows, self.num_columns)) #position corresponding to current state
            for action in range(self.num_actions):
                if action == self.NORTH:
                    next_position = current_position + np.array([1,0]) + self.get_wind_effect(current_position)
                elif action == self.EAST:
                    next_position = current_position + np.array([0,1]) + self.get_wind_effect(current_position)
                elif action == self.SOUTH:
                    next_position = current_position + np.array([-1,0]) + self.get_wind_effect(current_position)
                elif action == self.WEST:
                    next_position = current_position + np.array([0,-1]) + self.get_wind_effect(current_position)

                next_position = self.restrict_tranisition_inside_gridworld(next_position)
                next_state = np.ravel_multi_index(next_position, [self.num_rows, self.num_columns])

                self.transitions[state][action][0] = int(next_state)

                if np.all(self.end_position == next_position): #if we are transitting to goal / end state
                    self.transitions[state][action][1] = 0 #reward
                else:
                    self.transitions[state][action][1] = -1 #reward

    def get_start_state(self):
        return np.ravel_multi_index(self.start_position, [self.num_rows, self.num_columns])

    def get_end_state(self):
        return np.ravel_multi_index(self.end_position, [self.num_rows, self.num_columns])

        
    def get_wind_effect(self, _position):
        return self.wind_strength[_position[1]] * np.array([1,0]) 

    def restrict_tranisition_inside_gridworld(self, _position):
        _position[0] = min(self.num_rows-1, _position[0]) #position cannot be above top most row 
        _position[0] = max(0, _position[0]) #position cannot be below bottom most row 
        _position[1] = min(self.num_columns-1, _position[1]) #position cannot be on right of rightmost column 
        _position[1] = max(0, _position[1]) #position cannot be on left of rightmost column
        return _position

class WindyGridWorldWithKingsMoves:

    def __init__(self):
        self.num_rows = 7
        self.num_columns = 10
        self.num_actions = 8
        self.num_states = self.num_rows * self.num_columns

        self.start_position = (3,0)
        self.end_position = (3,7)
        
        self.wind_strength = np.array([0,0,0,1,1,1,2,2,1,0])

        self.NORTH = 0
        self.NORTH_EAST = 1
        self.EAST = 2
        self.SOUTH_EAST = 3
        self.SOUTH = 4
        self.SOUTH_WEST = 5
        self.WEST = 6
        self.NORTH_WEST = 7

        self.transitions = np.zeros((self.num_states, self.num_actions, 2), dtype=np.int) #2 for next state and reward

        self.init_transitions()
        self.name = 'Windy Gridworld with Kings moves'

        #TODO explicitly set next state and reward for final state?

    def get_next_state_reward(self, _state, _action):
        return self.transitions[_state, _action]

    def init_transitions(self):
        '''
        Initialise transitions in each position of grid world for each action considering the 
        wind strength at that position.
        '''
        for state in range(self.num_states):
            current_position = np.unravel_index(state, (self.num_rows, self.num_columns)) #position corresponding to current state
            for action in range(self.num_actions):
                if action == self.NORTH:
                    next_position = current_position + np.array([1,0]) + self.get_wind_effect(current_position)
                elif action == self.NORTH_EAST:
                    next_position = current_position + np.array([1,1]) + self.get_wind_effect(current_position)
                elif action == self.EAST:
                    next_position = current_position + np.array([0,1]) + self.get_wind_effect(current_position)
                elif action == self.SOUTH_EAST:
                    next_position = current_position + np.array([-1,1]) + self.get_wind_effect(current_position)
                elif action == self.SOUTH:
                    next_position = current_position + np.array([-1,0]) + self.get_wind_effect(current_position)
                elif action == self.SOUTH_WEST:
                    next_position = current_position + np.array([-1,-1]) + self.get_wind_effect(current_position)
                elif action == self.WEST:
                    next_position = current_position + np.array([0,-1]) + self.get_wind_effect(current_position)
                elif action == self.NORTH_WEST:
                    next_position = current_position + np.array([1,-1]) + self.get_wind_effect(current_position)

                next_position = self.restrict_tranisition_inside_gridworld(next_position)
                next_state = np.ravel_multi_index(next_position, [self.num_rows, self.num_columns])

                self.transitions[state][action][0] = int(next_state) 
                
                if np.all(self.end_position == next_position): #if we are transitting to goal / end state
                    self.transitions[state][action][1] = 0 #reward
                else:
                    self.transitions[state][action][1] = -1 #reward
    
    def get_start_state(self):
        return np.ravel_multi_index(self.start_position, [self.num_rows, self.num_columns])

    def get_end_state(self):
        return np.ravel_multi_index(self.end_position, [self.num_rows, self.num_columns])
        
    def get_wind_effect(self, _position):
        return self.wind_strength[_position[1]] * np.array([1,0]) 

    def restrict_tranisition_inside_gridworld(self, _position):
        _position[0] = min(self.num_rows-1, _position[0]) #position cannot be above top most row 
        _position[0] = max(0, _position[0]) #position cannot be below bottom most row 
        _position[1] = min(self.num_columns-1, _position[1]) #position cannot be on right of rightmost column 
        _position[1] = max(0, _position[1]) #position cannot be on left of rightmost column
        return _position

class WindyStochaticEverywhereGridWorldWithKingsMoves:
    '''
    Stochastic-ness presents in all columns of grid, not just in windy columns.
    This differs from StochaticWindGridWorldWithKingsMoves where wind is present onlu in windy columns.
    '''

    def __init__(self):
        self.num_rows = 7
        self.num_columns = 10
        self.num_actions = 8
        self.num_states = self.num_rows * self.num_columns

        self.start_position = (3,0)
        self.end_position = (3,7)
        
        self.wind_strength = np.array([0,0,0,1,1,1,2,2,1,0])

        self.NORTH = 0
        self.NORTH_EAST = 1
        self.EAST = 2
        self.SOUTH_EAST = 3
        self.SOUTH = 4
        self.SOUTH_WEST = 5
        self.WEST = 6
        self.NORTH_WEST = 7

        self.transitions = np.zeros((self.num_states, self.num_actions, 2), dtype=np.int) #2 for next state and reward

        self.init_transitions()

        self.name = 'Windy Gridworld with Kings moves and sotchastic noise everywhere'

        #TODO explicitly set next state and reward for final state?

    def get_next_state_reward(self, _state, _action):
        '''
        Implements stochastic behavior in all columns (not just in windy columns).
        '''
        _state_position =  np.unravel_index(_state, (self.num_rows, self.num_columns))         
        next_state, reward = self.transitions[_state, _action] 
        next_position = np.unravel_index(next_state, (self.num_rows, self.num_columns)) 
        next_position = list(next_position)
        next_position[0] = next_position[0] + np.random.randint(-1,2)
        next_position = self.restrict_tranisition_inside_gridworld(next_position)
        return np.ravel_multi_index(next_position, [self.num_rows, self.num_columns]), reward
        

    def init_transitions(self):
        '''
        Initialise transitions in each position of grid world for each action considering the 
        wind strength at that position.
        '''
        for state in range(self.num_states):
            current_position = np.unravel_index(state, (self.num_rows, self.num_columns)) #position corresponding to current state
            for action in range(self.num_actions):
                if action == self.NORTH:
                    next_position = current_position + np.array([1,0]) + self.get_wind_effect(current_position)
                elif action == self.NORTH_EAST:
                    next_position = current_position + np.array([1,1]) + self.get_wind_effect(current_position)
                elif action == self.EAST:
                    next_position = current_position + np.array([0,1]) + self.get_wind_effect(current_position)
                elif action == self.SOUTH_EAST:
                    next_position = current_position + np.array([-1,1]) + self.get_wind_effect(current_position)
                elif action == self.SOUTH:
                    next_position = current_position + np.array([-1,0]) + self.get_wind_effect(current_position)
                elif action == self.SOUTH_WEST:
                    next_position = current_position + np.array([-1,-1]) + self.get_wind_effect(current_position)
                elif action == self.WEST:
                    next_position = current_position + np.array([0,-1]) + self.get_wind_effect(current_position)
                elif action == self.NORTH_WEST:
                    next_position = current_position + np.array([1,-1]) + self.get_wind_effect(current_position)

                next_position = self.restrict_tranisition_inside_gridworld(next_position)
                next_state = np.ravel_multi_index(next_position, [self.num_rows, self.num_columns])

                self.transitions[state][action][0] = int(next_state) 

                if np.all(self.end_position == next_position): #if we are transitting to goal / end state
                    self.transitions[state][action][1] = 0 #reward
                else:
                    self.transitions[state][action][1] = -1 #reward
    
    def get_start_state(self):
        return np.ravel_multi_index(self.start_position, [self.num_rows, self.num_columns])

    def get_end_state(self):
        return np.ravel_multi_index(self.end_position, [self.num_rows, self.num_columns])
        
    def get_wind_effect(self, _position):
        return self.wind_strength[_position[1]] * np.array([1,0]) 

    def restrict_tranisition_inside_gridworld(self, _position):
        _position[0] = min(self.num_rows-1, _position[0]) #position cannot be above top most row 
        _position[0] = max(0, _position[0]) #position cannot be below bottom most row 
        _position[1] = min(self.num_columns-1, _position[1]) #position cannot be on right of rightmost column 
        _position[1] = max(0, _position[1]) #position cannot be on left of rightmost column
        return _position

class StochaticWindGridWorldWithKingsMoves:
    '''
    Stochastic-ness presents only in windy columns of grid, not in all columns.
    This differs from WindyStochaticEverywhereGridWorldWithKingsMoves where wind is present in all columns / everywhere.
    ''' 

    def __init__(self):
        self.num_rows = 7
        self.num_columns = 10
        self.num_actions = 8
        self.num_states = self.num_rows * self.num_columns

        self.start_position = (3,0)
        self.end_position = (3,7)
        
        self.wind_strength = np.array([0,0,0,1,1,1,2,2,1,0])

        self.NORTH = 0
        self.NORTH_EAST = 1
        self.EAST = 2
        self.SOUTH_EAST = 3
        self.SOUTH = 4
        self.SOUTH_WEST = 5
        self.WEST = 6
        self.NORTH_WEST = 7

        self.transitions = np.zeros((self.num_states, self.num_actions, 2), dtype=np.int) #2 for next state and reward

        self.init_transitions()

        self.name = 'Windy Gridworld with Kings moves and stochastic noise on windy columns'

        #TODO explicitly set next state and reward for final state?

    def get_next_state_reward(self, _state, _action):
        '''
        Implements stochastic behavior in windy columns.
        '''
        _state_position =  np.unravel_index(_state, (self.num_rows, self.num_columns))         
        next_state, reward = self.transitions[_state, _action] 

        if self.wind_strength[_state_position[1]] > 1: #for windy column, return next column considering stochstic behavior
            next_position = np.unravel_index(next_state, (self.num_rows, self.num_columns)) 
            next_position = list(next_position) # convert tuple to list as we cannot modify position in tuple directly
            next_position[0] = next_position[0] + np.random.randint(-1,2)
            next_position = self.restrict_tranisition_inside_gridworld(next_position)
            return np.ravel_multi_index(next_position, [self.num_rows, self.num_columns]), reward
        else: #dont apply stochasticness in non windy column
            return next_state, reward
        

    def init_transitions(self):
        '''
        Initialise transitions in each position of grid world for each action considering the 
        wind strength at that position.
        '''
        for state in range(self.num_states):
            current_position = np.unravel_index(state, (self.num_rows, self.num_columns)) #position corresponding to current state
            for action in range(self.num_actions):
                if action == self.NORTH:
                    next_position = current_position + np.array([1,0]) + self.get_wind_effect(current_position)
                elif action == self.NORTH_EAST:
                    next_position = current_position + np.array([1,1]) + self.get_wind_effect(current_position)
                elif action == self.EAST:
                    next_position = current_position + np.array([0,1]) + self.get_wind_effect(current_position)
                elif action == self.SOUTH_EAST:
                    next_position = current_position + np.array([-1,1]) + self.get_wind_effect(current_position)
                elif action == self.SOUTH:
                    next_position = current_position + np.array([-1,0]) + self.get_wind_effect(current_position)
                elif action == self.SOUTH_WEST:
                    next_position = current_position + np.array([-1,-1]) + self.get_wind_effect(current_position)
                elif action == self.WEST:
                    next_position = current_position + np.array([0,-1]) + self.get_wind_effect(current_position)
                elif action == self.NORTH_WEST:
                    next_position = current_position + np.array([1,-1]) + self.get_wind_effect(current_position)

                next_position = self.restrict_tranisition_inside_gridworld(next_position)
                next_state = np.ravel_multi_index(next_position, [self.num_rows, self.num_columns])

                self.transitions[state][action][0] = int(next_state) 

                if np.all(self.end_position == next_position): #if we are transitting to goal / end state
                    self.transitions[state][action][1] = 0 #reward
                else:
                    self.transitions[state][action][1] = -1 #reward
    
    def get_start_state(self):
        return np.ravel_multi_index(self.start_position, [self.num_rows, self.num_columns])

    def get_end_state(self):
        return np.ravel_multi_index(self.end_position, [self.num_rows, self.num_columns])
        
    def get_wind_effect(self, _position):
        return self.wind_strength[_position[1]] * np.array([1,0]) 

    def restrict_tranisition_inside_gridworld(self, _position):
        _position[0] = min(self.num_rows-1, _position[0]) #position cannot be above top most row 
        _position[0] = max(0, _position[0]) #position cannot be below bottom most row 
        _position[1] = min(self.num_columns-1, _position[1]) #position cannot be on right of rightmost column 
        _position[1] = max(0, _position[1]) #position cannot be on left of rightmost column
        return _position
