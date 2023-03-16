from collections import defaultdict
import numpy as np
from itertools import product
from utils import display_grid, generate_array

def get_true_Qs(input_grid, discount_factor=0.9, display=False, get_is_action_of_interest=False):
    '''
    Returns 'solved' input grid for downstream CNN training. 
    
    First, computes discounted Q(s,a) values for all possible (state, action) pairs using value iteration. 
    Returns state representations and Q values for only the series of 5 states that represent the best solution to 
    the grid. 

        Params:
            input_grid (Grid): Grid object to be solved
            discount_factor (float): for computing time-discounted Q values 
            display (bool): whether to output solution
            get_is_action_of_interest(bool): Whether or not to return is_action_of_interest
        Returns: (grids_array, actions_array, reward)
            grids_array ((5,4,5,5) np.array): a series of 5 np array state representations for the best solution 
            actions_array ((5,5) np.array): Q values for all 5 possible actions at each state in grids_array
            reward (int): reward value achieved by best solution
            is_action_of_interest(bool, optional): True grid transition involved a cargo push or switch action, false otherwise
    '''

    Q = defaultdict(lambda: list(0 for i in range(len(grid.all_actions))))
    reward_dict = defaultdict(lambda: list(None for i in range(len(grid.all_actions))))
    next_state_dict = defaultdict(lambda: defaultdict(int)) #maps each state to a dict of action:nextstate 
    delta = 100 #some nonzero number

    # Tries all 5^5 possible series of actions and records reward achieved
    action_inds = [0,1,2,3,4]
    for combo in product(action_inds, repeat=5):
        grid = input_grid.copy()  
        step = 0
        state = grid.current_state
        while not grid.terminal_state: 
            action = grid.all_actions[combo[step]]
            reward = grid.R(action)
            newstate = grid.T(action)
            reward_dict[state][combo[step]] = reward
            next_state_dict[state][combo[step]] = newstate
            state = newstate
            step += 1

    #algo for value iteration for Q values
    while delta > 0:
        delta = 0
        for state in reward_dict.keys():
            for a in action_inds:
                tmp = Q[state][a]
                next_state = next_state_dict[state][a]
                max_next_state_q = max(Q[next_state])
                Q[state][a] = reward_dict[state][a] + discount_factor*max_next_state_q
                delta = max(delta,abs(Q[state][a]-tmp))
    
    #run optimal policy based on max Q values and generate output for NN training
    total_reward = 0 
    grids_array = []
    is_action_of_interest_array = []
    action_val_array = np.empty((1,grid.size),dtype=int)
    grid = input_grid.copy()  
    while not grid.terminal_state: 
        state = grid.current_state
        action_ind = np.argmax(Q[state])
        action = grid.all_actions[action_ind]
        if display: 
            display_grid(grid) 
            print(action)
        action_val_array = np.concatenate((action_val_array,np.array([Q[state]])))
        grids_array.append(generate_array(grid))
        total_reward += grid.R(action)
        newstate, is_action_of_interest = grid.T(action, get_is_action_of_interest = True)
        is_action_of_interest_array. append(is_action_of_interest)
        state = newstate
    grids_array = np.array(grids_array)
    if display: print(total_reward)
    if get_is_action_of_interest:
        return grids_array, action_val_array[1:], total_reward, is_action_of_interest_array
    return grids_array, action_val_array[1:], total_reward

