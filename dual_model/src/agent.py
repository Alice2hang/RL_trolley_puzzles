import numpy as np
from collections import Counter, defaultdict
from scipy.special import softmax

import neural_net
from utils import generate_array, display_grid

ACTION_DICT = {(0, 0):0, (-1, 0):1, (0, 1):2, (1, 0):3, (0, -1):4} 

class Agent:
    """
    Dual processing agent.
    Given a set of actions and a view of the grid, decides what action to take
    """

    def load_neural_net(self):
        """
        Attempt to load neural net from 'neural_net' file. 
        """
        try:
            return self.net
        except(AttributeError):
            try:
                net = neural_net.load()
                print("neural net loaded")
                self.net = net
                return net
            except:
                raise AttributeError("No net specified")
            
    def neural_net_output(self, grid):
        '''
        Get NN Q-value predictions for all possible actions at a given grid state
        '''
        net = self.load_neural_net()
        state_array = generate_array(grid)
        out = neural_net.predict(net, state_array)
        return out[0]

    def run_model_free_policy(self, grid, display=False):
        """
        Use neural network to solve MDP (no Monte-Carlo Exploration)
        Params:
            - grid (Grid): grid object initialized with starting state
            - display (bool): whether to display the solution
        Returns:
            - total_reward (int): reward achieved by model-free NN on input grid
        """
        if display: display_grid(grid)
        net = self.load_neural_net()
        total_reward = 0

        while not grid.terminal_state: 
            test_input = generate_array(grid)
            NN_output = neural_net.predict(net, test_input)
            action_ind = np.argmax(NN_output) #pick action with highest predicted value
            action = grid.all_actions[action_ind]
            total_reward += grid.R(action)
            grid.T(action)
            if display: 
                print(NN_output)
                print(action)
                display_grid(grid)
        return total_reward

    def _create_epsilon_greedy_policy(self, Q_dict, epsilon=0.2, nn_init = False):
        """
        Based on Q-values, returns a policy that selects the best action with probability 1-epsilon, and acts randomly with probability epsilon. 
        Params: 
            - Q_dict(defaultdict(list)): maps grid state to array of Q values for each of the 5 possible actions at that state
            - epsilon(float): exploration parameter
            - nn_init(bool): whether we want to use the neural network to determine initial Q-values
        Returns: 
            policy(function): epsilon greedy policy that returns the probability of selecting each action for a given grid
        """
        def policy(grid):
            state = grid.current_state
            if state not in Q_dict and nn_init:
                Q_dict[state] = self.neural_net_output(grid)
            Q_values = Q_dict[state]
            action_probs = [0 for k in range(len(Q_values))]
            best_actions = np.argwhere(Q_values == np.max(Q_values))
            # if multiple best actions, choose randomly between them with probability 1-epsilon
            for i in range(len(Q_values)):
                if i in best_actions:
                    action_probs[i] = (1-epsilon)/len(best_actions)+(epsilon/len(Q_values))
                else:
                    action_probs[i] = epsilon/len(Q_values)
            return action_probs
        return policy

    def _create_softmax_policy(self, Q_dict, nn_init=False):
        """
        Based on Q-values, returns a policy that selects actions based on softmax(Q-values). This means that actions with greater
        Q-value are more likely to be selected.
        Params: 
            - Q_dict(defaultdict(list)): maps grid state to array of Q values for each of the 5 possible actions at that state
            - nn_init(bool): whether we want to use the neural network to determine initial Q-values
        Returns: 
            -policy(function): softmax policy that returns the probability of selecting each action for a given grid
        """
        def policy(grid):
            state = grid.current_state
            if state not in Q_dict and nn_init:
                Q_dict[state] = self.neural_net_output(grid)
            Q_values = Q_dict[state]
            action_probs = softmax(Q_values)
            return action_probs
        return policy

    def run_final_policy(self, grid, Q_dict, nn_init=False, display=False):
        """
        Use Q_dict to solve MDP (no exploration)
        Params: 
            - grid (Grid): grid object initialized with starting state
            - Q_dict(defaultdict(list)): maps grid state to array of Q values for each of the 5 possible actions at that state
            - nn_init(bool): whether we want to use the neural network to initialise the monte carlo policy
            - display (bool): whether to display the solution
        returns: 
            - total_reward(int): final score achieved by policy
        """
        policy = self._create_epsilon_greedy_policy(Q_dict,epsilon=0,nn_init=nn_init) #optimal policy, eps=0 always chooses best value
        total_reward = 0    
        while not grid.terminal_state: # max number of steps per episode
            action_probs = policy(grid)
            action_ind = np.argmax(action_probs)
            action = grid.all_actions[action_ind]
            if display: 
                display_grid(grid)
                print(action)
            total_reward += grid.R(action)
            grid.T(action)
        if display: print(total_reward)
        return total_reward

    def mc_first_visit_control(self, start_grid, iters, discount_factor=0.9, epsilon=0.2, nn_init=False, softmax=False) -> tuple:
        """
        Monte Carlo first visit control algo. Uses epsilon greedy strategy to find optimal policy. 
        Details can be found page 101 of Sutton and Barto RL Book
        Args:
            - start_grid (Grid): grid object initialized with starting state
            - iters (int): number of iterations to run monte carlo simulation for
            - discount_factor (int): how much to discount future rewards
            - epsilon (int): how much to favor random choices in strategy
            - nn_init (bool): whether to initialize Q-values with neural net outputs
            - softmax (bool): whether to select actions to explore using a softmax or epsilon-greedy policy
        Returns: (Q_values, policy)
            - Q_values(defaultdict(list)): learned mapping of grid state to array of Q values for each of the 5 possible actions at that state
        """
        grid = start_grid.copy()

        if nn_init:
            Q = {}
        else:
            Q = defaultdict(lambda: np.zeros(5,dtype=np.float32))

        if softmax: 
            policy = self._create_softmax_policy(Q, nn_init) 
        else:
            policy = self._create_epsilon_greedy_policy(Q,epsilon, nn_init) #initial function

        sa_reward_sum, total_sa_counts = defaultdict(int), defaultdict(int) #keep track of total reward and count over all episodes
        for n in range(iters):
            # generate episode
            episode = []
            grid = start_grid.copy() # opy because running episode mutates grid object
            state = grid.current_state
            while not grid.terminal_state: # max number of steps per episode
                action_probs = policy(grid)
                action_ind = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                action = grid.all_actions[action_ind]
                reward = grid.R(action)  #must calculate reward before transitioning state, otherwise reward will be calculated for action in newstate
                newstate = grid.T(action)
                episode.append((state, action, reward))
                state = newstate
            sa_counts = Counter([(x[0],x[1]) for x in episode]) #dictionary: [s,a]=count
            G = 0 #averaged reward
            for t in range(len(episode)-1,-1,-1):
                G = discount_factor*G + episode[t][2] #reward at the next time step
                state = episode[t][0]
                action = episode[t][1]
                action_index = grid.all_actions.index(action)
                sa_pair = state, action
                sa_counts[sa_pair] -= 1
                if sa_counts[sa_pair] == 0: #appears for the first time
                    sa_reward_sum[sa_pair] += G
                    total_sa_counts[sa_pair] += 1
                    Q[state][action_index] = sa_reward_sum[sa_pair]/total_sa_counts[sa_pair] #average reward over all episodes
                    if softmax: 
                        policy = self._create_softmax_policy(Q, nn_init) 
                    else:
                        policy = self._create_epsilon_greedy_policy(Q,epsilon, nn_init) #initial function
        return Q
