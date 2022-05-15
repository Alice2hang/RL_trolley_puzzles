from utils import Train, OtherMask, Switch, in_bounds

class Grid:
    '''
    Grid is the class for the MDP that defines transition and reward functions, allowing for agent interaction
    Keeps track of all aspects of grid state, including agent, cargo, train, and target positions, step, etc
    PLEASE NOTE: positions are (y,x) with (0,0) in the top left corner of the grid
    Params:
        - init_pos(dict): initial positions for the elements in the grid. Should be in this format: 
            {'train':(1,1),'agent':(2,2),'other1':(2,1),'switch':(4,1),'other2':(2,0),'other1num':3,...}
    '''

    def __init__(self, init_pos={}):
        # available actions: stay, north, east, south, west
        self.all_actions =[(0, 0), (-1, 0), (0, 1), (1, 0), (0, -1)]

        self.size = 5
        self.terminal_state = False
        self.step = 1
        self.alive = True
        self._place_all(init_pos)
        self.current_state = (self.agent_pos,self.train.pos,*self.other_agents.positions,self.step)
        self.rewards_dict = {'agent hit by train': -4, 'agent pushes others':0,
                            'others hit by train':-1, 'agent push switch': 0,
                            'others on target': 1, 'do nothing':0}

    def copy(self):
        """
        Returns a deep copy of grid - because grid is mutated in learning (agent performs transition and
        observes result), copying is necessary to maintain original state
        Returns:
            - copy(Grid): identical grid object
        """
        copy = Grid()
        copy.train = self.train.copy()
        copy.other_agents = self.other_agents.copy()
        copy.switch = self.switch
        copy.agent_pos = self.agent_pos
        copy.current_state = self.current_state
        return copy

    def _place_all(self, init_pos) -> None:
        """
        Initializes all positions and train velocity based on init_pos, or defaults to values set in utils.py
        for each object type.
        Params:
            - init_pos(dict): defined starting configuration  
        """
        if init_pos:
            init_pos = init_pos.copy()
            for key,val in init_pos.items():
                if key == "trainvel":
                    init_pos[key] = (-val[1],val[0])
                elif type(val) == tuple :
                    init_pos[key] = (4-val[1],val[0])
            self.train = Train(self.size,pos=init_pos['train'],velocity=init_pos['trainvel'])
            self.agent_pos = init_pos['agent']
            self.switch = Switch(self.size,pos=init_pos['switch'])
            others_pos = [init_pos['cargo1'],]
            targets = [init_pos['target1']]
            num = [init_pos['num1']]
            if 'num2' in init_pos:
                others_pos.append(init_pos['cargo2'])
                num.append(init_pos['num2'])
                targets.append(init_pos['target2'])
            self.other_agents = OtherMask(positions=others_pos, num=num, targets=targets)
        else:
            self.train = Train(self.size)
            self.other_agents = OtherMask()
            self.switch = Switch(self.size)
            self.agent_pos = (0,2)
       

    def legal_actions(self) -> set:
        """
        Agent can not move outside of the pre-defined grid boundaries
        Returns:
            - legal_actions(set): set of tuples representing legal actions from the current state
        """
        legal_actions = self.all_actions.copy()
        for action in self.all_actions:
            new_position_y = self.agent_pos[0]+action[0]
            new_position_x = self.agent_pos[1]+action[1]
            if not in_bounds(self.size,(new_position_y,new_position_x)):
                legal_actions.remove(action)
        return legal_actions


    def T(self, action:tuple):
        """
        Precondition: action needs to be legal, board cannot be in terminal state         
        can be changed return duplicate grid object if mutation is bad
        Params: 
            - action(tuple): Agent action that causes state transition
        Returns:
            - next_state(tuple): state after transition is completed
        """

        #check not terminal state
        if self.terminal_state:
            return self.agent_pos, self.train.pos

        if not self.alive:
            action = (0,0)

        new_x = self.agent_pos[0] + action[0]
        new_y = self.agent_pos[1] + action[1]
        new_agent_pos = (new_x,new_y)

        #check that action is legal
        if action not in self.legal_actions():
            new_agent_pos = self.agent_pos

        #check if switch is pushed
        if new_agent_pos == self.switch.pos:
            new_agent_pos = self.agent_pos #agent
            if self.train.velocity[1] == 0:
                self.train.velocity = (0, self.train.velocity[0]-self.train.velocity[1])
            else:
                self.train.velocity = (self.train.velocity[0]-self.train.velocity[1],0)
        old_train_pos = self.train.pos
        self.train.update() #update train AFTER switch is hit

        #episode ends if train leaves screen or collides
        if self.step==5:
            self.terminal_state = True

        if new_agent_pos in self.other_agents.positions:
            #agent intersect cargo: push
            #moves both agent and cargo given that it will not push anyone out of bounds
            new_other_y = new_agent_pos[0] + action[0]
            new_other_x = new_agent_pos[1] + action[1]
            new_other_pos = (new_other_y,new_other_x)
            #no pushing allowed if crash has already occured
            train_stopped = new_agent_pos == self.train.pos and self.train.velocity == (0,0)
            #no pushing if another object or switch is in next location
            pos_open = new_other_pos not in self.other_agents.positions and new_other_pos != self.switch.pos
            if in_bounds(self.size,new_other_pos) and pos_open and not train_stopped:
                self.other_agents.push(new_agent_pos,action)
            else:
                new_agent_pos = self.agent_pos

        #collision detect
        if (self.agent_pos == self.train.pos) and (new_agent_pos == old_train_pos):
            #agent should not be able to pass through train
            new_agent_pos = self.agent_pos

        if new_agent_pos == self.train.pos and self.train.velocity != (0,0):
            #agent intersect train: death, terminal state
            self.train.velocity = (0,0)
            self.alive = False

        if self.train.pos in self.other_agents.positions:
            #other intersect train: death, terminal state
            self.train.velocity = (0,0)

        for pos in self.other_agents.positions:
            other = self.other_agents.mask[pos]
            train_hit = pos == self.train.pos and self.train.velocity == (0,0)
            if other.active:
                if pos != other.get_target():
                    other.toggle_active()
            else:
                if pos == other.get_target() and not train_hit:
                    other.toggle_active()

        self.agent_pos = new_agent_pos
        self.step += 1
        self.current_state = (self.agent_pos,self.train.pos,*self.other_agents.positions,self.step)
        return self.current_state

    def R(self, action:tuple) -> int:
        """
        Reward function follows same logic as transition function without mutating grid state. Assigns value for 
        key events of interest defined in self.rewards_dict triggered by action in the current state.
        Params: 
            - action(tuple): Agent action that generates reward
        Returns:
            - reward(int): Reward resulting from given action
        """
        reward = 0

        if self.terminal_state:
            return reward

        if not self.alive:
            action = (0,0)

        #check that action is legal
        new_x = self.agent_pos[0] + action[0]
        new_y = self.agent_pos[1] + action[1]
        new_agent_pos = (new_x,new_y)
        if action not in self.legal_actions():
            new_agent_pos = self.agent_pos

        new_train_pos = self.train.get_next_position(self.train.velocity)
        train_active = self.train.velocity != (0,0)

        if new_agent_pos == self.switch.pos:
            reward += self.rewards_dict['agent push switch']
            new_agent_pos = self.agent_pos
            if self.train.velocity[1] == 0:
                new_train_pos = self.train.get_next_position((0,self.train.velocity[0]-self.train.velocity[1]))
            else:
                new_train_pos = self.train.get_next_position((self.train.velocity[0]-self.train.velocity[1], 0))

        new_agent_mask = {}
        for other_pos in self.other_agents.mask.keys():
            if new_agent_pos == other_pos:
                reward += self.rewards_dict['agent pushes others']
                new_other_y = new_agent_pos[0] + action[0]
                new_other_x = new_agent_pos[1] + action[1]
                new_other_pos = (new_other_y,new_other_x)
                train_stopped = new_agent_pos == self.train.pos and self.train.velocity == (0,0)
                pos_open = new_other_pos not in self.other_agents.positions and new_other_pos != self.switch.pos
                if not in_bounds(self.size,new_other_pos) or not pos_open or train_stopped:
                    new_other_pos = other_pos
                    new_agent_pos = self.agent_pos
                new_agent_mask[new_other_pos] = self.other_agents.mask[other_pos].copy()
            else:
                new_agent_mask[other_pos] = self.other_agents.mask[other_pos].copy()

        if (self.agent_pos == new_train_pos) and (new_agent_pos == self.train.pos):
            #agent should not be able to pass through train
            new_agent_pos = self.agent_pos

        if new_agent_pos == new_train_pos and self.train.velocity != (0,0):
            #agent intersect train: death, terminal state
            reward += self.rewards_dict['agent hit by train']

        #after pushing logic, look at location for train and target collisions
        for pos in new_agent_mask.keys():
            other = new_agent_mask[pos]

            #other intersect train: death, terminal state
            if pos == new_train_pos and train_active:
                reward += self.rewards_dict['others hit by train'] * other.get_num()

            #no points for being in target if hit by train
            train_hit = pos == self.train.pos and self.train.velocity == (0,0)

            if other.active:
                if pos != other.get_target():
                    reward -= self.rewards_dict['others on target'] * other.get_num()
            else:
                if pos == other.get_target() and not train_hit:
                    reward += self.rewards_dict['others on target'] * other.get_num()

        return reward