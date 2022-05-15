import numpy as np
import string

# Mapping of elements in array-representation of grid state
ELEMENT_INT_DICT = {'agent':1,'train':2,'switch':3,'other1':4,'other2':5}

def display_grid(mdp):
    """
    Takes in a Grid mdp state and prints a visual representation with main agent(◉), cargo(numbers), switch(S) and train (V,<,>,^),
    and targets ('a' for cargo 1, 'b' for cargo 2)
    """
    dims = (mdp.size,mdp.size) 
    grid = np.full(dims, "_", dtype=str) #np has nice display built in
    others_dict = mdp.other_agents.get_mask()
    others_list = sorted(list(others_dict.values()), key = lambda x: x.num) #sorted lowest num first

    target_dict = {}
    for ind, other in enumerate(others_list):
        target_dict[other.num] = string.ascii_lowercase[ind]

    velocity_dict = {(1,0):'v',(0,1):'>',(-1,0):'^',(0,-1):'<'}

    grid[mdp.switch.pos[0], mdp.switch.pos[1]] = "S"

    for other in others_dict:
        num = others_dict[other].get_num()
        target_pos = others_dict[other].get_target()
        grid[target_pos[0],target_pos[1]] = target_dict[num]
        grid[other[0],other[1]] = str(num)

    grid[mdp.agent_pos[0],mdp.agent_pos[1]] = "◉" 
    if mdp.train.on_screen == True:
        # if agent is hit by train, X marks collision
        if mdp.train.pos == mdp.agent_pos:
            grid[mdp.train.pos[0],mdp.train.pos[1]] = "X"
        # if cargo is hit by train, X marks collision
        elif set(others_dict.keys()).intersection({mdp.train.pos}):
            grid[mdp.train.pos[0],mdp.train.pos[1]] = "x"
        else:
            train_velocity = mdp.train.velocity
            grid[mdp.train.pos[0],mdp.train.pos[1]] = velocity_dict[train_velocity]
    print(grid)
    print("   ----------------\n")  # divider between next step

def in_bounds(size,position:tuple) -> bool:
    """
    Given a positon (y,x), checks that it is within bounds for board of given size
    """
    if 0 <= position[0] < size and 0 <= position[1] < size:
        return True
    else:
        return False

def generate_array(mdp):
    """
    Takes in a Grid mdp state and generates a numpy array (4,size,size) 
        Layer 1: positions of  agent, train, switch, objs. 
        Layer 2: projected location of train in next step (represents current direction), 
        Layer 3: target locations
        Layer 4: time step
    Used to format grid for input to the CNN for predictions/training.
    Params:
        mdp (Grid): grid at current state in time
    """

    dims = (4,mdp.size,mdp.size)
    grid = np.full(dims, 0, dtype=int) #np has nice display built in
    others_dict = mdp.other_agents.mask
    for other_coord, other_obj in others_dict.items():
        target_coord = other_obj.target
        if other_obj.num == 1:
            grid[0,other_coord[0],other_coord[1]] = ELEMENT_INT_DICT['other1']
            grid[2, target_coord[0], target_coord[1]] = 1
        elif other_obj.num == 2:
            grid[0,other_coord[0],other_coord[1]] = ELEMENT_INT_DICT['other2']
            grid[2, target_coord[0], target_coord[1]] = 2

    grid[0,mdp.agent_pos[0],mdp.agent_pos[1]] = ELEMENT_INT_DICT['agent']
    grid[3, 0, mdp.step-1] = 1 # add index of 1 to indicate time step to model
    grid[0,mdp.switch.pos[0], mdp.switch.pos[1]] = ELEMENT_INT_DICT['switch']
    next_train_y = mdp.train.pos[0]+mdp.train.velocity[0]
    next_train_x = mdp.train.pos[1]+mdp.train.velocity[1]

    if mdp.train.on_screen == True:
        grid[0,mdp.train.pos[0],mdp.train.pos[1]] = ELEMENT_INT_DICT['train']

    if in_bounds(5,(next_train_y,next_train_x)):
        grid[1, next_train_y, next_train_x] = 1

    return grid

class OtherMask:
    """
    Represents pieces of cargo and their target locations in the Grid MDP including their position and value
    """

    def __init__(self, positions=[(1,3)], num=[1], init={}, targets=[(1,4)]):
        """
        """
        self.mask = {}
        self.init = init
        self.targets = []
        self.positions = []

        if len(init) > 0:
            self.mask = init
            self.positions = list(self.mask.keys())
            for pos in self.mask:
                self.targets.append(self.mask[pos].get_target())
        else:
            self.positions = positions
            for idx,pos in enumerate(positions):
                self.mask[pos] = Other(num[idx],targets[idx],targets[idx]==pos)
            self.targets = targets

    def push(self, position, action):
        other_pushed = self.mask.pop(position,None)
        new_pos = (position[0] + action[0], position[1] + action[1])
        self.mask[new_pos] = other_pushed
        old_pos_index = self.positions.index(position)
        self.positions[old_pos_index] = new_pos

    def copy(self):
        new_init = {}
        for pos in self.mask:
            new_init[pos] = self.mask[pos].copy()
        othermask = OtherMask(init=new_init)
        return othermask

    def get_mask(self):
        return self.mask

class Other:
    """
    Represents pieces of cargo in the Grid MDP 
    """
    def __init__(self,num,target,active):
        self.target = target
        self.num = num
        self.active = active
    def copy(self):
        return Other(self.num,self.target,self.active)
    def get_num(self):
        return self.num
    def get_target(self):
        return self.target
    def toggle_active(self):
        self.active = not self.active

class Switch:
    """
    Represents a switch object that can change the direction of the train within the Grid MDP
    """
    def __init__(self, size, pos=(0,4)):
        self.size = size
        self.pos = pos
        self.activated = False
    def copy(self):
        return Switch(self.size, self.pos)

class Train:
    """
    Represents a train within the Grid MDP and its properties
    """
    def __init__(self, size, pos=(1,0),velocity=(0,1)):
        self.pos = pos
        self.velocity = velocity
        self.size = size
        self.on_screen = True
    def update(self):
        newx = self.pos[0]+self.velocity[0]
        newy = self.pos[1]+self.velocity[1]
        self.pos = (newx,newy)
        if not in_bounds(self.size,self.pos):
            self.on_screen = False
        else:
            self.on_screen = True
    def get_next_position(self,velocity):
        newx = self.pos[0]+velocity[0]
        newy = self.pos[1]+velocity[1]
        return (newx,newy)
    def copy(self):
        return Train(self.size, self.pos,self.velocity)
