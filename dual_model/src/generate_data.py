import numpy as np
import time
import random
import json
from collections import defaultdict

from grid import Grid
from value_iteration import get_true_Qs

def get_within(steps,location):
    '''
    Helper function to get all positions within n steps of a given location, including location
    '''
    within = set([location])
    pos_list = [location]
    possible_moves = (1,0),(-1,0),(0,1),(0,-1)
    for i in range(steps):
        new_pos_list = []
        for pos in pos_list:
            for move in possible_moves:
                new_pos = (pos[0]+move[0],pos[1]+move[1])
                if new_pos not in within:
                    new_pos_list.append(new_pos)
                    within.add(new_pos)
        pos_list = new_pos_list
    return within

def grid_must_push(size):
    '''
    Generates initial grid setup (positions of all key items) so that the optimal solution involves 
    pushing cargo out of the way of the train to prevent negative reward
    '''
    open_grid_coords = set((i,j) for i in range(size) for j in range(size))

    train_orientation = np.random.choice(4) #random between 0-3, 0->right,1->left,2->down,3->up
    train_loc = np.random.choice(size-2)+1 #position along starting wall
    train_map = {0:((0,1),(train_loc,0)),1:((0,-1),(train_loc,size-1)),
                 2:((1,0),(0, train_loc)),3:((-1,0),(size-1, train_loc))}
    train_vel, train_pos = train_map[train_orientation]
    train_path = set()
    for i in range(5):
        train_path.add((train_pos[0]+i*train_vel[0],train_pos[1]+i*train_vel[1]))
    open_grid_coords.remove(train_pos)

    dist = np.random.choice(3)+2
    other_1 = (train_pos[0]+train_vel[0]*dist,train_pos[1]+train_vel[1]*dist)
    open_grid_coords.remove(other_1)

    agent_pos_choices = get_within(dist-2,(other_1[0]+train_vel[1],other_1[1]+train_vel[0]))
    agent_pos_choices = agent_pos_choices.union(get_within(dist-2,(other_1[0]-train_vel[1],other_1[1]-train_vel[0])))
    agent_pos_choices = open_grid_coords.intersection(agent_pos_choices)
    agent_pos_choices -= set([(2,2)])
    agent_pos = random.sample(agent_pos_choices,1)[0]
    open_grid_coords.remove(agent_pos)

    #rest of vars cannot be on train path
    open_grid_coords -= train_path

    unreachable_pos_choices = open_grid_coords-get_within(dist,agent_pos)
    other_2,switch_pos = random.sample(unreachable_pos_choices, 2)
    open_grid_coords.remove(other_2)
    open_grid_coords.remove(switch_pos)

    target_1,target_2 = random.sample(open_grid_coords, 2)

    if np.random.choice(2) == 1:
        other_1, other_2 = other_2, other_1
        target_1, target_2 = target_2, target_1

    return {'train':train_pos,'trainvel':train_vel,'cargo1':other_1,'num1':1,'target1':target_1,
            'switch':switch_pos,'agent':agent_pos,'cargo2':other_2,'num2':2,'target2':target_2}

def grid_must_switch(size):
    '''
    Generates initial grid setup (positions of all key items) so that the optimal solution involves 
    hitting the switch to prevent the train from colliding with cargo. 
    '''

    open_grid_coords = set((i,j) for i in range(size) for j in range(size))

    train_orientation = np.random.choice(4) #random between 0-3, 0->right,1->left,2->down,3->up
    train_loc = np.random.choice(size) #position along starting wall
    train_map = {0:((0,1),(train_loc,0)),1:((0,-1),(train_loc,size-1)),
                 2:((1,0),(0, train_loc)),3:((-1,0),(size-1, train_loc))}
    train_vel, train_pos = train_map[train_orientation]

    train_path = set()
    for i in range(5):
        train_path.add((train_pos[0]+i*train_vel[0],train_pos[1]+i*train_vel[1]))
    open_grid_coords.remove(train_pos)

    dist = np.random.choice(4)+1
    other_1 = (train_pos[0]+train_vel[0]*dist,train_pos[1]+train_vel[1]*dist)
    open_grid_coords.remove(other_1)

    open_grid_coords -= train_path
    switch_pos = random.sample(open_grid_coords,1)[0]
    open_grid_coords.remove(switch_pos)

    agent_pos_choices = open_grid_coords.intersection(get_within(dist,switch_pos))
    agent_pos_choices -= set([(2,2)])
    agent_pos = random.sample(agent_pos_choices,1)[0]
    open_grid_coords.remove(agent_pos)

    target_1, target_2, other_2 = random.sample(open_grid_coords, 3)

    if np.random.choice(2) == 1:
        other_1, other_2 = other_2, other_1
        target_1, target_2 = target_2, target_1

    return {'train':train_pos,'trainvel':train_vel,'cargo1':other_1,'num1':1,'target1':target_1,
            'switch':switch_pos,'agent':agent_pos,'cargo2':other_2,'num2':2,'target2':target_2}

def grid_get_targets(size):
    '''
    Generates initial grid setup (positions of all key items) so that the optimal solution involves 
    pushing cargo into its target location. 
    '''
    open_grid_coords = set((i,j) for i in range(size) for j in range(size))

    train_orientation = np.random.choice(4) #random between 0-3, 0->right,1->left,2->down,3->up
    train_loc = np.random.choice(size) #position along starting wall
    train_map = {0:((0,1),(train_loc,0)),1:((0,-1),(train_loc,size-1)),
                 2:((1,0),(0, train_loc)),3:((-1,0),(size-1, train_loc))}
    train_vel, train_pos = train_map[train_orientation]
    train_path = set()
    for i in range(5):
        train_path.add((train_pos[0]+i*train_vel[0],train_pos[1]+i*train_vel[1]))

    open_grid_coords.remove(train_pos)
    open_grid_coords -= train_path

    target_1 = random.sample(open_grid_coords,1)[0]
    open_grid_coords.remove(target_1)

    corner_coords = set([(0,0),(0,4),(4,0),(4,4)])
    other_1_choices = open_grid_coords.intersection(get_within(2,target_1))-corner_coords
    other_1 = random.sample(other_1_choices,1)[0]
    open_grid_coords.remove(other_1)

    move_vector = [target_1[0]-other_1[0],target_1[1]-other_1[1]]
    if move_vector[0]>0: move_vector[0] = 1
    if move_vector[0]<0: move_vector[0] = -1
    if move_vector[1]>0: move_vector[1] = 1
    if move_vector[1]<0: move_vector[1] = -1

    agent_pos_choices = get_within(2,(other_1[0]-move_vector[0],other_1[1])).union(get_within(2,(other_1[0],other_1[1]-move_vector[1])))
    agent_pos_choices = open_grid_coords.intersection(agent_pos_choices)
    if len(agent_pos_choices) == 0:
        return False
    agent_pos = random.sample(agent_pos_choices,1)[0]
    open_grid_coords.remove(agent_pos)

    open_grid_coords -= get_within(2,other_1)
    other_2, target_2, switch_pos = random.sample(open_grid_coords,3)

    if np.random.choice(2) == 1:
        other_1, other_2 = other_2, other_1
        target_1, target_2 = target_2, target_1

    return {'train':train_pos,'trainvel':train_vel,'cargo1':other_1,'num1':1,'target1':target_1,
            'switch':switch_pos,'agent':agent_pos,'cargo2':other_2,'num2':2,'target2':target_2}


def grid_nothing_lose(size):
    '''
    Generates initial grid setup (positions of all key items) so that the train will collide with
    cargo no matter what the agent does. 
    '''
    open_grid_coords = set((i,j) for i in range(size) for j in range(size))

    train_orientation = np.random.choice(4) #random between 0-3, 0->right,1->left,2->down,3->up
    train_loc = np.random.choice(size) #position along starting wall
    train_map = {0:((0,1),(train_loc,0)),1:((0,-1),(train_loc,size-1)),
                 2:((1,0),(0, train_loc)),3:((-1,0),(size-1, train_loc))}
    train_vel, train_pos = train_map[train_orientation]
    open_grid_coords.remove(train_pos)

    dist = np.random.choice(3)+1
    other_1 = (train_pos[0]+train_vel[0]*dist,train_pos[1]+train_vel[1]*dist)
    open_grid_coords.remove(other_1)

    agent_pos_choices = open_grid_coords - get_within(dist,other_1)
    agent_pos = random.sample(agent_pos_choices,1)[0]
    open_grid_coords.remove(agent_pos)

    switch_pos_choices = open_grid_coords-get_within(dist,agent_pos)
    switch_pos = random.sample(switch_pos_choices, 1)[0]
    open_grid_coords.remove(switch_pos)

    target_1,target_2,other_2 = random.sample(open_grid_coords, 3)

    if np.random.choice(2) == 1:
        other_1, other_2 = other_2, other_1
        target_1, target_2 = target_2, target_1

    return {'train':train_pos,'trainvel':train_vel,'cargo1':other_1,'num1':1,'target1':target_1,
            'switch':switch_pos,'agent':agent_pos,'cargo2':other_2,'num2':2,'target2':target_2}


def collect_grid(size, grid_type):
    """
    Generates random grid of grid_type and uses value-iteration to compute the series of grid states 
    and actions in the solution. Ensures that the final reward is a valid value for
    the given grid type.
    Params: 
        - size(int): dimension of square grid
        - grid_type(str): one of four predefined types: 'push', 'switch', 'targets', or 'lose'
    Returns: (grids, action_values, reward, init_pos)
        - grids(np.ndarray): (n, 3, size, size) 
        - action_values(): (n,5) generated
        - reward(int): reward achieved by Q-value-iteration solver
        - init_pos(dict): init info for randomly generated grid of grid_type, for generating grids for web experiment
    """

    # maps grid type string to generative function
    func_dict = {'push': grid_must_push,'switch':grid_must_switch,'targets':grid_get_targets,'lose':grid_nothing_lose}

    # Filters valid reward outcomes for each grid type
    #PUSH includes save or save and put cargo into target scenarios (any non-negative reward)
    #SWITCH includes same as push, 1 is excluded because it may reflect a 'switch-sacrifice' scenario 
    #TARGETS includes 1,2,3 for successfully getting any combo of boxes into targets
    #LOSE includes all possible negative values, as well as 1 if value 1 cargo is hit and value 2 cargo is pushed onto target
    valid_dict = {'push':[0,1,2],'switch':[0,2],'targets':[1,2,3],'lose':[-2,-1,1]}
    valid_rewards = valid_dict[grid_type]
    reward = -100 #invalid initialization
    while reward not in valid_rewards:
        init_pos = func_dict[grid_type](size)
        while init_pos == False:   #if grid_gets_targets fails to generate a initial config, try again
            init_pos = grid_get_targets(size)
        testgrid = Grid(init_pos=init_pos)
        grids, action_values, reward = get_true_Qs(testgrid.copy())
    return grids, action_values, reward, init_pos


def data_gen(num_grids=1000, grid_size=5, distribution=None, save=True, filename=""):
    """
    Saves 2 ndarrays, actions_val_array (n,5) and grids_array (n,2, grid_size, grid_size), where the second dim is for future train pos
    Each grid generates 5 data points (one for each timestep)

    Files should appear as "filename_grids_data.npy" and "filename_actions_data.npy" in the same directory as this script
    Params:
        - num_grids(int): number of grid examples to generate data for
        - grid_size(int): dimensions of square grid
        - distribution(dict): mapping of grid_type string to percentage value (should sum to 100)
        - save(bool): whether to save output to file
        - filename(str): filename to save actions and grids to
    Returns:
        - user_testing_grids(list): list of grid initialization dictionaries, to be used for web experiment
    """
    start = time.time()
    print("Started data generation")
    grids_data = np.empty((1,4,grid_size,grid_size),dtype=int)
    actions_data = np.empty((1, grid_size),dtype=int)
    reward_dist = defaultdict(int)

    if distribution == None:
        distribution = {'push':25,'switch':25,'targets':25,'lose':25}
    user_testing_grids = [] #accumulates grid objects, to have playable ui

    count = 0
    for type in distribution:
        num_type = int(distribution[type]*num_grids/100)
        for i in range(num_type):
            grids,actions,reward,init_info = collect_grid(grid_size,type)
            init_info = coords_for_web(init_info)
            user_testing_grids.append((init_info,reward))
            reward_dist[reward] += 1
            actions_data = np.concatenate((actions_data,actions))
            grids_data = np.vstack((grids_data,grids))
            count += 1
            if count % 100 == 0:
                print("generated grid",count)
    if save:
        shuffle_order = np.random.permutation(len(actions_data[1:]))   # Randomize order of grid-action pairs for downstream training/test
        np.save("grids_"+filename,grids_data[1:][shuffle_order])
        np.save("actions_"+filename,actions_data[1:][shuffle_order])
    print("finished in", time.time()-start)
    print("reward_dist: ", reward_dist)

    return user_testing_grids

def coords_for_web(init_info):
    '''
    Converts initialization dictionary to format for web experiment
    '''
    new_init = {}
    for key,value in init_info.items():
        if key not in ('num1','num2','trainvel'):
            new_init[key] = (value[1],4-value[0])
        elif key == 'trainvel':
            new_init[key] = (value[1],-value[0])
        else:
            new_init[key] = value
    return new_init

def make_train_json(num):
    '''
    Generate training grids from a fixed distribution for web experiment and write to train_data.json
    '''
    grids = data_gen(num, distribution={'push':25,'switch':25,'targets':40,'lose':10}, save=False)
    random.shuffle(grids)
    data = {}
    for idx,sample in enumerate(grids):
        init_pos = sample[0]
        del init_pos['num1']
        del init_pos['num2']
        init_pos['best_reward'] = sample[1]
        data[idx] = init_pos
    with open('train_data.json', 'w') as outfile:
        json.dump(data, outfile)


if __name__ == "__main__":
    # Generate 200,000 initial grids (each with 5 grid action pairs) for training CNN, takes ~1/2 day
    CNN_training_distrib = {'push':23,'switch':23,'targets':39,'lose':15}
    data_gen(200000, distribution = CNN_training_distrib, save = True,filename = "../training_data/200000")
