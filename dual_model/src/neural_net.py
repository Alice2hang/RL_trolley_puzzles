import numpy as np
import time, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

cuda = True if torch.cuda.is_available() else False
# Total number of input channels 
# overall_pos_mask, agent, train, next_train, switch, obj1, target1, obj2, target2, timestep
CHANNELS = 10 

base_path = os.path.dirname(os.path.dirname(__file__))
NN_FILE = os.path.join(base_path,'models/nn_model')

class Net(nn.Module):
    '''
    Convolutional neural network takes in grid state (4,5,5) and predicts Q-values for
    all actions at that state (5)
    '''
    def __init__(self, C=CHANNELS):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(C, 100, 3, padding=1) 
        self.conv2 = nn.Conv2d(100, 100, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(100, 100, 3, padding=1)
        self.conv4 = nn.Conv2d(100, 100, 3, padding=1)
        self.fc1 = nn.Linear(100 * 1 * 1, 64)
        self.fc2 = nn.Linear(64, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 100 * 1 * 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def make_onehot_data(inputs, labels):
    """
    Preprocesses training data to one hot format for downstream model training/validation 
    Params:
        - inputs: Nx4x5x5 numpy array representing N input grid states
        - labels: Nx5 numpy array representing N training output sets of action Q values
    """
    inputs = torch.from_numpy(inputs).float()
    labels = torch.from_numpy(labels).float()

    base = inputs[:, 0:1, :, :] #this contains all elements except next train, targets, and timestep
    targets = inputs[:, 2:3, :, :]
    timestep = inputs[:, 3:4, :, :]
    
    mask_layer = (base != 0).float() # represents any space that is occupied
    agent_layer = (base == 1).float()
    train_layer = (base == 2).float()
    next_train_layer = inputs[:, 1:2, :, :]
    switch_layer = (base == 3).float()
    object1_layer = (base == 4).float()
    target1_layer = (targets == 1).float()
    object2_layer = (base == 5).float()
    target2_layer = (targets == 2).float()

    onehot_inputs = torch.cat((mask_layer, agent_layer, train_layer, next_train_layer,switch_layer,object1_layer,target1_layer,object2_layer, target2_layer, timestep), dim=1)

    if cuda:
        onehot_inputs = onehot_inputs.cuda()
        labels = labels.cuda()

    B = inputs.shape[0]
    onehot_train_inputs = onehot_inputs[:9*B//10]
    train_labels = labels[:9*B//10]

    onehot_test_inputs = onehot_inputs[9*B//10:]
    test_labels = labels[9*B//10:]

    return onehot_train_inputs, onehot_test_inputs, train_labels, test_labels


def train(grids_file, actions_file, num_epochs=100, C=CHANNELS):
    '''
    Params:
        - grids_file(str): filepath for training grids data .npy generated by generate_data.py
        - actions_file(str): filepath for training actions data .npy generated by generate_data.py
        - num_epochs(int): how many epochs to train for
        - C(int): number of channels for input data
    '''
    xs = np.load(grids_file)['data']
    ys = np.load(actions_file)['data']
    onehot_train_xs, onehot_test_xs, train_ys, test_ys = make_onehot_data(xs, ys)

    net = Net(C)
    criterion = nn.MSELoss()#CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001) #original lr 0.001
    batch_size = 1000

    if cuda:
        net = net.cuda()

    start = time.time()
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = []

        #trains on the remainder if not fully divisible
        for j in range((len(train_ys)-1)//batch_size+1):
            inputs, labels = onehot_train_xs[batch_size*j:batch_size*(j+1)], train_ys[batch_size*j:batch_size*(j+1)]
            optimizer.zero_grad()  # zero the parameter gradients
            outputs = net(inputs)  # forward, backward, optimize
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss.append(loss.item())
        if epoch%10==0 or (epoch == num_epochs-1):
            # Print running statistics: train loss, test loss, and accuracy
            test_loss = []
            test_accuracy = []
            with torch.no_grad():
                for j in range((len(test_ys)-1)//batch_size+1):
                    inputs, labels = onehot_test_xs[batch_size*j:batch_size*(j+1)], test_ys[batch_size*j:batch_size*(j+1)]
                    outputs = net(inputs)
                    label_argmax = torch.argmax(labels, dim=1)
                    output_argmax = torch.argmax(outputs, dim=1)
                    accuracy = torch.mean((label_argmax==output_argmax).float())  #Accuracy is not fully reflective of performance, as several actions may have exact same Q-value
                    test_accuracy.append(accuracy.item())
                    loss = criterion(outputs, labels)
                    test_loss.append(loss.item())
            print('Epoch:{}, train loss: {:.3f}, test loss: {:.3f}, test accuracy: {:.3f}'.format(epoch + 1, np.mean(running_loss), np.mean(test_loss), np.mean(test_accuracy)))
            running_loss = 0.0
    print("training took", time.time() - start, "seconds")
    torch.save(net.state_dict(), "../models/nn_model")
    print("Model saved as nn_model")

def load(C=CHANNELS):
    '''
    attempt to load neural net from file
    '''
    model = Net(C)
    if not cuda:
        model.load_state_dict(torch.load(NN_FILE ,map_location='cpu'))
    else:
        model.load_state_dict(torch.load(NN_FILE))
    return model

def predict(model, state):
    '''
    Params:
        - model: pytorch model, output of load()
        - state: 4x5x5 numpy array corresponding to grid state input 
    Returns:
        - outputs: 1x5(num actions) numpy array of predicted Q-values for each action at the given state
    '''

    inputs = torch.from_numpy(state).float()

    base = inputs[0:1, :, :] 
    targets = inputs[2:3, :, :]
    timestep = inputs[3:4, :, :]

    mask_layer = (base != 0).float()
    agent_layer = (base == 1).float()
    train_layer = (base == 2).float()
    next_train_layer = inputs[1:2, :, :]
    switch_layer = (base == 3).float()
    object1_layer = (base == 4).float()
    target1_layer = (targets == 1).float()
    object2_layer = (base == 5).float()
    target2_layer = (targets == 2).float()

    onehot_inputs = torch.cat((mask_layer, agent_layer, train_layer, next_train_layer,switch_layer,object1_layer,target1_layer,object2_layer, target2_layer,timestep), dim=0)
    onehot_inputs = torch.unsqueeze(onehot_inputs, dim=0)
    outputs = model(onehot_inputs)

    return outputs.detach().numpy()

if __name__ == "__main__":
    train("../training_data/grids_200000.npz","../training_data/actions_200000.npz",100)