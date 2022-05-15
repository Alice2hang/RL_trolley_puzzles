# Dual-Agent Code and Analysis
* `grid.py` defines the gridworld MDP including transition and reward functions.
* `utils.py` includes helper functions for `grid.py`, as well as graphics and converting state representations to an array for downstream neural net processing.
* `value_iteration.py` helper function for `generate_data.py` that solves a given input grid using value iteration.
* `generate_data.py` generates data for neural network training and for experimental component.  
* `neural_net.py` defines neural net layers, training, and prediction.
* `agent.py` implements monte carlo tree search to solve gridworld problems and allows for neural network initialization of Q-values.
* `dual_model_analysis.ipynb` generates graphs of model performance seen in paper. A html version is also provided with final outputs.

## Data 
Neural net training data is stored in `training_data/`. `grids_200000.npz` contains a (1000000, 4, 5, 5) numpy array of 1,000,000 input grid state representations. `actions_200000.npz` contains a (1000000, 5) numpy array of matched action Q values. The files are saved in compressed .npz format and can be loaded as 
```
actions_array = np.load("training_data/actions_200000.npz")[data]
```

## Neural Network
The CNN model in `models/nn_model` was trained on 200,000 grids or 1,000,000 data points (5 timesteps per grid) for 100 epochs. 
```
# Training Log
Epoch:1, train loss: 0.916, test loss: 0.651, test accuracy: 0.463
Epoch:11, train loss: 0.156, test loss: 0.157, test accuracy: 0.384
Epoch:21, train loss: 0.108, test loss: 0.113, test accuracy: 0.415
Epoch:31, train loss: 0.083, test loss: 0.088, test accuracy: 0.477
Epoch:41, train loss: 0.067, test loss: 0.072, test accuracy: 0.509
Epoch:51, train loss: 0.056, test loss: 0.062, test accuracy: 0.529
Epoch:61, train loss: 0.048, test loss: 0.054, test accuracy: 0.507
Epoch:71, train loss: 0.042, test loss: 0.048, test accuracy: 0.504
Epoch:81, train loss: 0.037, test loss: 0.043, test accuracy: 0.500
Epoch:91, train loss: 0.034, test loss: 0.040, test accuracy: 0.483
Epoch:100, train loss: 0.031, test loss: 0.037, test accuracy: 0.494
training took 8222.492278575897 seconds
Model saved as nn_model
```
