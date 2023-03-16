# Dual-Agent Code and Analysis
* `grid.py` defines the gridworld MDP including transition and reward functions.
* `utils.py` includes helper functions for `grid.py`, as well as graphics and converting state representations to an array for downstream neural net processing.
* `value_iteration.py` helper function for `generate_data.py` that solves a given input grid using value iteration.
* `generate_data.py` generates data for neural network training and for experimental component.  
* `neural_net.py` defines neural net layers, training, and prediction.
* `agent.py` implements monte carlo tree search to solve gridworld problems and allows for neural network initialization of Q-values.
* `dual_model_analysis.ipynb` generates graphs of model performance seen in paper. A html version is also provided with final outputs.
* `supplementals.ipynb` generates exploratory figures not featured in the paper or supplement.

## Data 
Neural net training data is stored in `training_data/`. `grids_200000.npz` contains a (1000000, 4, 5, 5) numpy array of 1,000,000 input grid state representations. `actions_200000.npz` contains a (1000000, 5) numpy array of matched action Q values. The files are saved in compressed .npz format and can be loaded as 
```
actions_array = np.load("training_data/actions_200000.npz")[data]
```

## Neural Network
The CNN model in `models/nn_model` was trained on 200,000 grids or 1,000,000 data points (5 timesteps per grid) for 100 epochs. Because action labels are computed with Q-value iteration, multiple actions may be tied for "best action" at a given grid state. Accuracy reflects whether the predicted "best action" matches one of the highest valued label actions.
```
# Training Log
Epoch:1, train loss: 0.922, test loss: 0.652, test accuracy: 0.549
Epoch:11, train loss: 0.153, test loss: 0.155, test accuracy: 0.896
Epoch:21, train loss: 0.103, test loss: 0.106, test accuracy: 0.923
Epoch:31, train loss: 0.077, test loss: 0.081, test accuracy: 0.930
Epoch:41, train loss: 0.062, test loss: 0.067, test accuracy: 0.938
Epoch:51, train loss: 0.052, test loss: 0.057, test accuracy: 0.944
Epoch:61, train loss: 0.045, test loss: 0.050, test accuracy: 0.950
Epoch:71, train loss: 0.040, test loss: 0.045, test accuracy: 0.953
Epoch:81, train loss: 0.036, test loss: 0.044, test accuracy: 0.946
Epoch:91, train loss: 0.032, test loss: 0.038, test accuracy: 0.952
Epoch:100, train loss: 0.030, test loss: 0.042, test accuracy: 0.939
```
