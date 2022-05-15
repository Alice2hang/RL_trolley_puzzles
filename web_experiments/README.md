# Web Experiment Code
* `time_experiment/` contains code for the version of the experiment with 60 training trials followed by 50 test trials (under time pressure or time delay).   
* `training_experiment/` contains code for the version of the experiment with 60 or 0 training trials (depending on assigned conditions) followed by 24 test trials. 

While the code for these two experiments has significant overlap, they are shared here as two separate folders/packages in order to present things as they were at experiment runtime. 

## Gridworld Task
The code for `gridworld-task` was adapted from https://github.com/markkho/gridworld-task-js. The code in `gridworld-task-src/` is written in es6 rather than javascript, and is compiled into gridworld-task.js using babel to be compatible with older browsers. `gridworld-task.js` is the same in both experiments (training_experiment and time_experiment) and is generated from the provided source. 

## A Note on PHP Files and Backend Data Handling
All database authentication information in *_config.php files has been removed, meaning that data will not be saved on the backend, although the experiments can still be run locally without impact to user experience.

## Run Locally
To set up the package, once `npm` is installed:
```
npm install
```

To run and open page locally (requires python3):
```
npm start
```
