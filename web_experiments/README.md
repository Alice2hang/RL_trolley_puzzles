# Web Experiment Code
* `insight_experiment/` contains code for experiment 1.  
* `time_experiment/` contains code for experiment 2.   
* `training_experiment/` contains code for experiment 3.   

While the code for these experiments has significant overlap, they are shared here as separate folders in order to present things as they were at experiment runtime. 

## Gridworld Task
The code for `gridworld-task` was adapted from https://github.com/markkho/gridworld-task-js. The code in `gridworld-task-src/` is written in es6 rather than javascript, and is compiled into gridworld-task.js using babel to be compatible with older browsers. `gridworld-task.js` is the same across experiments and is generated from the provided source. 

## A Note on PHP Files and Backend Data Handling
All database authentication information in *_config.php files has been removed, meaning that data will not be saved on the backend, although the experiments can still be run locally without impact to user experience.

## Run Locally
Requires `npm (>=9.5.0)` and `python3 (>=3.9.16)`. In any of the experiment subdirectories, run:
```
npm install
```
Followed by:
```
npm start
```
This should take a few seconds only and will launch a local version of the web task at http://localhost:8123/.
