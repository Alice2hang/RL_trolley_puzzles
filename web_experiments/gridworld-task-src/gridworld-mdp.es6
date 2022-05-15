// let product = require('cartesian-product');
// let range = require('lodash.range');
import range from 'lodash/range';
import map from 'lodash/map';
import fromPairs from 'lodash/frompairs';
import get from 'lodash/get';
import includes from 'lodash/includes';
import cloneDeep from 'lodash/cloneDeep';
import keys from 'lodash/keys';
const _ = {range, map, fromPairs, get, includes, keys, cloneDeep};
import product from 'cartesian-product';

let ACTION_CODES = {
    '>': [1, 0],
    '<': [-1,0],
    'v': [0,-1],
    '^': [0, 1],
    'x': [0, 0]
};

let REWARD_DICT = {
    'agent_death': -4,
    'cargo1_death': -1,
    'cargo2_death': -2,
    'cargo1_target': 1,
    'cargo2_target': 2
};

String.prototype.replaceAt=function(index, replacement) {
    return this.substr(0, index) + replacement+ this.substr(index + replacement.length);
}

export class GridWorldMDP {
    constructor ({
        switch_pos,
        targets,
    }) {        
        //setting up blue (target2) and green (target1) squares at target locations
        var feature_array = ['.....','.....','.....','.....','.....']
        feature_array[4-targets['target1'][1]] = feature_array[4-targets['target1'][1]].replaceAt(targets['target1'][0], "g") 
        feature_array[4-targets['target2'][1]] = feature_array[4-targets['target2'][1]].replaceAt(targets['target2'][0], "b") 

        this.height = feature_array.length;
        this.width = feature_array[0].length;

        //class vars for switch and targets pos
        this.switch_pos = switch_pos;
        this.target1 = targets['target1'];
        this.target2 = targets['target2'];

        this.states = product([range(this.width), range(this.height)]);
        this.state_features = _.map(this.states, (s) => {
            let [x, y] = s;
            let f = feature_array[this.height - y - 1][x];
            return [s, f]
        });
        this.state_features = _.fromPairs(this.state_features);
		this.actions = ['^', 'v', '<', '>', 'x'];
		this.terminal_state = [-1, -1];
    }

    out_of_bounds(state) {
        return state[0] > 4 || state[0] < 0 || state[1] > 4 || state[1] <0
    }

    arraysEqual(a, b) {
        if (a === b) return true;
        if (a == null || b == null) return false;
        if (a.length != b.length) return false;

        for (var i = 0; i < a.length; ++i) {
            if (a[i] !== b[i]) return false;
        }
        return true;
}

    transition ({state, action}) {
        //if (typeof(state) === 'string') {
        //    state = _.map(state.split(","), parseInt);
        //}
        this.terminal_state = false;

        let agent = state["agent"];
        let cargo1 = state["cargo1"];
        let cargo2 = state["cargo2"];
        let train = state["train"];
        let trainvel = state["trainvel"];

        action = ACTION_CODES[action];

        let new_agent = [agent[0]+action[0], agent[1]+action[1]];
        let new_trainvel = trainvel;
        let new_cargo1 = cargo1;
        let new_cargo2 = cargo2;

        var hitswitch = 0;
        var push1 = 0;
        var push2 = 0;
        var hitagent = 0;

        if (this.out_of_bounds(new_agent)) {
            new_agent = agent;
        }

        if (this.arraysEqual(new_agent,this.switch_pos)){
            new_agent = agent;
            hitswitch = 1;
            if (trainvel[1] == 0){
                new_trainvel = [0, trainvel[0]-trainvel[1]];
            } else {
                new_trainvel = [trainvel[0]-trainvel[1], 0];
            }
        }

        let new_train = [train[0]+new_trainvel[0],train[1]+new_trainvel[1]];

        var new_cargo = [new_agent[0] + action[0],new_agent[1] + action[1]];
        var train_stopped = this.arraysEqual(new_agent,train) && this.arraysEqual(trainvel,[0,0]);

        if (this.arraysEqual(new_agent,cargo1)) {
            var pos_open = !this.arraysEqual(new_cargo,cargo2) && !this.arraysEqual(new_cargo,this.switch_pos);
            push1 = 1;
            if (this.out_of_bounds(new_cargo) || !pos_open || train_stopped){
                new_agent = agent;
            } else {
                new_cargo1 = new_cargo;
            }
        }

        if (this.arraysEqual(new_agent,cargo2)){
            var pos_open = !this.arraysEqual(new_cargo,cargo1) && !this.arraysEqual(new_cargo,this.switch_pos);
            push2 = 1;
            if (this.out_of_bounds(new_cargo) || !pos_open || train_stopped){
                new_agent = agent;
            } else {
                new_cargo2 = new_cargo;
            }
        }

        if (this.arraysEqual(agent,new_train) && this.arraysEqual(new_agent,train)){
            new_agent = agent;
        }

        if (this.arraysEqual(new_agent,new_train) && !this.arraysEqual(new_trainvel,[0,0])){
            hitagent = 1;
            new_trainvel = [0,0];
            this.terminal_state = true
        }

        if (this.arraysEqual(new_train,new_cargo1) || this.arraysEqual(new_train,new_cargo2)){
            new_trainvel = [0,0];
        }

        return {'agent': new_agent,
                'cargo1': new_cargo1,
                'cargo2': new_cargo2,
                'train': new_train,
                'trainvel': new_trainvel,
                'hitswitch': hitswitch,
                'push1': push1,
                'push2': push2,
                'hitagent': hitagent}
    }

    reward ({
        state,
        action,
        nextstate,
    }) {
        
        let r = 0;
        let position = [0,0]

        var target1 = this.arraysEqual(state["cargo1"],this.target1)
        var target1_next = this.arraysEqual(nextstate["cargo1"],this.target1)
        var target2 = this.arraysEqual(state["cargo2"],this.target2)
        var target2_next = this.arraysEqual(nextstate["cargo2"],this.target2)
        var collision_cargo1 = this.arraysEqual(state["cargo1"],state["train"]) 
        var collision_cargo2 = this.arraysEqual(state["cargo2"],state["train"])
        var collision_next_cargo1 = this.arraysEqual(nextstate["cargo1"],nextstate["train"]) 
        var collision_next_cargo2 = this.arraysEqual(nextstate["cargo2"],nextstate["train"])
        var collision_agent = this.arraysEqual(nextstate["agent"],nextstate["train"])
 
        var hit1 = 0;
        var hit2 = 0;
        var get1 = 0;
        var get2 = 0;

        if (target1_next && !target1){
            r += REWARD_DICT['cargo1_target'];
            position = this.target1;
            get1 = 1;
        }
        if (target2_next && !target2){
            r += REWARD_DICT['cargo2_target'];
            position = this.target2
            get2 = 1;
        }
        if (!target1_next && target1){
            r -= REWARD_DICT['cargo1_target'];
            position = this.target1
        }
        if (!target2_next && target2){
            r -= REWARD_DICT['cargo2_target'];
            position = this.target2
        }
        if (collision_agent){
            r += REWARD_DICT['agent_death'];
            position = nextstate["train"]
        }
        if (!collision_cargo1 && collision_next_cargo1){
            hit1 = 1;
            if (this.arraysEqual(nextstate['cargo1'], this.target1)){
                r -= REWARD_DICT['cargo1_target'];
            }
            r += REWARD_DICT['cargo1_death']
            position = nextstate['train']
        }
        if (!collision_cargo2 && collision_next_cargo2){
            hit2 = 1;
            if (this.arraysEqual(nextstate['cargo2'], this.target2)){
                r -= REWARD_DICT['cargo2_target'];
            }
            r += REWARD_DICT['cargo2_death']
            position = nextstate['train']
        }

        return {'value':r,'position':position,'hit1':hit1,'hit2':hit2,'get1':get1,'get2':get2}
    }

    is_terminal(){
        return this.terminal_state;
    }


}