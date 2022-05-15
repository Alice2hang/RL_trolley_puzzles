const num_training = 60
const num_test = 24
var total_score = 0;
var total_best = 0;
var userid = -1;

function clickStart(hide, show){
    if (hide!="consent" || document.getElementById("consent_checkbox").checked){
        document.getElementById(hide).style.display="none";
        document.getElementById(show).style.display = "block";
        window.scrollTo(0,0);
    } 
    if (show == 'finishpage'){
        var bonus = total_best * 0.05
        bonus = Math.round(100*bonus)/100 
        if (bonus < 0){ bonus = 0 }
        document.getElementById("completioncode").innerHTML =  "Secret Completion Code: 123"
        document.getElementById("bonusmsg").innerHTML = "You got the best possible score on  " + parseInt(total_best) + " grids, providing a bonus of $" + parseFloat(bonus) + ". Please enter your mTurk ID so that we can correctly assign your bonus. Please be aware that this may take some time to process."    
    }
}

function test_info(GridWorldTask){
    var test_info = ""
    if (train_group == 1){
        test_info = "You have completed " + String(num_training) + "/" + String(num_training+num_test) + " trials. ";
    }
    test_info += "For the next " + String(num_test) + " trials, you will be placed under a time constraint.";   
    document.getElementById('test_info').innerHTML = test_info  
    
    var test_info3 = ""
    test_group = 1
    if (test_group==0){
        test_info3 += "You should take 7 seconds to look at the board and plan your moves. When 7 seconds is up, the counter will turn green and you can then take your 5 steps<br><br>Try one practice trial below, and get the best score to continue!" 
    } else {
        test_info3 += "You will have a time limit of 7 seconds to complete the board. There will be a counter on the right displaying your time remaining. <b>If you do not complete all 5 moves before time runs out, you will get the lowest possible reward of -4.</b> Therefore, you should always try to finish 5 moves in each round, even if this means hitting the spacebar several times after you have accomplished your goals. <br><br>Try one practice trial below, and get the best score to continue!"
    }
    document.getElementById('test_info3').innerHTML = test_info3  
    document.getElementById('test_button').disabled = true
    document.getElementById('test_button').style.color = "red"
}

function get_random_idx(){
    idxs = [1,2,3,4,5,6]
    test_idxs = []
    special_test = [
        [101,102,103,104,105,106,107,108],
        [201,202,203,204,205,206,207,208],
        [301,302,303,304,305,306,307,308],
        [401,402,403,404,405,406,407,408]]
    for (index = 0; index < special_test.length; index++) { 
        shuffle(special_test[index])
        test_idxs.push(...special_test[index].slice(0, 3))
    } 
    filler_idxs = [7,8,9,10,11,12]
    test_idxs.push(...filler_idxs)
    shuffle(idxs)
    shuffle(test_idxs)
    return [...idxs,...test_idxs]
}

function shuffle(a) {
    var j, x, i;
    for (i = a.length - 1; i > 0; i--) {
        j = Math.floor(Math.random() * (i + 1));
        x = a[i];
        a[i] = a[j];
        a[j] = x;
    }
    return a;
}

function run_train(data,GridWorldTask,num=1,idxs=undefined) {
    if (idxs == undefined){
        idxs = Array.apply(null, {length: num_training}).map(Number.call, Number)
    }

    idx = Math.floor(Math.random() * idxs.length);
    idx = idxs[idx]
    idxs.splice(idxs.indexOf(idx), 1);
    console.log(idx)

    document.getElementById('tasknum').innerText = "Trial " + String(num) + "/" + String(num_test+num_training);
    trial_data = data[idx]
    let task = new GridWorldTask({
        reset: false,
        container: $("#task")[0],
        reward_container: $("#reward")[0],
        step_callback: (d) => {},
        endtask_callback: (result_data,r,timeout) => {
            total_score += r;
            saveData(num, idx, result_data, r, trial_data['best_reward'],"train",train_group, false)
            if (num >= num_training){
                document.getElementById('test_button').style.color = "red";
                document.getElementById('test_button').disabled = true;
                test_info(GridWorldTask);
                clickStart('page1','testinfo')
                clearGrid('page1')
            }
            else {run_train(data,GridWorldTask,num+1,idxs)}
        }
    });
    task.init({
        init_state: {
            'agent': trial_data['agent'],
            'cargo1': trial_data['cargo1'],
            'cargo2': trial_data['cargo2'],
            'train': trial_data['train'],
            'trainvel': trial_data['trainvel']
        },
        switch_pos: trial_data['switch'],
        targets: {
            'target1': trial_data['target1'],
            'target2': trial_data['target2']
        },
        show_rewards: true,
        wait_time: 0,
        time_limit: undefined,
        best_reward: trial_data['best_reward']
    });
    task.start();
}

function run_test(data,GridWorldTask,test_group,num=1,idxs=get_random_idx()) {
    
    var trial_num = num_training * train_group + num
    var num_total = num_training * train_group + num_test
    document.getElementById('tasknum2').innerText = "Trial " + String(trial_num) + "/" + String(num_total);

    var wait_time
    var time_limit

    if (test_group == 0){ //time delay group
        wait_time = 7;
    } else { //time pressure group
        wait_time = 0;
        time_limit = 7;
    }

    idx = idxs[num-1]
    trial_data = data[idx]
    console.log(idx)
    
    task = new GridWorldTask({
        reset: false,
        container: $("#task2")[0],
        reward_container: $("#reward2")[0],
        time_container: $("#timer2")[0],
        step_callback: (d) => {},
        endtask_callback: (result_data,r,timeout) => {
            total_score += r;
            if (r === trial_data['best_reward']){
                total_best += 1;
            }
            saveData(num+num_training, idx, result_data, r, trial_data['best_reward'],"test", train_group,timeout)
            if (num >= num_test){clickStart('page2','feedbackpage')}
            else {run_test(data,GridWorldTask, test_group, num+1, idxs)}
        }
    });
    task.init({
        init_state: {
            'agent': trial_data['agent'],
            'cargo1': trial_data['cargo1'],
            'cargo2': trial_data['cargo2'],
            'train': trial_data['train'],
            'trainvel': trial_data['trainvel']
        },
        switch_pos: trial_data['switch'],
        targets: {
            'target1': trial_data['target1'],
            'target2': trial_data['target2']
        },
        show_rewards: true,
        wait_time: wait_time,
        time_limit: time_limit,
        best_reward: trial_data['best_reward']
    });
    task.start();
}

function saveData(num, idx, trial_data, r, rmax, type, train_group, timeout) {
    var datajson = {};
    var time_out = 0
    if (timeout==true){
        time_out = 1
    }

    for (i = 0; i < 5; i++){
        var data = trial_data[i];
        var step = i+1;
        var action = undefined;
        var millis = undefined;
        var reward_step = undefined;
        var reward_cum = undefined;
        var hitswitch = undefined;
        var push1 = undefined;
        var push2 = undefined;
        var hitagent = undefined;
        var hit1 = undefined;
        var hit2 = undefined;
        var get1 = undefined;
        var get2 = undefined
        var state = undefined;

        if (data != undefined){
            action = data[0];
            millis = data[1];
            reward_step = data[2]
            reward_cum = data[3]
            hitswitch = data[4]
            push1 = data[5]
            push2 = data[6]
            hitagent = data[7]
            hit1 = data[8]
            hit2 = data[9]
            get1 = data[10]
            get2 = data[11]
            state = data[12]
        }
        datajson[i] = {
            'userid': userid,
            'trialnum': num,
            'gridnum': idx,
            'type': type,
            'training': train_group,
            'step': step,
            'action': action,
            'reaction_millis': millis,
            'reward_step': reward_step,
            'reward_cum': reward_cum,
            'reward_max': undefined,
            'hitswitch': hitswitch,
            'push1': push1,
            'push2': push2,
            'hitagent': hitagent,
            'hit1': hit1,
            'hit2': hit2,
            'get1': get1,
            'get2': get2,
            'state': state
        }
    }

    datajson[5] = {
        'userid': userid,
        'trialnum': num,
        'gridnum': idx,
        'type': type,
        'timeout': time_out,
        'training': train_group,
        'step': 6,
        'reward_cum': r,
        'reward_max': rmax,
    }

    datajson = JSON.stringify(datajson)

    var xhr = new XMLHttpRequest();
    xhr.open('POST', 'write_data.php');
    xhr.setRequestHeader('Content-Type', 'application/json');
    xhr.onload = function() {
      if(xhr.status == 200){
        var response = JSON.parse(xhr.responseText); 
        userid = response["userid"];
      }
    };
    xhr.send(datajson);
  }
