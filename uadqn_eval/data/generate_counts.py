import datetime
import json
import os

import numpy as np
from environments.gridworld import Gridworld

# creates a set of datasets, each dataset with less transitions is a proper subset of any of the datasets with more transitions
# for example, all the transitions in dataset of size 25 will also be present in the dataset of size 50
# environment and datasets generated using parameters in config.json

if __name__ == '__main__':

    data_dir = os.path.dirname(os.path.relpath(__file__))

    config = json.load(open(os.path.join(data_dir,"config.json")))
    env_args = config["env_args"]
    data_args = config["data_args"]

    num_transitions = data_args["num_transitions"]
    seed = data_args["seed"]

    for dataset_size in num_transitions:
    
        np.random.seed(seed)

        ground_truth_mdp = Gridworld(**env_args)

        transitions = []
        num_states, num_actions = ground_truth_mdp.num_states, ground_truth_mdp.num_actions
        data_policy = np.ones((num_states, num_actions))/num_actions

        for t in range(dataset_size):

            done = False
            state = int(ground_truth_mdp.reset())
            
            while not done:
                if len(data_policy.shape) == 1:
                    action = data_policy[state]
                if len(data_policy.shape) == 2:
                    action = np.random.choice(np.arange(num_actions), p=data_policy[state])
                next_state, reward, done, info = ground_truth_mdp.step(action)
                transitions.append([int(state), int(action), int(next_state), float(reward), int(done)])
                state = next_state
                done = True

        counts = np.zeros((num_states, num_actions, num_states), dtype=int)
        for state, action, next_state, reward, done in transitions:
            counts[state, action, next_state] += 1

        name = f"{dataset_size}"

        run_folder = os.path.join(data_dir,"datasets",name)
        
        if not os.path.exists(run_folder):
            os.makedirs(run_folder)

        if len(data_policy.shape) == 1:
            policy_to_log = [int(p) for p in data_policy]
        else:
            policy_to_log = data_args["policy"]
        with open(os.path.join(run_folder,"run_args.json"), "w+") as f:
            json.dump({'gridworld_args':env_args, 'seed':seed, 'num_transitions':dataset_size, 'policy':policy_to_log}, f)
        np.save(os.path.join(run_folder,"counts"), np.array(counts))

