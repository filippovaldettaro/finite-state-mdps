import os
import json
import copy

import numpy as np
import torch

from environments.mdp import TabularMDP
from environments.gridworld import Gridworld
from experiments.policies.mle import MLESolver
from experiments.policies.nominal import NominalSolver

from utils.posterior import get_dynamics_pseudocounts, get_dynamics_sample

class MSBISolver:

    def __init__(self, init_policy, batch_size, epsilon, num_iters, early_stop):
        self.init_policy = init_policy
        self.epsilon = epsilon             #value iteration tolerance parameter
        self.batch_size = batch_size
        self.num_iters = num_iters
        self.early_stop = early_stop
    
    def solve(self, counts, base_mdp: TabularMDP):
        num_states, num_actions, reward = base_mdp.num_states, base_mdp.num_actions, base_mdp.reward_matrix
        start_distribution, gamma, terminal_states = base_mdp.start_distribution, base_mdp.gamma, base_mdp.terminal_states
        
        if self.init_policy is None:
            policy = np.zeros(num_states, dtype=int)
        elif self.init_policy == 'random':
            policy = np.random.randint(num_actions, size=num_states)
        elif self.init_policy == "mle":
            init_solver = MLESolver(1e-8)
        elif self.init_policy == "nominal":
            init_solver = NominalSolver(1e-8)

        policy = init_solver.solve(counts, base_mdp)


        N = self.batch_size
        reward_ensemble = np.stack([reward for _ in range(N)], axis=0)
        pseudocounts = get_dynamics_pseudocounts(counts, base_mdp)
        transition_samples = get_dynamics_sample(pseudocounts, base_mdp)
        
        V = np.zeros((N, base_mdp.num_states))
        
        tried_policies = []
        tried_values = []
        tried_deltas = []
        have_tried = False

        delta = np.inf
        num_iter = 0
        while delta > self.epsilon and not have_tried and num_iter < self.num_iters:
            num_iter += 1
            delta_old = delta

            Q_next = (transition_samples*(reward_ensemble+gamma*V.reshape(N,1,1,-1))).sum(3)
            V_next = np.take_along_axis(Q_next, policy.reshape(1,-1,1), 2).squeeze()

            delta = np.abs(V_next.mean(0) - V.mean(0)).max()

            V = V_next
            if delta > delta_old:
                tried_policies.append(copy.copy(policy))
                tried_values.append(V.mean(0))
                tried_deltas.append(delta_old)

            policy = np.argmax(Q_next.mean(0), 1)

            if self.early_stop:
                for p in tried_policies:
                    if (p==policy).all():
                        have_tried=True
        
        if self.early_stop:
            if not have_tried:
                #policy not in tried_policies:
                tried_policies.append(policy)
                tried_values.append(V.mean(0))
                tried_deltas.append(delta)

            best_policy_idx = np.array(tried_deltas).argmin()
            policy = tried_policies[best_policy_idx]
            delta = tried_deltas[best_policy_idx]

            values = tried_values[best_policy_idx]
        else:
            values = V.mean(0)

        return policy

if __name__ == '__main__':

    solver_args = {
        "init_policy":"nominal",
        "batch_size":32768,
        "epsilon":1e-10,
        "num_iters":2000,
        "early_stop":False,
    }

    datasets_dir = os.path.join(os.path.dirname(os.path.relpath(__file__)), "data","datasets")
    data_config = json.load(open(os.path.join(os.path.relpath(os.path.dirname(__file__)),"data","config.json")))
    base_mdp = Gridworld(**data_config["env_args"])

    results_base_dir = os.path.join("results","policies","msbi")
    os.makedirs(results_base_dir)
    json.dump(solver_args, open(os.path.join(results_base_dir,"solver_args.json"),"x"))

    for root, dirs, files in os.walk(datasets_dir):
        if "counts.npy" in os.listdir(root):
            print("new dataset")
            dataset_size = os.path.split(os.path.dirname(root))[-1]
            dataset_seed = os.path.split(root)[-1]

            counts = np.load(os.path.join(root, "counts.npy"))

            solver = MSBISolver(**solver_args)
            policy = solver.solve(counts, base_mdp)

            results_folder = os.path.join(results_base_dir,dataset_size,dataset_seed)
            os.makedirs(results_folder)
            np.save(os.path.join(results_folder,"policy.npy"), policy)
