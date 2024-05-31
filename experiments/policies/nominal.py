import numpy as np
from environments.mdp import TabularMDP
from environments.gridworld import Gridworld
import os
import json

from utils.posterior import get_dynamics_pseudocounts

class NominalSolver:

    def __init__(self, epsilon):

        self.epsilon = epsilon #value iteration tolerance parameter
    
    def solve_mdp(self, mdp: TabularMDP):

        V = np.zeros(mdp.num_states)
        delta = np.inf

        while delta > self.epsilon:
            
            Q_next = (mdp.transition_matrix*(mdp.reward_matrix+mdp.gamma*V.reshape(1,1,-1))).sum(2)
            V_next = Q_next.max(1)

            delta = np.abs(V_next - V).max()

            V = V_next
        
        policy = np.zeros(mdp.num_states, dtype=int)
        
        for state in range(mdp.num_states):
            policy[state] = Q_next[state].argmax()

        return policy, V
    
    def get_transition_from_counts(self, counts, base_mdp:TabularMDP):
        num_states = base_mdp.num_states
        num_actions = base_mdp.num_actions
        terminal_states = base_mdp.terminal_states
        #edge case if counts for s,a pair are zero - give equal probability to the neighbours
        for state in range(num_states):
            for action in range(num_actions):
                if not counts[state, action].any():
                    for neighbour in base_mdp.get_neighbour_states(state):
                        #if neighbour not in terminal_states:
                        counts[state, action, neighbour] = 1
            
        transition_matrix = counts/np.expand_dims(counts.sum(-1), -1)
        if len(terminal_states)>0:
            transition_matrix[np.array(terminal_states)] = 0
            transition_matrix[np.array(terminal_states),:, np.array(terminal_states)] = 1.
        return transition_matrix
    
    def solve(self, counts, base_mdp: TabularMDP, prior_param: float = 1):

        pseudocounts = get_dynamics_pseudocounts(counts, base_mdp, prior_param=prior_param)
        dynamics_model = self.get_transition_from_counts(pseudocounts, base_mdp)
        base_mdp.transition_matrix = dynamics_model

        policy, V = self.solve_mdp(base_mdp)

        return policy

if __name__ == '__main__':

    solver_args = {
        'epsilon':1e-10
    }

    datasets_dir = os.path.join(os.path.dirname(os.path.relpath(__file__)), "data","datasets")
    data_config = json.load(open(os.path.join(os.path.relpath(os.path.dirname(__file__)),"data","config.json")))
    base_mdp = Gridworld(**data_config["env_args"])

    results_base_dir = os.path.join("results","policies","nominal")
    if not os.path.isdir(results_base_dir):
        os.makedirs(results_base_dir)
    json.dump(solver_args, open(os.path.join(results_base_dir,"solver_args.json"),"x"))

    for root, dirs, files in os.walk(datasets_dir):
        if "counts.npy" in os.listdir(root):
            dataset_size = os.path.split(os.path.dirname(root))[-1]
            dataset_seed = os.path.split(root)[-1]

            counts = np.load(os.path.join(root, "counts.npy"))

            solver = NominalSolver(**solver_args)
            policy = solver.solve(counts, base_mdp)

            results_folder = os.path.join(results_base_dir,dataset_size,dataset_seed)
            if not os.path.isdir(results_folder):
                os.makedirs(results_folder)
            
            np.save(os.path.join(results_folder,"policy.npy"), policy)
