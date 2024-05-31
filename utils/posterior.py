import numpy as np
import json
import os
from environments.gridworld import Gridworld
from environments.mdp import TabularMDP

def get_dynamics_pseudocounts(counts, base_mdp, prior_param=1):
    #prior param = 1 corresponds to flat Dirichlet prior
    
    num_states, num_actions = base_mdp.num_states, base_mdp.num_actions
    pseudocounts = np.zeros((num_states, num_actions, num_states))
    for state in range(num_states):
        neighbours = base_mdp.get_neighbour_states(state)
        for action in range(num_actions):
            pseudocounts[state, action, neighbours] = prior_param  

    pseudocounts += counts
    return pseudocounts

def get_dynamics_sample(pseudocounts, base_mdp:TabularMDP):
    
    num_states, num_actions, terminal_states = base_mdp.num_states, base_mdp.num_actions, base_mdp.terminal_states
    transition_matrix_sample = np.zeros((num_states, num_actions, num_states))
    for state in range(num_states):
        if state not in base_mdp.terminal_states:
            neighbours = np.sort(np.unique(base_mdp.get_neighbour_states(state)))
            for action in range(num_actions):
                transition_matrix_sample[state, action][neighbours] = np.random.dirichlet(pseudocounts[state, action][neighbours])
    if len(terminal_states)>0:
        transition_matrix_sample[np.array(terminal_states)] = 0
        transition_matrix_sample[np.array(terminal_states),:, np.array(terminal_states)] = 1.

    return transition_matrix_sample

def get_dynamics_batch_sample(num_samples, pseudocounts, base_mdp:TabularMDP):
 
    num_states, num_actions, terminal_states = base_mdp.num_states, base_mdp.num_actions, base_mdp.terminal_states
    transition_matrix_batch = np.zeros((num_samples, num_states, num_actions, num_states))
    for state in range(num_states):
        neighbours = np.sort(np.unique(base_mdp.get_neighbour_states(state)))
        for action in range(num_actions):
            transition_matrix_batch[:,state, action].T[neighbours] = np.random.dirichlet(pseudocounts[state, action][neighbours], num_samples).T
    if len(terminal_states)>0:
        transition_matrix_batch[:,np.array(terminal_states)] = 0
        transition_matrix_batch[:,np.array(terminal_states),:, np.array(terminal_states)] = 1.

    return transition_matrix_batch