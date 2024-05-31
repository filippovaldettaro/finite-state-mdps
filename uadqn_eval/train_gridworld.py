# adapt implementation of UA-DQN to carry out policy evaluation
# original implementation at https://github.com/IndustAI/risk-and-uncertainty

import os
import numpy as np
import json

from uadqn_eval.agents.uadqn.sarsa_uadqn import SARSA_UADQN
from uadqn_eval.agents.common.networks.mlp import MLP
from utils.value_iteration import VIMDPSolver

from environments.uadqn_gridworld import UADQNGridworld as Gridworld

from environments.mdp import TabularMDP

def get_transition_from_counts(counts, base_mdp:TabularMDP):
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
        transition_matrix[np.array(terminal_states)] = 0
        transition_matrix[np.array(terminal_states),:, np.array(terminal_states)] = 1.
        return transition_matrix

def sarsa_eval(env, num_steps, agent_args):

    gamma = env.gamma
    agent_args['gamma'] = gamma
    agent = SARSA_UADQN(**agent_args)

    agent.learn(timesteps=num_steps)

if __name__ == '__main__':

    
    solver_args = {
         'num_train_steps':10000
    }    

    num_steps = solver_args['num_train_steps']
    data_config = json.load(open(os.path.join(os.path.dirname(__file__),"data","config.json")))

    for num_data in data_config["data_args"]["num_transitions"]:

        print(f'training agent on {num_data} transitions')
        
        base_mdp = Gridworld(**data_config["env_args"])

        run_name = f'{num_data}'
        results_dir = os.path.join('results','uadqn_eval',run_name)

        counts = np.load(os.path.join(os.path.dirname(__file__),"data","datasets",f"{num_data}","counts.npy"))

        base_mdp.transition_matrix = get_transition_from_counts(counts, base_mdp)

        #get MLE policy, to be evaluated by SARSA agent
        
        solver_args = {'epsilon':1e-10}
        solver = VIMDPSolver(solver_args['epsilon'])
        policy, inferred_values = solver.solve_mdp(base_mdp)

        #train agent to evaluate given policy
        logging_folder = os.path.join("results", "uadqn_eval", f'{num_data}_transitions')
        agent_args = {'env':base_mdp, 'policy_arr':policy, 'network':MLP, 'logging':True, 'log_folder_details':logging_folder, 'update_target_frequency':100}

        sarsa_eval(base_mdp, num_steps=num_steps, agent_args=agent_args)
