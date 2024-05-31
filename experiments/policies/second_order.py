import torch
from torch.nn import Parameter
import numpy as np
from environments.mdp import TabularMDP
from environments.gridworld import Gridworld
import os
import json

from utils.posterior import get_dynamics_pseudocounts
from utils.moments import MomentsCalculator
from experiments.policies.mle import MLESolver
from experiments.policies.nominal import NominalSolver

      
class SecondOrderSolver:

    def __init__(self, init_policy, policy_softness, lr, num_steps, state_distribution, prior_param=1):

        self.prior_param = 1
        self.init_policy = init_policy
        self.policy_softness = policy_softness
        self.lr = lr
        self.num_steps = num_steps
        self.state_distribution = state_distribution
        if state_distribution != "uniform" and not state_distribution is None:
            raise(NotImplementedError)
    
    def set_theta_covars(self, base_mdp, pseudocounts):

        self.covariances = torch.zeros((base_mdp.num_states, base_mdp.num_actions, base_mdp.num_states, base_mdp.num_states))   #[S,A,S_1,S_2] covar between s1 and s2 in state-action S,A

        for state in range(base_mdp.num_states):
            if state not in base_mdp.terminal_states:
                neighbours = base_mdp.get_neighbour_states(state)
                for action in range(base_mdp.num_actions):
                    alphas = pseudocounts[state,action]
                    alpha_0 = alphas[torch.tensor(neighbours)].sum()
                    for sn1 in neighbours:
                        for sn2 in neighbours:
                            self.covariances[state,action,sn1,sn2] = (-alphas[sn1]*alphas[sn2] + (sn1==sn2)*alphas[sn1]*alpha_0)/(alpha_0**2*(alpha_0+1))

        return self.covariances
    
    def second_order_eval(self, nominal_mdp, pseudocounts, policy):
     
        gamma = nominal_mdp.gamma
        q = torch.ones(nominal_mdp.num_states)/torch.ones(nominal_mdp.num_states+1).sum() #state distribution
          
        XPi = torch.inverse(torch.eye(nominal_mdp.num_states) - gamma*(policy.unsqueeze(2)*torch.tensor(self.nominal_transition)).sum(1)).float()
        
        #first term
        first_term = q.T.float()@XPi.float()@self.reward_vec.float()
       
        #second term
        PiQ = torch.einsum("ia, si, iasj -> ij", (policy**2), XPi, self.covariances)
        second_term = gamma**2 * (q.T@XPi@PiQ@XPi@self.reward_vec.float())

        self.first_terms.append(first_term.item())
        self.second_terms.append(second_term.item())
        return first_term + second_term

    def get_transition_from_counts(self, counts, base_mdp:TabularMDP):
        num_states = base_mdp.num_states
        num_actions = base_mdp.num_actions
        terminal_states = base_mdp.terminal_states
           
        transition_matrix = counts/np.expand_dims(counts.sum(-1), -1)
        if len(terminal_states)>0:
            transition_matrix[np.array(terminal_states)] = 0
            transition_matrix[np.array(terminal_states),:, np.array(terminal_states)] = 1.
        return transition_matrix
    
    def solve(self, counts, base_mdp: TabularMDP, return_losses=False, verbose=False):

        num_states, num_actions = base_mdp.num_states, base_mdp.num_actions
        pseudocounts = torch.tensor(get_dynamics_pseudocounts(counts, base_mdp, prior_param=self.prior_param))
        self.nominal_transition = torch.tensor(self.get_transition_from_counts(pseudocounts, base_mdp))

        for terminal_state in base_mdp.terminal_states:
            self.nominal_transition[terminal_state] = 0
            self.nominal_transition[terminal_state,:,terminal_state] = 1
        self.reward_vec = torch.tensor(base_mdp.reward_matrix[:,0,0])   #assume reward only depends on state
        
        self.set_theta_covars(base_mdp, pseudocounts)
        
        if self.init_policy is None:
            init_params = torch.randn((num_states, num_actions))
        elif self.init_policy == 'random':
            init_params = torch.zeros((num_states, num_actions))
        elif self.init_policy in ('mle', 'nominal'):
            if self.init_policy == "mle":
                init_solver = MLESolver(1e-8)
            elif self.init_policy == "nominal":
                init_solver = NominalSolver(1e-8)
            init_policy = init_solver.solve(counts, base_mdp)
            assert(0<self.policy_softness<1)
            p = 1-self.policy_softness
            q = (1-p)/(num_actions-1)
            init_params = torch.log(q*torch.ones(num_states, num_actions))
            for s in range(num_states):
                init_params[s, init_policy[s]] = torch.log(torch.tensor(p))
        
        self.params = Parameter(init_params)
        self.optim = torch.optim.Adam([self.params], lr=self.lr)

        self.losses = []
        self.first_terms = []
        self.second_terms = []
        for step in range(self.num_steps):
            soft_q = torch.exp(self.params)
            policy = soft_q/soft_q.sum(1).view((num_states,1))
            
            self.optim.zero_grad()
            loss = -self.second_order_eval(base_mdp, pseudocounts, policy)

            loss.backward()
            self.optim.step()
            self.losses.append(loss.item())
            if step%500 == 1 and verbose:
                print("step ", step)
                print("loss", loss.item())
        policy = soft_q/soft_q.sum(1).view((num_states,1))
        if return_losses:
            return policy.detach(), self.losses 
        return policy.detach()

if __name__ == '__main__':

    solver_args = {'init_policy':"nominal",
                    'policy_softness':0.1,
                    'lr':1e-2,
                    'num_steps':10000,
                    'state_distribution':None}

    datasets_dir = os.path.join(os.path.dirname(os.path.relpath(__file__)), "data","datasets")
    data_config = json.load(open(os.path.join(os.path.relpath(os.path.dirname(__file__)),"data","config.json")))
    base_mdp = Gridworld(**data_config["env_args"])

    results_base_dir = os.path.join("results","policies","second_order")
    if not os.path.isdir(results_base_dir):
        os.makedirs(results_base_dir)
    json.dump(solver_args, open(os.path.join(results_base_dir,"solver_args.json"),"x"))

    for root, dirs, files in os.walk(datasets_dir):
        if "counts.npy" in os.listdir(root):
            dataset_size = os.path.split(os.path.dirname(root))[-1]
            dataset_seed = os.path.split(root)[-1]


            counts = np.load(os.path.join(root, "counts.npy"))
            solver = SecondOrderSolver(**solver_args)
            modified_reward_matrix = np.zeros_like(base_mdp.reward_matrix)
            modified_reward_matrix[-1] = (1-base_mdp.gamma)*base_mdp.success_reward    #only valid for gridworld: reformulates transition matrix to be state-dependent only. synthetic mdps don't run this file as main
            base_mdp.reward_matrix = modified_reward_matrix
            policy = solver.solve(counts, base_mdp)
            
            results_folder = os.path.join(results_base_dir,dataset_size,dataset_seed)
            if not os.path.isdir(results_folder):
                os.makedirs(results_folder)
                            
            np.save(os.path.join(results_folder,"policy.npy"), policy.detach().numpy())
            np.save(os.path.join(results_folder,"losses.npy"), np.array(solver.losses))
            np.save(os.path.join(results_folder,"first_terms.npy"), np.array(solver.first_terms))
            np.save(os.path.join(results_folder,"second_terms.npy"), np.array(solver.second_terms))

