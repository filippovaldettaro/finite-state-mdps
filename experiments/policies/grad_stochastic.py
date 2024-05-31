import os
import json

import numpy as np
import torch
from torch.nn.parameter import Parameter

from utils.posterior import get_dynamics_pseudocounts, get_dynamics_batch_sample

from environments.gridworld import Gridworld
from environments.mdp import TabularMDP
from experiments.policies.mle import MLESolver
from experiments.policies.nominal import NominalSolver


class GradStochasticSolver:

    def __init__(self, init_policy, policy_softness, lr, batch_size, num_steps, state_distribution, snap_to_deterministic):

        self.lr = lr
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.state_distribution = state_distribution

        self.init_policy = init_policy
        if init_policy is None:
            self.policy_softness = None
        else:
            self.policy_softness = policy_softness
        self.lr = lr
        self.snap_to_deterministic = snap_to_deterministic
        
    def sample_fn(self):
        return get_dynamics_batch_sample(self.batch_size, self.pseudocounts, self.base_mdp)

    def solve(self, counts, base_mdp:TabularMDP, return_losses=False):
        num_states, num_actions, reward = base_mdp.num_states, base_mdp.num_actions, base_mdp.reward_matrix
        start_distribution, gamma, terminal_states = base_mdp.start_distribution, base_mdp.gamma, base_mdp.terminal_states
        N = self.batch_size
        reward_ensemble = torch.tensor(np.stack([reward for _ in range(N)], axis=0))
        pseudocounts = get_dynamics_pseudocounts(counts, base_mdp)


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
        self.optim = torch.optim.SGD([self.params], lr=self.lr)

        losses = []
        print("beginning")
        for step in range(self.num_steps):

            transition_samples = torch.tensor(get_dynamics_batch_sample(self.batch_size, pseudocounts, base_mdp))  #independent samples each step

            soft_q = torch.exp(self.params)
            policy = soft_q/soft_q.sum(1).view((num_states,1))

            transition_matrix_ens = (transition_samples*policy.view((1,num_states,num_actions,1))).sum(2) # N,s,s'

            to_invert = torch.eye(num_states).repeat((N,1,1)) - gamma*transition_matrix_ens
            inverse = torch.linalg.inv(to_invert)

            value = (inverse*((reward_ensemble*transition_samples*policy.view((1,num_states,num_actions,1))).sum((2,3)).view(N,1,-1))).sum(-1)
            
            objective = value.mean(0)
            
            if self.state_distribution is None:
                loss = -objective.mean()  #uniform sum over states
            elif isinstance(self.state_distribution, int):
                loss = -objective[self.state_distribution]
            self.optim.zero_grad()

            loss.backward()
            self.optim.step()

            if step%500 == 1:
                print("step ", step)
                print("loss", np.mean(losses[-500:]))
            losses.append(loss.item())

        soft_q = torch.exp(self.params)
        policy = soft_q/soft_q.sum(1).view((num_states,1))
        losses = np.array(losses)
        policy = policy.detach().numpy()

        if return_losses:
            return policy, losses
        
        return policy

if __name__ == '__main__':

    solver_args = {'init_policy':"nominal",
                    'policy_softness':0.1,
                    'lr':10,
                    'batch_size':128,
                    'num_steps':10000,
                    'state_distribution':None,
                    'snap_to_deterministic':True}

    datasets_dir = os.path.join(os.path.dirname(os.path.relpath(__file__)), "data","datasets")
    data_config = json.load(open(os.path.join(os.path.relpath(os.path.dirname(__file__)),"data","config.json")))
    base_mdp = Gridworld(**data_config["env_args"])

    results_base_dir = os.path.join("results","policies","grad_stochastic")
    os.makedirs(results_base_dir)
    json.dump(solver_args, open(os.path.join(results_base_dir,"solver_args.json"),"x"))

    for root, dirs, files in os.walk(datasets_dir):
        if "counts.npy" in os.listdir(root):
            dataset_size = os.path.split(os.path.dirname(root))[-1]
            dataset_seed = os.path.split(root)[-1]

            counts = np.load(os.path.join(root, "counts.npy"))

            solver = GradStochasticSolver(**solver_args)
            policy = solver.solve(counts, base_mdp)

            results_folder = os.path.join(results_base_dir,dataset_size,dataset_seed)
            os.makedirs(results_folder)
            np.save(os.path.join(results_folder,"policy.npy"), policy)
