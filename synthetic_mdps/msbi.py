
import os
import torch
import numpy as np
from environments.mdp import TabularMDP
from experiments.policies.msbi import MSBISolver
from experiments.policies.grad_stochastic import GradStochasticSolver
import matplotlib.pyplot as plt

from utils.posterior import get_dynamics_pseudocounts, get_dynamics_batch_sample
from utils.moments import MomentsCalculator

num_states = 5
num_actions = 5
seed = 0
prior_alpha = 1.
gamma = 0.999
num_visits_list = [1,2,3,4,5,6,7,8,9,10]
num_runs = 250
num_eval_samples = 10000

prior_distr = torch.distributions.dirichlet.Dirichlet(concentration=torch.tensor(prior_alpha*torch.ones(num_states)))
torch.manual_seed(seed)
np.random.seed(seed)


env_r = np.array(torch.randn(num_states)).repeat(num_states*num_actions).reshape(-1,num_actions,num_states)
if not os.path.isdir("synthetic_mdps/msbi"):
    os.mkdir("synthetic_mdps/msbi")

for num_visits in num_visits_list:
    os.mkdir(f"synthetic_mdps/msbi/M{num_visits}")

    for run in range(num_runs):
        print("data number of visits:", num_visits, "run number:", run)
        torch.manual_seed(run)
        np.random.seed(run)
        env_p = np.array(prior_distr.sample((num_states, num_actions)))
        base_mdp = TabularMDP(num_states, num_actions, transition_matrix=env_p, reward_matrix=env_r, gamma=gamma)

        
        counts = torch.zeros(num_states, num_actions, num_states)
        for visit in range(num_visits):
            for s in range(num_states):
                for a in range(num_actions):
                    next_state = torch.multinomial(torch.tensor(env_p[s, a]),1)
                    counts[s,a,next_state] += 1
        pseudocounts = counts + prior_alpha 


        dynamics_samples = get_dynamics_batch_sample(num_eval_samples, pseudocounts, base_mdp)

        
        grad_solver = GradStochasticSolver('nominal', 0.5, 0.01, 256, 1000, None, False)
        grad_policy, grad_losses = grad_solver.solve(np.array(counts), base_mdp, return_losses=True)

        grad_values = []
        eval_mdp = base_mdp
            
        for sample in range(num_eval_samples):

            dynamics_sample = dynamics_samples[sample]
            eval_mdp.transition_matrix = dynamics_sample
            calc = MomentsCalculator(eval_mdp, grad_policy)
            grad_values.append(calc.get_expected_value())

        grad_values = np.array(grad_values)

        
        msbi_solver = MSBISolver(init_policy="nominal",
        batch_size=2048,
        epsilon=1e-10,
        num_iters=500,
        early_stop=False)

        msbi_policy =  msbi_solver.solve(np.array(counts), base_mdp)

        msbi_values = []
        eval_mdp = base_mdp
            
        for sample in range(num_eval_samples):

            dynamics_sample = dynamics_samples[sample]
            eval_mdp.transition_matrix = dynamics_sample
            calc = MomentsCalculator(eval_mdp, msbi_policy)
            msbi_values.append(calc.get_expected_value())

        msbi_values = np.array(msbi_values)

        
        grad_ground_truth = MomentsCalculator(base_mdp, grad_policy).get_expected_value().mean()
        msbi_ground_truth = MomentsCalculator(base_mdp, msbi_policy).get_expected_value().mean()

        
        grad_bayes_value = grad_values.mean()
        msbi_bayes_value = msbi_values.mean()

        os.mkdir(f"synthetic_mdps/msbi/M{num_visits}/run{run}")
        for string, array in [("grad", grad_ground_truth), ("msbi", msbi_ground_truth)]:
            np.save(f"synthetic_mdps/msbi/M{num_visits}/run{run}/{string}_ground_truths", array)
        
        for string, array in [("grad", grad_bayes_value), ("msbi", msbi_bayes_value)]:
            np.save(f"synthetic_mdps/msbi/M{num_visits}/run{run}/{string}_bayes_values", array)
    
