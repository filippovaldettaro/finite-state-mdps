# %%
import os
import torch
import numpy as np
from environments.mdp import TabularMDP
from experiments.policies.mle import MLESolver
from experiments.policies.nominal import NominalSolver
from experiments.policies.grad_stochastic import GradStochasticSolver
from experiments.policies.second_order import SecondOrderSolver
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

for num_visits in num_visits_list:
    os.mkdir(f"synthetic_mdps/M{num_visits}")

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

        mle_solver = MLESolver(1e-8)
        mle_policy = mle_solver.solve(np.array(counts), base_mdp)

        mle_values = []
        eval_mdp = base_mdp
            
        for sample in range(num_eval_samples):

            dynamics_sample = dynamics_samples[sample]
            eval_mdp.transition_matrix = dynamics_sample
            calc = MomentsCalculator(eval_mdp, mle_policy)
            mle_values.append(calc.get_expected_value())

        mle_values = np.array(mle_values)

       
        nominal_solver = NominalSolver(1e-8)
        nominal_policy = nominal_solver.solve(np.array(counts), base_mdp, prior_param=prior_alpha)

        nominal_values = []
        eval_mdp = base_mdp
            
        for sample in range(num_eval_samples):

            dynamics_sample = dynamics_samples[sample]
            eval_mdp.transition_matrix = dynamics_sample
            calc = MomentsCalculator(eval_mdp, nominal_policy)
            nominal_values.append(calc.get_expected_value())

        nominal_values = np.array(nominal_values)

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

        second_order_solver = SecondOrderSolver("nominal", 0.5, 0.01, 1000, None, prior_param=prior_alpha)
        second_order_policy, second_order_losses = second_order_solver.solve(np.array(counts), base_mdp, return_losses=True)

        second_order_values = []
        eval_mdp = base_mdp
            
        for sample in range(num_eval_samples):

            dynamics_sample = dynamics_samples[sample]
            eval_mdp.transition_matrix = dynamics_sample
            calc = MomentsCalculator(eval_mdp, second_order_policy.detach())
            second_order_values.append(calc.get_expected_value())

        second_order_values = np.array(second_order_values)

        mle_ground_truth = MomentsCalculator(base_mdp, mle_policy).get_expected_value().mean()
        nominal_ground_truth = MomentsCalculator(base_mdp, nominal_policy).get_expected_value().mean()
        grad_ground_truth = MomentsCalculator(base_mdp, grad_policy).get_expected_value().mean()
        second_order_ground_truth = MomentsCalculator(base_mdp, second_order_policy).get_expected_value().mean()

        mle_bayes_value = mle_values.mean()
        nominal_bayes_value = nominal_values.mean()
        grad_bayes_value = grad_values.mean()
        second_order_bayes_value = second_order_values.mean()
       
        os.mkdir(f"synthetic_mdps/M{num_visits}/run{run}")
        for string, array in [("mle", mle_ground_truth), ("nominal", nominal_ground_truth), ("grad", grad_ground_truth), ("second_order", second_order_ground_truth)]:
            np.save(f"synthetic_mdps/M{num_visits}/run{run}/{string}_ground_truths", array)
        
        for string, array in [("mle", mle_bayes_value), ("nominal", nominal_bayes_value), ("grad", grad_bayes_value), ("second_order", second_order_bayes_value)]:
            np.save(f"synthetic_mdps/M{num_visits}/run{run}/{string}_bayes_values", array)
    
