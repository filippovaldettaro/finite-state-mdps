# aleatoric vs epistemic experiment

import os
import numpy as np
import json

from experiments.policies.mle import MLESolver

from environments.gridworld import Gridworld
from environments.mdp import TabularMDP
from utils.posterior import get_dynamics_pseudocounts, get_dynamics_batch_sample
from utils.moments import MomentsCalculator

def get_value_ep_al_stds(policy, mdp_samples, base_mdp:TabularMDP):

    values = []
    al_vars = []

    for dynamic in mdp_samples:

        base_mdp.transition_matrix = dynamic
        moments_calc = MomentsCalculator(base_mdp, policy)
        values.append(moments_calc.get_expected_value())
        al_vars.append(moments_calc.get_variance())
    
    means = np.array(values)
    vars = np.array(al_vars)
    
    avg_value = means.mean(0)
    ep_var = means.var(axis=0, ddof=1)
    al_var = vars.mean(0)

    return avg_value, np.sqrt(ep_var), np.sqrt(al_var), means


def get_run_ep_al_stds(policy, run_name, num_samples, results_dir, save=True):

    data_dir = os.path.join(os.path.dirname(os.path.relpath(__file__)), "data")

    run_args = json.load(open(os.path.join(data_dir, "config.json")))
    counts = np.load(os.path.join(data_dir, "datasets", run_name, "counts.npy"))

    run_args['env_args']["p_successful_step"] = 0
    base_mdp = Gridworld(**run_args['env_args'])
    base_mdp.transition_matrix = None
    pseudocounts = get_dynamics_pseudocounts(counts, base_mdp)
    mdp_samples = get_dynamics_batch_sample(num_samples, pseudocounts, base_mdp)

    val, ep_std, al_std, value_samples = get_value_ep_al_stds(policy, mdp_samples, base_mdp)

    if save:
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        np.save(os.path.join(results_dir,'values'), val)
        np.save(os.path.join(results_dir,'al_std'), al_std)
        np.save(os.path.join(results_dir,'ep_std'), ep_std)
        np.save(os.path.join(results_dir,'num_samples'), num_samples)
        np.save(os.path.join(results_dir,'value_samples'), value_samples)
    else:
        return val, ep_std, al_std

if __name__ == '__main__':

    num_samples = 10000
    
    data_config = json.load(open(os.path.join(os.path.dirname(__file__),"data","config.json")))
    p_succs = data_config["env_args"]["p_successful_step"]
    p_falls = [1-p for p in p_succs]

    for p_fall in p_falls:
        
        data_config["env_args"]["p_successful_step"] = 1-p_fall
        base_mdp = Gridworld(**data_config["env_args"])

        for num_data in data_config["data_args"]["num_transitions"]:
            print(f'analysing dataset with {num_data} transitions')

            run_name = os.path.join(f'p_{1-p_fall}',f'{num_data}')
            results_dir = os.path.join('results','uncertainty_quantification',run_name)

            counts = np.load(os.path.join(os.path.dirname(__file__),"data","datasets",f"p_{1-p_fall}",f"{num_data}","counts.npy"))

            solver = MLESolver(epsilon=1e-10)
            policy = solver.solve(counts, base_mdp)

            get_run_ep_al_stds(policy, run_name, num_samples, results_dir)

