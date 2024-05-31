import os
import json

import numpy as np
from utils.posterior import get_dynamics_pseudocounts, get_dynamics_batch_sample
from utils.moments import MomentsCalculator
from environments.gridworld import Gridworld

if __name__ == "__main__":

    eval_args = {
        "policies":["mle", "nominal", "second_order", "grad_stochastic"], #include "msbi" for appendix results
        "num_samples":10000,
        "eval_seed":0
    }

    np.random.seed(eval_args["eval_seed"])
    current_dir = os.path.dirname(os.path.join(os.path.relpath(__file__)))
    data_config = json.load(open(os.path.join(current_dir,"data","config.json"))) 
    data_args = data_config["data_args"]
    base_mdp = Gridworld(**data_config["env_args"])

    datasets_dir = os.path.join(current_dir,"data","datasets")
    policies_dir = os.path.join("results","policies")
    dataset_sizes = os.listdir(datasets_dir)
    for dataset_size in data_args["num_transitions"]:
        dataset_size = str(dataset_size)
        print(f"evaluating policies' Bayesian objective on dataset of size {dataset_size}")
        for seed in range(data_args["num_seeds"]):
            print(f"seed {seed} of {data_args['num_seeds']}")
            counts = np.load(os.path.join(datasets_dir,dataset_size, str(seed),"counts.npy"))
            pseudocounts = get_dynamics_pseudocounts(counts, base_mdp)
            dynamics_samples = get_dynamics_batch_sample(eval_args["num_samples"], pseudocounts, base_mdp)

            #evaluate all policies on the same posterior samples

            for policy_type in eval_args["policies"]:

                policy = np.load(os.path.join(policies_dir, policy_type, dataset_size, str(seed), "policy.npy"), allow_pickle=True)
               
                values = []
                
                for sample in range(eval_args["num_samples"]):

                    dynamics_sample = dynamics_samples[sample]
                    base_mdp.transition_matrix = dynamics_sample
                    calc = MomentsCalculator(base_mdp, policy)
                    values.append(calc.get_expected_value())
                
                values = np.array(values)

                np.save(os.path.join(policies_dir,policy_type,dataset_size,str(seed),f"value{eval_args['num_samples']}samples"), values)

    
