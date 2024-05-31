import os
import numpy as np
import matplotlib.pyplot as plt

policy_type = 'mle_policy'
s = 0

p_falls = [0., 0.25, 0.5]
width , height= (5,5)
data_policy_type = 'random'
fig, axes = plt.subplots(figsize=(14,4.5), nrows=1, ncols=len(p_falls), sharex=True, sharey=False)

for run_idx, ax in enumerate(axes):
    p_fall = p_falls[run_idx]
    dataset_sizes = [25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 15000, 20000, 25000]
    vals = []
    al_stds = []
    ep_stds = []

    for num_data in dataset_sizes:

        base_dir = os.path.join('gridworld',f'{width}x{height}',f'p_succ{1-p_fall}',f'{data_policy_type}_{num_data}steps')
        results_analysis_dir = os.path.join('results',base_dir,f'{policy_type}','analysis')

        val = np.load(os.path.join(results_analysis_dir,'values.npy'))
        al_std = np.load(os.path.join(results_analysis_dir,'al_std.npy'))
        ep_std = np.load(os.path.join(results_analysis_dir,'ep_std.npy'))
        num_samples = np.load(os.path.join(results_analysis_dir,'num_samples.npy'))

        vals.append(val)
        al_stds.append(al_std)
        ep_stds.append(ep_std)

    vals = np.array(vals)
    al_stds = np.array(al_stds)
    ep_stds = np.array(ep_stds)

    ax.plot(np.log10(np.array(dataset_sizes)), al_stds[:,s], label=f"aleatoric", c='r')
    ax.plot(np.log10(np.array(dataset_sizes)), ep_stds[:,s], label=f"epistemic", c='b')
    ax.set_ybound(0,0.55)
    ax.set_xlabel(r"$\log_{10}$ dataset size")
    ax.legend()

    if run_idx == 0:
        ax.tick_params(axis='y', which='both', labelleft=True)
        ax.set_ylabel("standard deviation")
    
    plt.savefig("figures/combined_uncertainties")