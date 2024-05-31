import os
import numpy as np
import matplotlib.pyplot as plt

policy_type = 'mle_policy'

p_fall = 0.
width , height= (5,5)
data_policy_type = 'random'

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


s=0
plt.plot(np.log10(np.array(dataset_sizes)), al_stds[:,s], label=f"aleatoric", c='r')
plt.plot(np.log10(np.array(dataset_sizes)), ep_stds[:,s], label=f"epistemic", c='b')

plt.title('$p_{fall}$'+ f' = {p_fall}')
plt.legend()
plt.xlabel("$\log_{10}$ dataset size")
plt.ylabel("standard deviation")
plt.savefig(f'figures/al_ep_stds_{policy_type}_state{s}')
plt.show()

plt.plot(np.log10(np.array(dataset_sizes)), vals[:,s], label="values")
plt.plot(np.log10(np.array(dataset_sizes)), ep_stds[:,s], label="epistemic")
plt.savefig(f'figures/value_vs_ep_{policy_type}_state{s}')
plt.legend()
plt.show()