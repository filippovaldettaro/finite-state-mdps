import os
import numpy as np
import matplotlib.pyplot as plt

def plot_bayesian_value(dataset_sizes, policies, labels_dict, value_samples_dict, state=None):
    
    plt.figure(figsize=(4,3.5))
    c=-1
    colors = ['r', 'b', 'y', 'g', 'c']
    assert(len(policies)<=len(colors))
    for policy_type in policies:
        c+=1
        vals = value_samples_dict[policy_type]

        x = np.log10(np.array(dataset_sizes))
        if state is None:
            y = vals.mean(-1)
        else:
            y = vals[:,state]

        plt.plot(np.log10(np.array(dataset_sizes)), y, label=labels_dict[policy_type], c=colors[c])
    
    plt.legend(fontsize=11)
    plt.xlabel("$\log_{10}$ dataset size", fontsize=12)
    plt.ylabel(f"Value", fontsize=12)
    plt.tight_layout()
    if state is not None:
        plt.savefig(f'figures/value_comparison_state{state}.pdf')
    else:
        plt.savefig(f'figures/value_comparison.pdf')

    plt.show()

def plot_relative_bayesian_value(dataset_sizes, reference_policy, compared_policy, value_samples_dict, num_seeds, scatter=False, state=None):
    
    plt.figure(figsize=(4,3.5))
    
    comp_vals = []
    ref_vals = []
    for seed in range(num_seeds):
        comp_vals.append(value_samples_dict[compared_policy][seed])   
        ref_vals.append(value_samples_dict[reference_policy][seed])
    comp_vals = np.array(comp_vals)
    ref_vals = np.array(ref_vals)
    if state is None:
        comp_vals = comp_vals.mean(-1)
        ref_vals = ref_vals.mean(-1)
    else:
        comp_vals = comp_vals[:,:,:,state]
        ref_vals = ref_vals[:,:,:,state]
    comp_vals = np.array(comp_vals).mean(2)
    ref_vals = np.array(ref_vals).mean(2)
    rel_vals =  (ref_vals - comp_vals)

    x = np.log10(np.array(dataset_sizes))

    y = rel_vals.mean(0)
    std = rel_vals.std(0,ddof=1)


    plt.hlines(0, x[0], x[-1], color='r', linestyles='dashed')
    plt.plot(x, y, color='b')


    if scatter:
        for seed in range(num_seeds):
            plt.scatter(x, rel_vals[seed], c='b', alpha=0.2)

    else:
        plt.fill_between(np.log10(np.array(dataset_sizes)), y+std, y-std, color='b', alpha = 0.25)
    
    plt.xlabel("$\log_{10}$ dataset size", fontsize=12)
    plt.ylabel(f"value difference", fontsize=12)
    plt.tight_layout()

    plt.savefig(f'figures/relative_comparison.pdf')
    plt.show()