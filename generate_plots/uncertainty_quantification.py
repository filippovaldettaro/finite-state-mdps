import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.lines as mlines
    

def plot_uncertainty_quantification(dataset_sizes, p_successes, epistemic_uncertainties, aleatoric_uncertainties, state, create_pdf=True):

    s = state
    #sort with p_rand = 0 as first
    p_falls = np.array([1-p for p in p_successes])
    indeces = np.argsort(p_falls)
    p_falls = [p_falls[i] for i in indeces]
    aleatoric_uncertainties = [aleatoric_uncertainties[i] for i in indeces]
    epistemic_uncertainties = [epistemic_uncertainties[i] for i in indeces]

    fig, ax = plt.subplots(figsize=(5,4.2), nrows=1, ncols=1, sharex=True, sharey=False)

    al_stds = aleatoric_uncertainties
    ep_stds = epistemic_uncertainties
    styles = ['solid','dashed','dotted']

    fsize = 12
    params = {'legend.fontsize': fsize,
            'axes.labelsize': fsize,
            'axes.titlesize':fsize,
            'xtick.labelsize':fsize,
            'ytick.labelsize':fsize}
    plt.rcParams.update(params)

    aleatorics = []
    epistemics = []

    for idx, p_fall in enumerate(p_falls):

        al_stds = np.array(aleatoric_uncertainties[idx])
        ep_stds = np.array(epistemic_uncertainties[idx])
        
        style = styles[idx]
        aleatorics.append(ax.plot(np.log10(np.array(dataset_sizes)), al_stds[:,s], label=f"aleat.", c='r', linestyle=style))
        epistemics.append(ax.plot(np.log10(np.array(dataset_sizes)), ep_stds[:,s], label=f"epist.", c='b', linestyle=style))
        #ax.set_ybound(0,0.5)
        ax.set_xlabel(r"$\log_{10}$ dataset size", fontsize=fsize)
        ax.tick_params(axis='y', which='both', labelleft=True, labelsize=fsize)
        ax.set_ylabel("Standard deviation", fontsize=fsize)

    blank = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)

    legend_handle = [blank, blank, blank, blank,  blank,
                    mlines.Line2D([], [], color='red', linestyle='solid'),
                    mlines.Line2D([], [], color='red', linestyle='dashed'),
                    mlines.Line2D([], [], color='red', linestyle='dotted'), blank,
                    mlines.Line2D([], [], color='blue', linestyle='solid'),
                    mlines.Line2D([], [], color='blue', linestyle='dashed'),
                    mlines.Line2D([], [], color='blue', linestyle='dotted')]

    label_col_1 = [r"$p_{rand}$", r"$0$", r"$0.25$", r"$0.5$"]
    label_empty = [""]

    #order labels for legend
    legend_labels = np.concatenate([label_col_1, ["aleat."], label_empty * 3, ["epist."], label_empty * 3])

    ax.legend(legend_handle, legend_labels, ncol=3, handletextpad = -2, bbox_to_anchor=(0.55, 0.55), columnspacing=0.5)

    fig.subplots_adjust(hspace=0)
    plt.tight_layout()
    if create_pdf:
        if not os.path.isdir("figures"):
            os.mkdir("figures")
        plt.savefig("figures/uncertainty_quantification.pdf")
    plt.show()