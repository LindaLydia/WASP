import sys, os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import pylab


results = {
    'IMDb': {
        'GPT-2': [85.382, 86.34, 86.832],
        'ChatGLM3': [85.816, 86.407, 87.152],
        'Flan-T5': [89.004, 89.24, 89.484],
    },
    'Yelp-Rating': {
        'GPT-2': [45.28, 46.78, 48.02],
        'ChatGLM3': [55.173, 57.46, 61.88],
        'Flan-T5': [58.69, 59.86, 62.42],
    },
}

colormap_position_dict = {
    'GPT-2': 0.5/6,
    'Llama-2': 1.5/6,
    'Vicuna': 2.5/6,
    'OPT': 3.5/6,
    'ChatGLM3': 4.5/6,
    'Flan-T5': 5.6/6,
}


def plot_gold_increase(results):
    fig, axs = plt.subplots(nrows=1, ncols=len(list(results.keys())), figsize=(10, 3), sharex=False, sharey=False)
    cmap = plt.cm.viridis
    cmap = plt.cm.BrBG
    for _i, _task in enumerate(results.keys()):
        for _plm in results[_task]:
            x = [0,1,2]
            axs[_i].plot(x, results[_task][_plm], marker='s', markersize=6, linewidth=4, color=cmap(colormap_position_dict[_plm]), label=_plm)
        x_ticks = [0,1,2]
        x_tick_labels = [100,1000,10000]
        axs[_i].set_xticks(x_ticks)
        axs[_i].set_xticklabels(x_tick_labels,fontsize=15) # 
        axs[_i].set_xlabel('Number of Private Samples',fontsize=17) #
        axs[_i].set_ylabel('ACC',fontsize=17) #

        axs[_i].legend() #loc='upper center'
            
        axs[_i].set_title(_task, fontsize=21)

    fig.tight_layout()
    plt.tight_layout()
    if not os.path.exists(f'./figure/introduction/'):
        os.makedirs(f'./figure/introduction/')
    print(f'./figure/introduction/change_of_gold.png')
    plt.savefig(f'./figure/introduction/increase_of_gold.png',dpi=200)


    # # Create positions for the bars
    # x = np.arange(3)  # Base positions for settings
    # # width = 0.2  # Width of each bar
    # width = 0.12
    # # offset = [width*(k+0.5) for k in range(-3,3,1)]

    # # Create the plot
    # fig, ax = plt.subplots(figsize=(10, 3))

    # for i, (model, values) in enumerate(results['IMDb'].items()):
    #     ax.bar(x + i * width, values, width, label=model, color=cmap(colormap_position_dict[model]))

    # # Add labels, title, and legend
    # ax.set_xticks(x + width)
    # ax.set_xticklabels([10,1000,10000])
    # ax.set_ylabel("Performance")
    # ax.set_title("Model Performance Under Different Settings")
    # ax.legend()
    # plt.tight_layout()
    # if not os.path.exists(f'./figure/introduction/'):
    #     os.makedirs(f'./figure/introduction/')
    # print(f'./figure/introduction/temp.png')
    # plt.savefig(f'./figure/introduction/temp.png',dpi=200)



if __name__ == "__main__":
    plot_gold_increase(results)
