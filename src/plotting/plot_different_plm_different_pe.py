import sys, os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import pylab


results = [
    [85.382, 85.772, 82.756, 83.856, 85.816, 89.004], #IMDb
    [45.28, 47.42, 54.42, 50.81, 55.173, 58.69], # Yelp-Rating
    [31.451, 32.666, 32.273, 34.644, 33.81, 34.543], # Openreview-Area
    [75.625, 84.625, 86.75, 79.25, 88.5, 81.25], # Banking
]




def plot_vary_J(results, ncol=2, nrow=2, fig_size=(8,4)):
    NUM_MODELS = 6
    
    # y_tick_labels = [0.0, 0.25, 00.5, 0.75, 1.0]

    results = np.asarray(results)

    x = np.arange(0, NUM_MODELS)
    width = 0.12
    offset = [width*(k+0.5) for k in range(-3,3,1)]
    print(offset)
    
    legend_list = ['GPT2', 'Llama-2', 'Vicuna', 'OPT', 'ChatGLM', 'Flan-T5'] #, 'mixed', 'FuseGen'
    # legend_list = [r'$m_{GPT2}$', r'$m_{Llama-2}$', r'$m_{Vicuna}$', r'$m_{OPT}$', r'$m_{ChatGLM3}$', r'$m_{Flan-T5}$', 'mixed']
    hatch_list = ['xxxx']*(NUM_MODELS-1)+['///']
    alpha_list = [0.5]*(NUM_MODELS-1)+[0.2,0.2]
    if len(results) == 4:
        fig_labels = ['IMDb', 'Yelp-Rating', 'Openreview-Category', 'Banking']
        value_limt_list = [[82,90],[42,60],[30.5,35],[72,90]]
        x_tick_list = [np.arange(82,90.01,2), np.arange(42,60.01,4), np.arange(31,35.1,1), np.arange(72,90.01,4)]
    else:
        fig_labels = ['IMDb', 'QNLI']
        # value_limt_list = [[78,90.5],[54,76]]
        # x_tick_list = [np.arange(78,90.01,3), np.arange(55,76.01,5)]
    cmap = pylab.cm.get_cmap('seismic', 6) #'PiYG', 'twilight
    cmap = pylab.cm.get_cmap('YlGnBu', 6) #'PiYG', 'twilight
    cmap = pylab.cm.get_cmap('BrBG', 6) #'PiYG', 'twilight
    # cmap = pylab.cm.get_cmap('coolwarm', 6) #'coolwarm', 'twilight
    # cmap = pylab.cm.get_cmap('RdBu', 6) #'coolwarm', 'twilight
    # print(f"{cmap=}")
    # print(f"{cmap.colors=}")

    fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=fig_size)
    for i, ax in enumerate(axes.flat):
        # ax.set(xticks=[],yticks=[])
        # s='subplot(2,2)'+str(i)+')'
        # ax.text(0.5,0.5,s,ha='center',va='center',size=20,alpha=0.5)
        ax.invert_yaxis()
        print(f"{x=}, {results[i]=}, {alpha_list=}, {hatch_list=}")
        for _x, _data, _alpha, _hatch in zip(x, results[i], alpha_list, hatch_list):
            if _x <= 5:
                ax.barh(_x, _data, color=cmap((_x+0.5)/6)) # , , alpha=_alpha, hatch=_hatch
            elif _x == 6:
                ax.barh(_x, _data, color='grey', alpha=_alpha, edgecolor="k", hatch='++') # , hatch=hatch_list
            elif _x == 7:
                ax.barh(_x, _data, color='orange', alpha=_alpha, edgecolor="k", hatch='xxx') # , hatch=hatch_list
            # ax.barh(_x, _data, color='white', alpha=1.0, edgecolor="k", hatch=_hatch) # , hatch=hatch_list
        # ax.barh(x, results[i], color='orange', alpha=0.5, ) # , hatch=hatch_list
        # # ax.axvline(x=results[i][-2], color='grey', linestyle='--', linewidth=2)
        # # ax.axvline(x=results[i][-1], color='orange', linestyle='--', linewidth=2)
        ax.set_xlim(value_limt_list[i])
        x_ticks = x_tick_list[i]
        x_tick_labels = [f'{tick:.0f}' for tick in x_ticks]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_tick_labels,fontsize=15)
        ax.set_ylabel(fig_labels[i] ,fontsize=17)
        ax.set_yticks(x, legend_list, fontsize=15)

    # handles, labels = axes[0, 0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, fontsize=12)
    plt.tight_layout()
    if not os.path.exists(f'./figure/introduction/'):
        os.makedirs(f'./figure/introduction/')
    print(f'./figure/introduction/fig_different_plm_different_pe_{ncol}.png')
    plt.savefig(f'./figure/introduction/fig_different_plm_different_pe_{ncol}.png', dpi=300)
    return


if __name__ == '__main__':
    # plot_vary_J(results, nrow=2, ncol=2, fig_size=(8,4))
    # plot_vary_J(results, nrow=4, ncol=1, fig_size=(6,12))

    plot_vary_J(results, nrow=2, ncol=2, fig_size=(12,5))
    plot_vary_J(results, nrow=1, ncol=4, fig_size=(20,3))
    # plot_vary_J([results[0],results[-1]], nrow=1, ncol=2, fig_size=(8,2.5))
    # plot_vary_J([results[0],results[-1]], nrow=1, ncol=2, fig_size=(8,2.5))
