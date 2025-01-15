import sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib_venn import venn3
import seaborn as sns
import pandas as pd
import pylab

def plot_3_circle_heatmap():
    # 61.22	62.89	63.04
    # 	61.52	62.92
    # 		62.82

    # Define the results for each region
    # Order of regions: (A, B, AB, C, AC, BC, ABC), A = GPT-3.5, B = GPT-4, C = GPT-4o
    results = {
        '100': 61.22,  # Only A
        '010': 61.52,  # Only B
        '110': 62.89,   # A ∩ B
        '001': 62.82,  # Only C
        '101': 63.04,   # A ∩ C
        '011': 62.92,  # B ∩ C
        '111': 64.48    # A ∩ B ∩ C
    }

    # Create a Venn diagram
    venn = venn3(subsets=(results['100'], results['010'], results['110'], 
                        results['001'], results['101'], results['011'], results['111']),
                #  set_labels=('A', 'B', 'C'))
                set_labels=('GPT-3.5', 'GPT-4', 'GPT-4o'))

    # Annotate each section with the corresponding value
    for region, value in results.items():
        venn.get_label_by_id(region).set_text(f'{value}')
        label = venn.get_label_by_id(region)
        if label:  # Only update non-empty regions
            label.set_fontsize(15)  # Adjust region annotation font size
    for label in venn.set_labels:
        if label:  # Only update non-empty labels
            label.set_fontsize(24)  # Adjust set label font size

    # Normalize the values to [0, 1] for color mapping
    values = np.array(list(results.values()))
    norm_values = ((values - values.min()) / (values.max() - values.min()))*0.5+0.25
    # Use a colormap (e.g., 'viridis')
    for cmap_name in ['viridis', 'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']:
        colormap = cm.get_cmap(cmap_name)
        # Set colors for each region based on normalized values
        for region, value, norm_value in zip(results.keys(), results.values(), norm_values):
            patch = venn.get_patch_by_id(region)
            if patch:  # Only update non-empty regions
                patch.set_color(colormap(norm_value))
                patch.set_alpha(0.9)  # Set transparency

        # # Annotate the regions with the values
        # for region, value in results.items():
        #     label = venn.get_label_by_id(region)
        #     if label:  # Only update non-empty regions
        #         label.set_text(f'{value}')

        # # Add a color bar for reference
        # sm = cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=values.min(), vmax=values.max()))
        # sm.set_array([])
        # cbar = plt.colorbar(sm)
        # cbar.set_label('Value')

        # # Set title
        # plt.title("Venn Diagram with Annotations")

        # Show plot
        plt.tight_layout()
        if not os.path.exists('./figure/increas_k/'):
            os.makedirs('./figure/increas_k/')
        print(f'./figure/increas_k/closedPLM_yelpRating_{cmap_name}.png')
        plt.savefig(f'./figure/increas_k/closedPLM_yelpRating_{cmap_name}.png',dpi=200)


def pair_wise_collaboration_heatmap():


    # pair_wise_acc = [
    #     [76.40, 76.82, 76.86, 76.06, 76.25, 77.17, 77.59, np.nan],
    #     [77.26, 77.48, 77.23, 77.18, 77.52, 77.41, np.nan, np.nan],
    #     [73.65, 74.92, 74.85, 73.80, 74.68, np.nan, np.nan, np.nan],
    #     [65.80, 73.25, 75.17, 65.42, np.nan, np.nan, np.nan, np.nan],
    #     [65.03, 73.26, 73.93, np.nan, np.nan, np.nan, np.nan, np.nan],
    #     [74.53, 73.79, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    #     [73.57, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    #     [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    # ]
    # single_acc = [
    #     [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 75.97],
    #     [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 77.11, np.nan],
    #     [np.nan, np.nan, np.nan, np.nan, np.nan, 73.60, np.nan, np.nan],
    #     [np.nan, np.nan, np.nan, np.nan, 64.93, np.nan, np.nan, np.nan],
    #     [np.nan, np.nan, np.nan, 59.03, np.nan, np.nan, np.nan, np.nan],
    #     [np.nan, np.nan, 73.34, np.nan, np.nan, np.nan, np.nan, np.nan],
    #     [np.nan, 73.22, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    #     [64.53, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    # ]
    # mixed_acc = [
    #     [76.40, 76.82, 76.86, 76.06, 76.25, 77.17, 77.59, 75.97],
    #     [77.26, 77.48, 77.23, 77.18, 77.52, 77.41, 77.11, np.nan],
    #     [73.65, 74.92, 74.85, 73.80, 74.68, 73.60, np.nan, np.nan],
    #     [65.80, 73.25, 75.17, 65.42, 64.93, np.nan, np.nan, np.nan],
    #     [65.03, 73.26, 73.93, 59.03, np.nan, np.nan, np.nan, np.nan],
    #     [74.53, 73.79, 73.34, np.nan, np.nan, np.nan, np.nan, np.nan],
    #     [73.57, 73.22, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    #     [64.53, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    # ]
    # df_2plm = pd.DataFrame(pair_wise_acc, columns=['GPT-2', 'Llama-2', 'Vicuna', 'OPT', 'ChatGLM3', 'Flan-T5', 'GPT-3.5', 'GPT-4'])
    # df_1plm = pd.DataFrame(single_acc, columns=['GPT-2', 'Llama-2', 'Vicuna', 'OPT', 'ChatGLM3', 'Flan-T5', 'GPT-3.5', 'GPT-4'])
    # df_mix = pd.DataFrame(mixed_acc, columns=['GPT-2', 'Llama-2', 'Vicuna', 'OPT', 'ChatGLM3', 'Flan-T5', 'GPT-3.5', 'GPT-4'])

    # df_2plm.index = ['GPT-4', 'GPT-3.5', 'Flan-T5', 'ChatGLM3', 'OPT', 'Vicuna', 'Llama-2', 'GPT-2']
    # df_1plm.index = ['GPT-4', 'GPT-3.5', 'Flan-T5', 'ChatGLM3', 'OPT', 'Vicuna', 'Llama-2', 'GPT-2']
    # df_mix.index = ['GPT-4', 'GPT-3.5', 'Flan-T5', 'ChatGLM3', 'OPT', 'Vicuna', 'Llama-2', 'GPT-2']

    mixed_acc = [
        [63.04, 62.92, 62.82],
        [62.89, 61.52, np.nan],
        [61.22, np.nan, np.nan],
    ]
    _min, _max = 61.22, 63.04
    # for _list in mixed_acc:
    #     for _value in _list:
    #         if _value != np.nan:
    #             _value = ((_value-_min) / (_max-_min))
    df_mix = pd.DataFrame(mixed_acc, columns=['GPT-3.5', 'GPT-4', 'GPT-4o'])
    df_mix.index = ['GPT-4o', 'GPT-4', 'GPT-3.5']

    for cmap_name in ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']:
        fig, axes = plt.subplots(1, 1, figsize=(3,3))
            # Plot the first heatmap
        sns.heatmap(df_mix, ax=axes, cmap=cmap_name, annot=True, fmt=".2f", cbar=True, linewidth=.5, vmin=_min-0.2*(_max-_min), vmax=_max+0.2*(_max-_min))
        
        plt.tight_layout()
        plt.savefig(f"./figure/pair_wise/single_and_pairwise_{cmap_name}.png", dpi=200)


def plot_vary_K_with_errorbar():
    results = [61.85333333, 62.95, 64.48]
    error_bar = np.asarray([0.850490055, 0.079372539, 0.0])
    NUM_MODELS = 3
    x_tick_labels = [i for i in range(1,NUM_MODELS+1)]

    results = np.asarray(results)
    assert results.shape[0] == len(x_tick_labels), f"{results.shape[0]=}, {len(x_tick_labels)=}"

    x = np.arange(0,len(x_tick_labels))
    width = 0.12
    offset = [width*(k+0.5) for k in range(-3,3,1)]
    print(offset)

    fig, axes = plt.subplots(1, 1, figsize=(4.5,4))
    # print(f"{cmap=}")
    # print(f"{cmap.colors=}")

    axes.plot(x, results, marker='o', markersize=12, color='darkblue', label='average') # , label=r'$\tilde{m}$'
    axes.fill_between(x, results - error_bar, results + error_bar, color='yellowgreen', alpha=0.5)
    axes.set_xlabel(r"$K$", fontsize=24)
    # axes.set_xticklabels(axes.get_xticks(), fontsize=20)
    axes.set_xlim(1-0.1, len(x_tick_labels)+0.1-1)
    axes.set_xticks(x, x_tick_labels, fontsize=20)

    axes.set_ylim([60.5,65])
    axes.set_ylabel(r'$\tilde{m}$ Performance',fontsize=24)
    y_ticks = np.arange(60.5, 64.6, 1)
    y_tick_labels = [f'{tick:.0f}' for tick in y_ticks]
    axes.set_yticks(y_ticks)
    axes.set_yticklabels(y_tick_labels,fontsize=20)

    plt.tight_layout()
    plt.savefig(f'./figure/increas_k/qnli_bert_K_onlyFuse_withErrorbar.png', dpi=300)

    # # Draw the line with error bars
    # plt.errorbar(x, y, yerr=error, fmt='-o', label='Data with Error Bars', color='blue', 
    #             ecolor='lightblue', elinewidth=2, capsize=4)



if __name__ == '__main__':
    # plot_3_circle_heatmap()
    # pair_wise_collaboration_heatmap()
    plot_vary_K_with_errorbar()
