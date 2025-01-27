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
    'Flan-T5': 5.5/6,
}



PLM_NAME = ['GPT-2', 'Llama-2', 'Vicuna', 'OPT', 'ChatGLM3', 'Flan-T5']
VALUES_GOLD_CHANGE = {
    '100': {
        'ACC': [
            [78.284, 72.966, 85.382, 76.562, 80.518], 
            [82.268, 85.772, 83.712, 83.824, 78.952],
            [76.920, 67.568, 71.432, 77.088, 82.756],
            [81.768, 83.856, 80.804, 74.208, 79.084],
            [85.328, 79.140, 85.816, 75.536, 69.600], 
            [85.204, 82.708, 89.004, 85.012, 88.068], 
        ],
        'FID': [
            [22.43714657951997, 22.812377934402644, 22.853686869851185, 22.90162741977168, 23.079607297367936], 
            [19.729484527493128, 22.40769088281461, 22.540429594882017, 22.8081894907551, 23.264911913165932], 
            [21.931871576548698, 22.92326049362733, 24.067964938138253, 24.215835714518324, 24.296321390926735], 
            [22.456248204367157, 22.926147510951353, 24.023683515144604, 24.300333249626448, 24.284272329672447], 
            [25.437763551394568, 26.95029482823163, 26.493643980824466, 27.590336942203372, 27.93594779984015], 
            [16.906484109424525, 13.635706683591973, 11.627216107810888, 11.734712474606116, 11.20402353833581], 
        ],
    },
    '1000': {
        'ACC': [
            [78.284, 86.340, 79.208, 84.484, 82.848], 
            [82.268, 71.492, 86.269, 83.752, 79.780],
            [76.920, 80.556, 83.496, 73.508, 84.928],
            [81.768, 84.396, 79.112, 84.848, 82.612],
            [85.328, 84.415, 78.515, 86.407, 70.783], 
            [85.204, 83.668, 89.240, 88.936, 88.484], 

        ],
        'FID': [
            [19.21506211575234, 17.34142724134457, 16.706371369458324, 16.69703870839927, 16.52391050449615], 
            [18.22529646525797, 18.69838649013951, 19.29963245567658, 18.947078194833807, 19.56124767559635],
            [19.350206326673934, 17.467459586066695, 16.556023818186382, 16.913858991772837, 17.12372088751186],
            [19.375857146589027, 19.511551001511336, 19.57967811663628, 19.80106521593109, 19.884457126063573],
            [22.58043299771937, 21.04768963525317, 19.60750111656867, 19.366560300505803, 19.802844481255963], 
            [14.069160302332563, 10.109686621586501, 7.091305881896059, 5.951271846258183, 5.357336251272768], 
        ],
    },
    '10000': {
        'ACC': [
            [78.284, 83.848, 82.852, 81.088, 86.832], 
            [82.268, 81.668, 74.228, 86.739, 80.428],
            [76.920, 71.288, 76.308, 84.060, 85.134],
            [81.768, 80.340, 85.438, 82.128, 83.232],
            [85.328, 87.152, 76.600, 84.488, 83.748], 
            [85.204, 88.988, 87.956, 87.060, 89.484], 
        ],
        'FID': [
            [18.186061490097792, 15.879831427853986, 15.365043392743118, 15.092013675096863, 14.883593030556938], 
            [17.674561819508686, 18.689392194933873, 18.336329723858917, 17.552555270910133, 17.551138785961292],
            [18.40838855704981, 16.877359164126208, 15.913900741048856, 15.70995338777093, 15.428424154321778],
            [19.095006108724014, 19.069481086615262, 18.32084454704366, 18.759100996753006, 18.892330407029377],
            [21.431109759924297, 16.54545656256558, 15.879928651561428, 16.22561343635718, 16.204030598645915], 
            [13.197600954091634, 9.200959986574686, 6.867603061445065, 5.661966442028136, 5.749677916110375], 
        ],
    },
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

def plot_fid_acc_6models(results, idx_list, method_name):
    plm_names = [PLM_NAME[_i] for _i in idx_list]
    num_colums = ((len(idx_list)+1)//2)
    # fig, axs = plt.subplots(nrows=2, ncols=num_colums, figsize=(18, 6), sharex=False, sharey=False)
    fig, axs = plt.subplots(nrows=2, ncols=num_colums, figsize=(18, 7.1), sharex=False, sharey=False)
    marker_list = ['o','s','^','v','x','H']
    linestyle_list = [':','--','-',':-']
    twin_axis_list = []
    for _method_name, _marker, _linestyle in zip(method_name, marker_list, linestyle_list):
        for _i, (_acc, _fid, _plm) in enumerate(zip(results[_method_name]['ACC'], results[_method_name]['FID'], plm_names)):
            _ax_ix = _i // num_colums
            _ax_iy = _i % num_colums
            x = np.arange(0,len(_acc),1)
            # 绘制第一条线，使用左侧 y 轴
            axs[_ax_ix][_ax_iy].plot(x, _acc, color='#4C8B8F', marker=_marker, linestyle=_linestyle, markersize=6, linewidth=2, label=f'ACC for {_method_name}')
            axs[_ax_ix][_ax_iy].set_xlabel('Iteration', fontsize=20)
            axs[_ax_ix][_ax_iy].set_xticks(x)
            axs[_ax_ix][_ax_iy].set_xticklabels(x, fontsize=16)
            axs[_ax_ix][_ax_iy].set_ylabel('ACC', color='#4C8B8F', fontsize=20)
            axs[_ax_ix][_ax_iy].set_yticklabels(axs[_ax_ix][_ax_iy].get_yticklabels(), fontsize=16)
            axs[_ax_ix][_ax_iy].tick_params(axis='y', labelcolor='#4C8B8F')

            # 创建第二个轴对象，共享 x 轴，使用不同的 y 轴
            if len(twin_axis_list) > _i:
                ax2 = twin_axis_list[_i]
            else:
                ax2 = axs[_ax_ix][_ax_iy].twinx()
                twin_axis_list.append(ax2)
            # 绘制第二条线，使用右侧 y 轴
            ax2.plot(x, _fid, color='#C76248', marker=_marker, linestyle=_linestyle, markersize=6, linewidth=2, label=f'FID for {_method_name}')
            ax2.set_ylabel('FID', color='#C76248', fontsize=20)
            ax2.tick_params(axis='y', labelcolor='#C76248')
            
            axs[_ax_ix][_ax_iy].set_title(_plm, fontsize=21)

        # 显示图例
        lines1, labels1 = axs[-1][-1].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        # axs[0][-1].legend(lines1 + lines2, labels1 + labels2, fontsize=16, bbox_to_anchor=(1.2, 0.85)) #(0,0) overlaps the left-down corner, (1,0) at the right-down corner outside the plot
    fig.legend(lines1 + lines2, labels1 + labels2, fontsize=16, ncol=6, loc='upper center', bbox_to_anchor=(0.5, 0.1)) #(0,0) overlaps the left-down corner, (1,0) at the right-down corner outside the plot

    fig.tight_layout()
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    if not os.path.exists(f'./figure/introduction/'):
        os.makedirs(f'./figure/introduction/')
    print(f'./figure/introduction/gold_change_fid_acc_{method_name}.png')
    plt.savefig(f'./figure/introduction/gold_change_fid_acc_{method_name}.png',dpi=200)


def plot_fid_acc(results, idx_list, method_name):
    plm_names = [PLM_NAME[_i] for _i in idx_list]
    fig, axs = plt.subplots(nrows=1, ncols=len(idx_list), figsize=(18, 3), sharex=False, sharey=False)
    marker_list = ['o','s','^','v','x','H']
    linestyle_list = [':','--','-',':-']
    twin_axis_list = []
    for _method_name, _marker, _linestyle in zip(method_name, marker_list, linestyle_list):
        for _i, (_acc, _fid, _plm) in enumerate(zip(results[_method_name]['ACC'], results[_method_name]['FID'], plm_names)):
        # for _i, (_acc, _fid, _plm) in enumerate(zip(results[_method_name]['ACC'], results[_method_name if 'Contrast' in _method_name else 'Random']['FID'], plm_names)):
            x = np.arange(0,len(_acc),1)
            # 绘制第一条线，使用左侧 y 轴
            axs[_i].plot(x, _acc, color='#4C8B8F', marker=_marker, linestyle=_linestyle, markersize=6, linewidth=2, label=f'ACC for {_method_name}')
            axs[_i].set_xlabel('Iteration', fontsize=20)
            axs[_i].set_xticks(x)
            axs[_i].set_xticklabels(x)
            axs[_i].set_ylabel('ACC', color='#4C8B8F', fontsize=20)
            axs[_i].tick_params(axis='y', labelcolor='#4C8B8F')

            # 创建第二个轴对象，共享 x 轴，使用不同的 y 轴
            if len(twin_axis_list) > _i:
                ax2 = twin_axis_list[_i]
            else:
                ax2 = axs[_i].twinx()
                twin_axis_list.append(ax2)
            # 绘制第二条线，使用右侧 y 轴
            ax2.plot(x, _fid, color='#C76248', marker=_marker, linestyle=_linestyle, markersize=6, linewidth=2, label=f'FID for {_method_name}')
            ax2.set_ylabel('FID', color='#C76248', fontsize=20)
            ax2.tick_params(axis='y', labelcolor='#C76248')
            
            axs[_i].set_title(_plm, fontsize=21)

        # 显示图例
        lines1, labels1 = axs[_i].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        axs[2].legend(lines1 + lines2, labels1 + labels2, bbox_to_anchor=(1.2, 0.85)) #(0,0) overlaps the left-down corner, (1,0) at the right-down corner outside the plot

    fig.tight_layout()
    plt.tight_layout()
    if not os.path.exists(f'./figure/introduction/'):
        os.makedirs(f'./figure/introduction/')
    print(f'./figure/introduction/gold_change_fid_acc_{method_name}.png')
    plt.savefig(f'./figure/introduction/gold_change_fid_acc_{method_name}.png',dpi=200)


def plot_only_acc_increase(type='original'):
    results = {
        'IMDb': [
            [83.976, 85.382, 85.868, 86.340, 86.832], # 'GPT-2'
            [84.516, 85.772, 86.086, 86.269, 86.739],
            [82.140, 82.756, 83.524, 84.928, 85.134],
            [82.777, 83.856, 83.976, 84.848, 85.438],
            [85.302, 85.816, 86.316, 86.807, 87.152],
            [88.362, 89.004, 89.016, 89.240, 89.484], # 'Flan-T5'
            [89.240, 89.520, 89.648, 89.843, 90.042], # 'CopeFuse (Ours)'
        ],
        'yelp-Rating': [
            [44.280, 45.280, 46.250, 46.780, 47.020], # 'GPT-2'
            [45.660, 47.420, 51.113, 52.740, 53.920],
            [52.953, 54.420, 54.760, 55.853, 56.760],
            [50.150, 50.810, 52.440, 54.025, 55.250],
            [54.526, 55.173, 56.940, 58.020, 58.990],
            [56.833, 58.690, 61.040, 61.860, 62.420], # 'Flan-T5'
            [60.680, 61.210, 61.460, 62.273, 62.836], # 'CopeFuse (Ours)' #60.78
        ]
    }

    num_gold_list = {
        'IMDb': [10,100,300,1000,10000], 'yelp-Rating': [25,100,300,1000,10000]
    }
    # method_list = ['GPT-2', 'Llama-2', 'Vicuna', 'OPT', 'ChatGLM3', 'Flan-T5']
    method_list = [r'Aug-PE$_{GPT-2}$', r'Aug-PE$_{Llama-2}$', r'Aug-PE$_{Vicuna}$', r'Aug-PE$_{OPT}$', r'Aug-PE$_{ChatGLM3}$', r'Aug-PE$_{Flan-T5}$', 'WASP (Ours)']

    fig, axs = plt.subplots(nrows=1, ncols=len(list(results.keys())), figsize=(10, 3.7), sharex=False, sharey=False)
    # fig, axs = plt.subplots(nrows=1, ncols=len(list(results.keys())), figsize=(5, 3), sharex=False, sharey=False)
    if len(list(results.keys())) == 1:
        axs = [axs]
    marker_list = ['o','^','v','s','x','H','*']
    linestyle_list = [':','--','-',':-']
    # cmap = pylab.cm.get_cmap('GnBu', 9)
    cmap = plt.get_cmap('GnBu',9)
    color_list = [cmap((_i+1+0.5)/6) for _i in range(len(method_list)-1)] + ['#C76248']
    for i_axis, _task in enumerate(list(results.keys())):
        x = np.arange(0,len(results[_task][0]),1)
        for _method_name, _acc, _color, _marker in zip(method_list, results[_task], color_list, marker_list):
            if i_axis == 0:
                if type == 'original':
                    axs[i_axis].plot(x, _acc, color=_color, marker=_marker, linestyle='-', markersize=7, linewidth=2, label=_method_name)
                elif type == 'difference':
                    axs[i_axis].plot(x, [_acc[_i]-_acc[-1] for _i in range(len(_acc))], color=_color, marker=_marker, linestyle='-', markersize=6, linewidth=2, label=_method_name)
                    # axs[i_axis].plot(x, [_acc[_i]-results[_task][-1][_i] for _i in range(len(_acc))], color=_color, marker=_marker, linestyle='-', markersize=6, linewidth=2, label=_method_name)
            else:
                if type == 'original':
                    axs[i_axis].plot(x, _acc, color=_color, marker=_marker, linestyle='-', markersize=7, linewidth=2)
                elif type == 'difference':
                    axs[i_axis].plot(x, [_acc[_i]-_acc[-1] for _i in range(len(_acc))], color=_color, marker=_marker, linestyle='-', markersize=6, linewidth=2)
                    # axs[i_axis].plot(x, [_acc[_i]-results[_task][-1][_i] for _i in range(len(_acc))], color=_color, marker=_marker, linestyle='-', markersize=6, linewidth=2)

        axs[i_axis].set_xlabel(r'$M$', fontsize=18)
        axs[i_axis].set_xticks(x)
        axs[i_axis].set_xticklabels(num_gold_list[_task], fontsize=14)
        axs[i_axis].set_yticklabels(axs[i_axis].get_yticklabels(), fontsize=14)
        axs[i_axis].set_ylabel('ACC', fontsize=18)
        # axs[i_axis].set_ylabel('ACC Difference', fontsize=20)
    if len(axs) == 1:
        axs[-1].legend(bbox_to_anchor=(1.2, 0.85)) #(0,0) overlaps the left-down corner, (1,0) at the right-down corner outside the plot
    else:
        fig.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, 0.25), fontsize=14)

    fig.tight_layout()
    plt.tight_layout()
    if len(axs) > 1:
        plt.subplots_adjust(bottom=0.4)
    if not os.path.exists(f'./figure/introduction/'):
        os.makedirs(f'./figure/introduction/')
    if type == 'original':
        print(f'./figure/introduction/gold_change_acc_{len(list(results.keys()))}.png')
        plt.savefig(f'./figure/introduction/gold_change_fid_acc_{len(list(results.keys()))}.png',dpi=200)
    elif type == 'difference':
        print(f'./figure/introduction/gold_change_acc_difference_{len(list(results.keys()))}.png')
        plt.savefig(f'./figure/introduction/gold_change_fid_acc_difference_{len(list(results.keys()))}.png',dpi=200)


if __name__ == "__main__":
    # plot_gold_increase(results)

    # plot_fid_acc(VALUES_GOLD_CHANGE, [0,4,5], ['100','1000', '10000'])
    # plot_fid_acc_6models(VALUES_GOLD_CHANGE, [0,1,2,3,4,5], ['100','1000', '10000'])
    plot_only_acc_increase(type='original')
    # plot_only_acc_increase(type='difference')
