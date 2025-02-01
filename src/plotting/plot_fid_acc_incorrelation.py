import sys, os
import matplotlib.pyplot as plt
import numpy as np
import torch
import re


PLM_NAME = ['GPT-2', 'Llama-2', 'Vicuna', 'OPT', 'ChatGLM3', 'Flan-T5']



# VALUES = {
#     "Random": { # w/o dp
#         "ACC": [
#             [78.28, 85.00, 82.25, 67.74, 73.30],
#             [83.81, 69.09, 65.80, 68.89, 67.82],
#             [71.74, 70.52, 75.74, 67.34, 78.82],
#             [77.95, 67.43, 73.92, 74.40, 61.96],
#             [85.33, 68.24, 63.34, 76.10, 69.36],
#             [85.20, 86.56, 89.12, 88.83, 89.34],
#         ],
#         "FID": [
#             [22.26, 21.95, 21.91, 20.96, 20.37],
#             [19.61, 18.79, 19.12, 19.65, 19.33],
#             [21.80, 18.80, 18.52, 18.34, 18.15],
#             [22.22, 19.76, 19.43, 19.07, 18.32],
#             [25.44, 25.68, 21.68, 21.25, 19.70],
#             [16.78, 10.95, 11.62, 12.01, 10.71],
#         ],
#     },
#     "Contrast": { # w/o dp
#         "ACC" : [
#             [78.284, 76.192, 83.388, 86.008, 82.980],
#             [83.812, 75.684, 85.776, 79.308, 85.392],
#             [71.736, 82.708, 70.928, 67.864, 65.012],
#             [77.948, 84.800, 79.872, 73.744, 82.056],
#             [85.328, 80.656, 87.788, 85.228, 84.886],
#             [85.204, 87.884, 89.240, 89.712, 89.568],
#         ],
#         "FID": [
#             [22.257, 23.547, 22.320, 23.145, 23.162],
#             [19.613, 19.196, 19.659, 19.811, 20.656],
#             [21.798, 18.108, 18.590, 19.322, 19.595],
#             [22.222, 21.237, 23.124, 21.042, 22.030],
#             [25.570, 19.038, 19.055, 19.414, 19.772],
#             [16.775, 12.228, 9.843, 12.513, 13.360],
#         ],
#     },
#     "PE": { # w/ (4.0,0.00001)-dp
#         "ACC": [
#             [78.284, 72.966, 85.382, 76.562, 80.518],
#             [83.812, 85.772, 83.712, 83.824, 79.952],
#             [71.736, 54.568, 71.432, 71.088, 82.756],
#             [77.948, 83.856, 80.804, 74.208, 79.084],
#             [85.328, 79.140, 85.816, 75.536, 69.600],
#             [85.204, 82.708, 89.004, 85.012, 88.068],
#         ],
#         "FID": [
#             [22.25714657951997, 23.65476266724693, 23.345218606266624, 23.43878093924372, 22.193230920783076],
#             [19.729484527493128, 26.248656432769906, 23.48684392977075, 24.712944037597367, 25.90246322195742],
#             [21.79829522218952, 24.668823271349353, 27.63649576704712, 25.273630784834914, 25.14140846094064],
#             [22.222441400485845, 24.35987014712511, 26.65422833188071, 25.61181546478432, 24.484749600360523],
#             [25.437763551394568, 32.85896653794217, 26.44527855966699, 34.25698116679615, 30.203017016262024],
#             [16.906484109424525, 12.354713390166237, 9.935373880591039, 14.558540749072947, 12.150284667994525],
#         ]
#     }
# }    


VALUES = { # w/ (4.0,0.00001)-DP
    "Random": {
        "ACC": [
            [78.284, 85.528, 73.836, 76.580, 80.764], 
            [83.812, 86.122, 79.580, 81.472, 83.744], 
            [65.808, 64.808, 68.696, 84.148, 83.004], 
            [77.948, 80.518, 84.042, 78.580, 80.174], 
            [85.328, 86.052, 76.832, 74.088, 82.504], 
            [85.204, 88.780, 89.132, 88.852, 89.012], 
        ],
        "FID": [
            [22.43714657951997, 22.826883804872015, 23.0419993175393, 23.29983505562805, 23.28258944668569], 
            [19.729484527493128, 19.724977052133703, 19.519466064425757, 19.545846352750107, 19.439081890628916], 
            [21.931871576548698, 19.573843312881426, 18.988329334908578, 18.680662374991382, 18.563928324064584], 
            [22.456248204367157, 21.875296904671742, 22.440241514141885, 22.140476791025158, 22.025921513056332], 
            [25.63754670875445, 25.626608774181342, 24.918496744080052, 25.487059252006205, 25.155660979782255], 
            [16.906484109424525, 13.86872382903316, 13.56413167232235, 12.610520907841519, 12.369933912754409], 
        ],
    },
    "Contrast": {
        "ACC" : [
            [78.284, 83.424, 85.808, 81.720, 82.732], 
            [83.812, 86.566, 84.732, 83.632, 84.744], 
            [65.808, 70.144, 73.140, 84.840, 83.792], 
            [77.948, 85.792, 79.625, 85.264, 86.134], 
            [85.328, 81.112, 80.804, 74.220, 86.124], 
            [85.204, 88.680, 87.364, 89.312, 89.232], 
        ],
        "FID": [
            [22.43714657951997, 22.751364945686433, 22.957112797635827, 23.03942715910797, 23.078420607683542], 
            [19.729484527493128, 18.737610770622332, 18.796091616271827, 18.71129011109948, 18.866331049886384], 
            [21.931871576548698, 19.44942521673634, 18.83174317296582, 18.561065908776428, 18.525854801587755], 
            [22.456248204367157, 21.680258138204785, 21.69233636804768, 21.516660949538498, 21.60972151315302], 
            [25.63754670875445, 22.666723597937455, 21.75791130672517, 21.55982788311426, 21.155322419425286], 
            [16.906484109424525, 13.033388492994318, 12.598446492408073, 12.636052316982383, 12.762672722149446], 
        ],
    },
    "PE": {
        "ACC": [
            [78.284, 72.966, 85.382, 76.562, 80.518], 
            [83.812, 85.772, 83.712, 83.824, 79.952], 
            [65.808, 60.496, 71.432, 77.088, 82.756], 
            [77.948, 83.856, 80.804, 74.208, 79.084], 
            [85.328, 79.140, 85.816, 75.536, 69.600], 
            [85.204, 82.708, 89.004, 85.012, 88.068], 
        ],
        "FID": [
            [22.43714657951997, 22.812377934402644, 22.853686869851185, 22.90162741977168, 23.079607297367936], 
            [19.729484527493128, 22.40769088281461, 22.540429594882017, 22.8081894907551, 23.264911913165932], 
            [21.931871576548698, 22.92326049362733, 24.067964938138253, 24.215835714518324, 24.296321390926735], 
            [22.456248204367157, 22.926147510951353, 24.023683515144604, 24.300333249626448, 24.284272329672447], 
            [25.63754670875445, 26.95029482823163, 26.493643980824466, 27.590336942203372, 27.93594779984015], 
            [16.906484109424525, 13.635706683591973, 11.627216107810888, 11.734712474606116, 11.20402353833581], 
        ]
    },
    "Ours": {
        'FID': [
            [17.23471867932253, 13.538982866877067, 12.447013296167533, 12.449327769732967, 11.016763186460742],
        ],
    },
}


def plot_fid_acc_incorrelation(acc_results, fid_results, plm_names, idx_list, method_name):
    fig, axs = plt.subplots(nrows=1, ncols=len(plm_names), figsize=(10, 3), sharex=False, sharey=False)
    for _i, (_acc, _fid, _plm) in enumerate(zip(acc_results, fid_results, plm_names)):
        x = np.arange(0,len(_acc),1)
        # 绘制第一条线，使用左侧 y 轴
        axs[_i].plot(x, _acc, color='#4C8B8F', marker='o', linestyle='-', markersize=6, linewidth=4, label='ACC')
        axs[_i].set_xlabel('Iteration', fontsize=20)
        axs[_i].set_ylabel('ACC', color='#4C8B8F', fontsize=20)
        axs[_i].tick_params(axis='y', labelcolor='#4C8B8F')

        # 创建第二个轴对象，共享 x 轴，使用不同的 y 轴
        ax2 = axs[_i].twinx()
        # 绘制第二条线，使用右侧 y 轴
        ax2.plot(x, _fid, color='#C76248', marker='s', linestyle=':', markersize=6, linewidth=4, label='FID')
        ax2.set_ylabel('FID', color='#C76248', fontsize=20)
        ax2.tick_params(axis='y', labelcolor='#C76248')

        axs[_i].set_xlim([0,4])
        x_ticks = [0,1,2,3,4]
        x_tick_labels = [f'{tick:.0f}' for tick in x_ticks]

        # 显示图例
        if _plm == 'Vicuna':
            lines1, labels1 = axs[_i].get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            # axs[_i].legend(lines1 + lines2, labels1 + labels2, loc='best')
            axs[_i].legend(lines1 + lines2, labels1 + labels2, loc='upper center')
            

        axs[_i].set_title(_plm, fontsize=21)

    fig.tight_layout()
    plt.tight_layout()
    if not os.path.exists(f'./figure/introduction/'):
        os.makedirs(f'./figure/introduction/')
    print(f'./figure/introduction/fid_acc_incorrelate_{method_name}.png')
    plt.savefig(f'./figure/introduction/fid_acc_incorrelate_{method_name}.png',dpi=200)


def plot_fid_acc_incorrelation_all(acc_results, fid_results, plm_names, idx_list, method_name):
    fig, axs = plt.subplots(nrows=2, ncols=(len(plm_names)+1)//2, figsize=(16, 6), sharex=False, sharey=False)
    for _i, (_acc, _fid, _plm) in enumerate(zip(acc_results, fid_results, plm_names)):
        _a_x = _i // ((len(plm_names)+1)//2)
        _a_y =  _i % ((len(plm_names)+1)//2)
        x = np.arange(0,len(_acc),1)
        # 绘制第一条线，使用左侧 y 轴
        axs[_a_x][_a_y].plot(x, _acc, color='#4C8B8F', marker='o', linestyle='--', markersize=6, linewidth=4, label='ACC')
        axs[_a_x][_a_y].set_xlabel('Iteration', fontsize=20)
        axs[_a_x][_a_y].set_ylabel('ACC', color='#4C8B8F', fontsize=20)
        axs[_a_x][_a_y].tick_params(axis='y', labelcolor='#4C8B8F')

        axs[_a_x][_a_y].set_xticks([0,1,2,3,4])
        axs[_a_x][_a_y].set_xticklabels([0,1,2,3,4])

        # 创建第二个轴对象，共享 x 轴，使用不同的 y 轴
        ax2 = axs[_a_x][_a_y].twinx()
        # 绘制第二条线，使用右侧 y 轴
        ax2.plot(x, _fid, color='#C76248', marker='^', linestyle=':', markersize=6, linewidth=4, label='FID')
        ax2.set_ylabel('FID', color='#C76248', fontsize=20)
        ax2.tick_params(axis='y', labelcolor='#C76248')

        # 显示图例
        lines1, labels1 = axs[_a_x][_a_y].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        axs[_a_x][_a_y].legend(lines1 + lines2, labels1 + labels2, loc='best')

        axs[_a_x][_a_y].set_title(_plm, fontsize=21)

    fig.tight_layout()
    plt.tight_layout()
    if not os.path.exists(f'./figure/introduction/'):
        os.makedirs(f'./figure/introduction/')
    print(f'./figure/introduction/fid_acc_incorrelate_all_{method_name}.png')
    plt.savefig(f'./figure/introduction/fid_acc_incorrelate_all_{method_name}.png',dpi=200)


def plot_fid_acc_incorrelation_compare(results, plm_names, idx_list, method_name):
    fig, axs = plt.subplots(nrows=2, ncols=(len(plm_names)+1)//2, figsize=(16, 6), sharex=False, sharey=False)
    marker_list = ['o','s','^','v','x','H']
    linestyle_list = [':','-','--',':-']
    twin_axis_list = []
    LEGEND_MAPPING = {'PE': 'Aug-PE', 'Random': 'w/o Con', 'Contrast': 'w/ Con'}
    for _method_name, _marker, _linestyle in zip(method_name, marker_list, linestyle_list):
        for _i, (_acc, _fid, _plm) in enumerate(zip(results[_method_name]['ACC'], results[_method_name]['FID'], plm_names)):
        # for _i, (_acc, _fid, _plm) in enumerate(zip(results[_method_name]['ACC'], results[_method_name if 'Contrast' in _method_name else 'Random']['FID'], plm_names)):
            _a_x = _i // ((len(plm_names)+1)//2)
            _a_y =  _i % ((len(plm_names)+1)//2)
            x = np.arange(0,len(_acc),1)
            # 绘制第一条线，使用左侧 y 轴
            axs[_a_x][_a_y].plot(x, _acc, color='#4C8B8F', marker=_marker, linestyle=_linestyle, markersize=6, linewidth=2, label=f'{LEGEND_MAPPING[_method_name]}')
            axs[_a_x][_a_y].set_xlabel('Iteration', fontsize=20)
            axs[_a_x][_a_y].set_xticks(x)
            axs[_a_x][_a_y].set_xticklabels(x)
            axs[_a_x][_a_y].set_ylabel('ACC', color='#4C8B8F', fontsize=20)
            axs[_a_x][_a_y].tick_params(axis='y', labelcolor='#4C8B8F')

            # 创建第二个轴对象，共享 x 轴，使用不同的 y 轴
            if len(twin_axis_list) > _i:
                ax2 = twin_axis_list[_i]
            else:
                ax2 = axs[_a_x][_a_y].twinx()
                twin_axis_list.append(ax2)
            # 绘制第二条线，使用右侧 y 轴
            ax2.plot(x, _fid, color='#C76248', marker=_marker, linestyle=_linestyle, markersize=6, linewidth=2, label=f'{LEGEND_MAPPING[_method_name]}')
            ax2.set_ylabel('FID', color='#C76248', fontsize=20)
            ax2.tick_params(axis='y', labelcolor='#C76248')
            
            axs[_a_x][_a_y].set_title(_plm, fontsize=21)

        # 显示图例
        _a_x = 0
        _a_y = 2
        lines1, labels1 = axs[_a_x][_a_y].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        # axs[_a_x][_a_y].legend(lines1 + lines2, labels1 + labels2, bbox_to_anchor=(1.7, 0.6), fontsize=15)
    # plt.legend(lines1 + lines2, labels1 + labels2, ncol=6, bbox_to_anchor=(-1.5, -0.5), fontsize=16)
    lines1 = [lines1[0],lines1[2],lines1[1]]
    labels1 = [labels1[0],labels1[2],labels1[1]]
    lines2 = [lines2[0],lines2[2],lines2[1]]
    labels2 = [labels2[0],labels2[2],labels2[1]]
    fig.legend(lines1 + lines2, labels1 + labels2, fontsize=16, ncol=6, loc='upper center', bbox_to_anchor=(0.5, 0.1)) #(0,0) overlaps the left-down corner, (1,0) at the right-down corner outside the plot


    fig.tight_layout()
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    if not os.path.exists(f'./figure/introduction/'):
        os.makedirs(f'./figure/introduction/')
    print(f'./figure/introduction/fid_acc_incorrelate_all_{method_name}.png')
    plt.savefig(f'./figure/introduction/fid_acc_incorrelate_all_{method_name}.png',dpi=200)


def plot_fid_acc_incorrelation_compare_v2(results, plm_names, idx_list, method_name, nrow=1, show_legend=True, acc_value_only=False):
    _nrow = nrow
    _ncol = ((len(idx_list)+nrow-1)//nrow)
    _fig_size = (18 if len(idx_list)==3 else 20, 3) if nrow == 1 else (12,5)
    fig, axs = plt.subplots(nrows=_nrow, ncols=_ncol, figsize=_fig_size, sharex=False, sharey=False)
    axs = axs.flat
    # marker_list = ['o','s','^','v','x','H']
    # linestyle_list = [':','-','--',':-']
    marker_map = {'PE': 'o', 'Random': '^', 'Contrast': 's'}
    linestyle_map = {'PE': ':', 'Random': '--', 'Contrast': '-'}
    twin_axis_list = []
    values_str_list = []
    for _method_name in method_name:
        axis_idx = -1
        for _i, (_acc, _fid, _plm) in enumerate(zip(results[_method_name]['ACC'], results[_method_name]['FID'], plm_names)):
            # print(f"{_i=}, {_acc=}, ")
            if _i not in idx_list:
                continue
            axis_idx += 1
            x = np.arange(0,len(_acc),1)
            if acc_value_only == False:
                # 绘制第一条线，使用左侧 y 轴
                axs[axis_idx].plot(x, _acc, color='#4C8B8F', marker=marker_map[_method_name], linestyle=linestyle_map[_method_name], markersize=6, linewidth=2, label=f'ACC for {_method_name}')
                axs[axis_idx].set_xlabel('Iteration', fontsize=20)
                axs[axis_idx].set_xticks(x)
                axs[axis_idx].set_xticklabels(x)
                axs[axis_idx].set_ylabel('ACC', color='#4C8B8F', fontsize=20)
                axs[axis_idx].tick_params(axis='y', labelcolor='#4C8B8F')

                # 创建第二个轴对象，共享 x 轴，使用不同的 y 轴
                if len(twin_axis_list) > axis_idx:
                    ax2 = twin_axis_list[axis_idx]
                else:
                    ax2 = axs[axis_idx].twinx()
                    twin_axis_list.append(ax2)
                # 绘制第二条线，使用右侧 y 轴
                ax2.plot(x, _fid, color='#C76248', marker=marker_map[_method_name], linestyle=linestyle_map[_method_name], markersize=6, linewidth=2, label=f'FID for {_method_name}')
                ax2.set_ylabel('FID', color='#C76248', fontsize=20)
                ax2.tick_params(axis='y', labelcolor='#C76248')
                
                axs[axis_idx].set_title(_plm, fontsize=21)
            else:
                LAGENT_MAPPING = {'PE': 'Aug-PE', 'Random': 'Refine'}
                axs[axis_idx].plot(x, _fid, color='#C76248', marker=marker_map[_method_name], linestyle=linestyle_map[_method_name], markersize=6, linewidth=2, label=f'{LAGENT_MAPPING[_method_name]} ({max(_acc):.2f})')
                axs[axis_idx].set_xlabel('Iteration', fontsize=20)
                axs[axis_idx].set_xticks(x)
                axs[axis_idx].set_xticklabels(x)
                axs[axis_idx].set_ylabel('FID', fontsize=20) #, color='#C76248'
                axs[axis_idx].tick_params(axis='y') #, labelcolor='#C76248'
                
                # if len(values_str_list) > axis_idx:
                #     print(f'{values_str_list[axis_idx]=}, {_acc=}, {max(_acc):.2f}, {_plm=}, {_method_name=}')
                #     values_str_list[axis_idx] += f', {max(_acc):.2f}'
                # else:
                #     print(f'{""}, {_acc=}, {max(_acc):.2f}, {_plm=}, {_method_name=}')
                #     values_str_list.append(f'{max(_acc):.2f}')
                # axs[axis_idx].set_title(f'{_plm} ({values_str_list[axis_idx]})', fontsize=21)
                axs[axis_idx].set_title(f'{_plm}', fontsize=21)
                if axis_idx != 2:
                    axs[axis_idx].legend(loc='best', fontsize=15)
                else:
                    axs[axis_idx].legend(loc='right', fontsize=15)

        if show_legend == True:
            # 显示图例
            if acc_value_only == False:
                lines1, labels1 = axs[axis_idx].get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                axs[axis_idx].legend(lines1 + lines2, labels1 + labels2, bbox_to_anchor=(1.2, 0.95), fontsize=18)
            # else:
            #     axs[axis_idx].legend(bbox_to_anchor=(1.2, 0.95), fontsize=18)


    fig.tight_layout()
    plt.tight_layout()
    if not os.path.exists(f'./figure/introduction/'):
        os.makedirs(f'./figure/introduction/')
    if acc_value_only == False:
        print(f'./figure/introduction/fid_acc_comparison_small_{method_name}_{nrow}.png')
        plt.savefig(f'./figure/introduction/fid_acc_comparison_small_{method_name}_{nrow}.png',dpi=200)
    else:
        print(f'./figure/introduction/fid_acc_comparison_small_ACCvalue_{method_name}_{nrow}.png')
        plt.savefig(f'./figure/introduction/fid_acc_comparison_small_ACCvalue_{method_name}_{nrow}.png',dpi=200)


def plot_fid_comparison():
    FID_VALUES = [
        [22.43714657951997, 22.812377934402644, 22.853686869851185, 22.90162741977168, 23.079607297367936], 
        [19.729484527493128, 22.40769088281461, 22.540429594882017, 22.8081894907551, 23.264911913165932], 
        [21.931871576548698, 22.92326049362733, 24.067964938138253, 24.215835714518324, 24.296321390926735], 
        [22.456248204367157, 22.926147510951353, 24.023683515144604, 24.300333249626448, 24.284272329672447], 
        [25.437763551394568, 26.95029482823163, 26.493643980824466, 27.590336942203372, 27.93594779984015], 
        [16.906484109424525, 13.635706683591973, 11.627216107810888, 11.734712474606116, 11.20402353833581], 
        [17.23471867932253, 13.538982866877067, 12.447013296167533, 11.349327769732967, 11.016763186460742],
    ]
    results = {'imdb': FID_VALUES}
    task_name = 'imdb'

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 4), sharex=False, sharey=False)
    axs = [axs]
    method_list = [r'Aug-PE$_{GPT-2}$', r'Aug-PE$_{Llama-2}$', r'Aug-PE$_{Vicuna}$', r'Aug-PE$_{OPT}$', r'Aug-PE$_{ChatGLM3}$', r'Aug-PE$_{Flan-T5}$', 'WASP (Ours)']
    cmap = plt.get_cmap('GnBu',9)
    color_list = [cmap((_i+1+0.5)/6) for _i in range(len(method_list)-1)] + ['#C76248']
    marker_list = ['o','^','v','s','x','H','*']
    linestyle_list = [':','--','-',':-']
    for i_axis, _task in enumerate(list(results.keys())):
        x = np.arange(0,len(results[_task][0]),1)
        for _method_name, _acc, _color, _marker in zip(method_list, results[_task], color_list, marker_list):
            if i_axis == 0:
                axs[i_axis].plot(x, _acc, color=_color, marker=_marker, linestyle='-', markersize=7, linewidth=2, label=_method_name)
            else:
                axs[i_axis].plot(x, _acc, color=_color, marker=_marker, linestyle='-', markersize=7, linewidth=2)

        axs[i_axis].set_xlabel('Iteration', fontsize=18)
        axs[i_axis].set_xticks(x)
        axs[i_axis].set_xticklabels(x, fontsize=14)
        axs[i_axis].set_yticklabels(axs[i_axis].get_yticklabels(), fontsize=14)
        axs[i_axis].set_ylabel('FID', fontsize=18)
        # axs[i_axis].set_ylabel('ACC Difference', fontsize=20)
        # axs[i_axis].legend(loc='best', fontsize=14)
    if len(axs) == 1:
        axs[-1].legend(bbox_to_anchor=(1.05, 0.9), fontsize=14) #(0,0) overlaps the left-down corner, (1,0) at the right-down corner outside the plot
    else:
        fig.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, 0.25), fontsize=14)
    
    fig.tight_layout()
    plt.tight_layout()
    if not os.path.exists(f'./figure/fid_compare_baslines/'):
        os.makedirs(f'./figure/fid_compare_baslines/')
    print(f'./figure/fid_compare_baslines/{task_name}.png')
    plt.savefig(f'./figure/fid_compare_baslines/{task_name}.png',dpi=200)





if __name__ == "__main__":
    # selected_idx = [0,2,5]
    # acc_list = [ACC[_i] for _i in selected_idx]
    # fid_list = [FID[_i] for _i in selected_idx]
    # plm_list = [PLM_NAME[_i] for _i in selected_idx]
    # plot_fid_acc_incorrelation(acc_list, fid_list, plm_list, selected_idx, 'Random')
    
    # plot_fid_acc_incorrelation_all(VALUES['Random']['ACC'], VALUES['Random']['FID'], PLM_NAME, [0,1,2,3,4,5], 'Random')
    # plot_fid_acc_incorrelation_all(VALUES['Contrast']['ACC'], VALUES['Contrast']['FID'], PLM_NAME, [0,1,2,3,4,5], 'Contrast')
    # plot_fid_acc_incorrelation_all(VALUES['PE']['ACC'], VALUES['PE']['FID'], PLM_NAME, [0,1,2,3,4,5], 'PE')

    # plot_fid_acc_incorrelation_compare(VALUES, PLM_NAME, [0,1,2,3,4,5], ['PE','Contrast'])
    # plot_fid_acc_incorrelation_compare(VALUES, PLM_NAME, [0,1,2,3,4,5], ['PE','Random'])
    # plot_fid_acc_incorrelation_compare(VALUES, PLM_NAME, [0,1,2,3,4,5], ['Random','Contrast'])
    plot_fid_acc_incorrelation_compare(VALUES, PLM_NAME, [0,1,2,3,4,5], ['PE','Contrast','Random'])
    # plot_fid_acc_incorrelation_compare_v2(VALUES, PLM_NAME, [2,4,5], ['PE','Contrast'])
    # plot_fid_acc_incorrelation_compare_v2(VALUES, PLM_NAME, [2,4,5], ['PE','Random'])
    # plot_fid_acc_incorrelation_compare_v2(VALUES, PLM_NAME, [2,4,5], ['Random','Contrast'])
    # plot_fid_acc_incorrelation_compare_v2(VALUES, PLM_NAME, [2,4,5], ['PE','Contrast','Random'])
    # plot_fid_acc_incorrelation_compare_v2(VALUES, PLM_NAME, [2,3,4,5], ['PE','Contrast'], nrow=1, show_legend=False, acc_value_only=False)
    # plot_fid_acc_incorrelation_compare_v2(VALUES, PLM_NAME, [2,3,4,5], ['PE','Random'], nrow=1, show_legend=False, acc_value_only=False)
    # plot_fid_acc_incorrelation_compare_v2(VALUES, PLM_NAME, [2,3,4,5], ['Random','Contrast'], nrow=1, show_legend=False, acc_value_only=False)
    # plot_fid_acc_incorrelation_compare_v2(VALUES, PLM_NAME, [2,3,4,5], ['PE','Contrast','Random'], nrow=1, show_legend=False, acc_value_only=False)
    # plot_fid_acc_incorrelation_compare_v2(VALUES, PLM_NAME, [2,3,4,5], ['PE','Contrast'], nrow=1, show_legend=False, acc_value_only=True)
    plot_fid_acc_incorrelation_compare_v2(VALUES, PLM_NAME, [1,2,4,5], ['PE','Random'], nrow=1, show_legend=True, acc_value_only=True)
    plot_fid_acc_incorrelation_compare_v2(VALUES, PLM_NAME, [1,2,4,5], ['PE','Random'], nrow=2, show_legend=True, acc_value_only=True)
    # plot_fid_acc_incorrelation_compare_v2(VALUES, PLM_NAME, [1,2,4,5], ['PE','Random'], nrow=1, show_legend=False, acc_value_only=True)
    # plot_fid_acc_incorrelation_compare_v2(VALUES, PLM_NAME, [2,3,4,5], ['Random','Contrast'], nrow=1, show_legend=False, acc_value_only=True)
    # plot_fid_acc_incorrelation_compare_v2(VALUES, PLM_NAME, [2,3,4,5], ['PE','Contrast','Random'], nrow=1, show_legend=False, acc_value_only=True)

    plot_fid_comparison()