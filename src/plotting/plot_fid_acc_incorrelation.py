import sys, os
import matplotlib.pyplot as plt
import numpy as np
import torch
import re


PLM_NAME = ['GPT-2', 'Llama-2', 'Vicuna', 'OPT', 'ChatGLM3', 'Flan-T5']

ACC = [
    [78.28, 85.00, 82.25, 67.74, 73.30],
    [83.81, 69.09, 65.80, 68.89, 67.82],
    [65.81, 64.59, 69.54, 61.41, 78.82],
    [77.95, 67.43, 73.92, 74.40, 61.96],
    [86.33, 68.24, 63.34, 76.10, 69.36],
    [85.20, 86.56, 89.12, 88.83, 89.34],
]

FID = [
    [22.26, 21.95, 21.91, 20.96, 20.37],
    [19.61, 18.79, 19.12, 19.65, 19.33],
    [21.80, 18.80, 18.52, 18.34, 18.15],
    [22.22, 19.76, 19.43, 19.07, 18.32],
    [25.44, 25.68, 21.68, 21.25, 19.70],
    [16.78, 10.95, 11.62, 12.01, 10.71],
]


def plot_fid_acc_incorrelation(acc_results, fid_results, plm_names, idx_list):
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
    print(f'./figure/introduction/fid_acc_incorrelate.png')
    plt.savefig(f'./figure/introduction/fid_acc_incorrelate.png',dpi=200)


def plot_fid_acc_incorrelation_all(acc_results, fid_results, plm_names, idx_list):
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
    print(f'./figure/introduction/fid_acc_incorrelate_all.png')
    plt.savefig(f'./figure/introduction/fid_acc_incorrelate_all.png',dpi=200)


if __name__ == "__main__":
    selected_idx = [0,2,5]
    acc_list = [ACC[_i] for _i in selected_idx]
    fid_list = [FID[_i] for _i in selected_idx]
    plm_list = [PLM_NAME[_i] for _i in selected_idx]
    plot_fid_acc_incorrelation(acc_list, fid_list, plm_list, selected_idx)
    # plot_fid_acc_incorrelation_all(ACC, FID, PLM_NAME, [0,1,2,3,4,5])
