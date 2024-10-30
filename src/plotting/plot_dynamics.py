import sys, os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import json
# import jsonlines
from collections import defaultdict
from typing import List


# # file_path = './results/eval_on_real/multi_local_data_new/bert-base-uncased/0.9_errorOnlyWrongOnlySelf_new_increasedTheta_Entropy_KD0_FuseDataset0/0_1/imdb/gpt2-xl_100__llama-2-7b-chat-hf_100__vicuna-7b-1.5v_100__opt-6.7b_100__chatglm3-6b-base_100__flan-t5-xl_100/12345/logits_label_of_all_epochs.pth' # for debug, 3 epochs trained
# # file_path = './results/eval_on_real/multi_local_data_new/bert-base-uncased/0.9_errorOnlyWrongOnlySelf_new_increasedTheta_Entropy_KD0_FuseDataset0/0_1/imdb/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__chatglm3-6b-base_1000__flan-t5-xl_1000/12345/logits_label_of_all_epochs_6epochs.pth'
# # file_path = './results/eval_on_real/multi_local_data_new/bert-base-uncased/0.9_errorOnlyWrongOnlySelf_new_increasedTheta_Entropy_KD0_FuseDataset0/0_1/imdb/gpt2-xl_10000__llama-2-7b-chat-hf_10000__vicuna-7b-1.5v_10000__opt-6.7b_10000__chatglm3-6b-base_10000__flan-t5-xl_10000/12345/logits_label_of_all_epochs_6epochs.pth'
# # file_path = './results/eval_on_real/multi_local_data_new/bert-base-uncased/0.9_errorOnlyWrongOnlySelf_new_increasedTheta_Entropy_KD0_FuseDataset0/0_1/yelp/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__chatglm3-6b-base_1000__flan-t5-xl_1000/12345/logits_label_of_all_epochs_6epochs.pth'
# # file_path = './results/eval_on_real/multi_local_data_new/bert-base-uncased/0.9_errorOnlyWrongOnlySelf_new_increasedTheta_Entropy_KD0_FuseDataset0/0_1/yelp/gpt2-xl_10000__llama-2-7b-chat-hf_10000__vicuna-7b-1.5v_10000__opt-6.7b_10000__chatglm3-6b-base_10000__flan-t5-xl_10000/12345/logits_label_of_all_epochs_6epochs.pth'
# # file_path = './results/eval_on_real/multi_local_data_flip/bert-base-uncased/0.9_errorOnlyWrongOnlySelf_new_increasedTheta_Entropy_KD0_FuseDataset0/0_1/imdb/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__chatglm3-6b-base_1000__flan-t5-xl_1000/12345/logits_label_of_all_epochs_6epochs.pth'
# file_path = './results/eval_on_real/multi_local_data_flip/bert-base-uncased/0.9_errorOnlyWrongOnlySelf_new_increasedTheta_Entropy_KD0_FuseDataset0/0_1/imdb/gpt2-xl_10000__llama-2-7b-chat-hf_10000__vicuna-7b-1.5v_10000__opt-6.7b_10000__chatglm3-6b-base_10000__flan-t5-xl_10000/12345/logits_label_of_all_epochs_6epochs.pth'


def plot_data_map(dataframe: pd.DataFrame,
                  plot_dir: os.path,
                  hue_metric: str = 'correct.',
                  title: str = '',
                  model: str = 'RoBERTa',
                  show_hist: bool = False,
                  max_instances_to_plot = 55000):
    # Set style.
    sns.set(style='whitegrid', font_scale=1.6, font='Georgia', context='paper')

    # Subsample data to plot, so the plot is not too busy.
    dataframe = dataframe.sample(n=max_instances_to_plot if dataframe.shape[0] > max_instances_to_plot else len(dataframe))

    # Normalize correctness to a value between 0 and 1.
    dataframe = dataframe.assign(corr_frac = lambda d: d.correctness / d.correctness.max())
    dataframe['correct.'] = [float(f"{x:.1f}") for x in dataframe['corr_frac']]

    main_metric = 'variability'
    other_metric = 'confidence'

    hue = hue_metric
    num_hues = len(dataframe[hue].unique().tolist())
    style = hue_metric if num_hues < 8 else None

    if not show_hist:
        fig, ax0 = plt.subplots(1, 1, figsize=(14, 9))
    else:
        fig = plt.figure(figsize=(14, 10), )
        gs = fig.add_gridspec(3, 2, width_ratios=[4.9, 1.1])
        ax0 = fig.add_subplot(gs[:, 0])

    # Make the scatterplot.
    # Choose a palette.
    pal = sns.diverging_palette(260, 15, n=num_hues, sep=10, center="dark")

    plot = sns.scatterplot(x=main_metric,
                           y=other_metric,
                           ax=ax0,
                           data=dataframe,
                           hue=hue,
                           palette=pal,
                           style=style,
                        #    s=30)
                           s=(240 if show_hist else 360))

    # Annotate Regions.
    bb = lambda c: dict(boxstyle="round,pad=0.3", ec=c, lw=2, fc="white")
    func_annotate = lambda  text, xyc, bbc : ax0.annotate(text,
                                                          xy=xyc,
                                                          xycoords="axes fraction",
                                                        #   fontsize=15,
                                                          fontsize=(24 if show_hist else 42),
                                                          color='black',
                                                          va="center",
                                                          ha="center",
                                                          rotation=350,
                                                          alpha=0.9,
                                                           bbox=bb(bbc))
    an1 = func_annotate("ambiguous", xyc=(0.8, 0.5), bbc='black')
    an2 = func_annotate("easy-to-learn", xyc=(0.27, 0.75), bbc='r')
    an3 = func_annotate("hard-to-learn", xyc=(0.35, 0.25), bbc='b')


    if not show_hist:
        lgnd = plot.legend(ncol=1, bbox_to_anchor=[0.175, 0.5], loc='right', markerscale=3, fontsize=28)
    else:
        lgnd = plot.legend(fancybox=True, shadow=True, ncol=1, fontsize=28)
        # for handle in lgnd.legend_handles:
        #     handle.set_sizes([100.0])
    plot.set_xlabel('variability',fontsize=(24 if show_hist else 48))
    plot.set_ylabel('confidence',fontsize=(24 if show_hist else 48))

    if show_hist:
        # plot.set_title(f"{title}-{model} Data Map", fontsize=17)
        # plot.set_title(f"{title}", fontsize=19)

        # Make the histograms.
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[1, 1])
        ax3 = fig.add_subplot(gs[2, 1])

        plott0 = dataframe.hist(column=['confidence'], ax=ax1, color='#622a87')
        plott0[0].set_title('')
        plott0[0].set_xlabel('confidence',fontsize=24)
        plott0[0].set_ylabel('density',fontsize=24)
        # plott0[0].xticks(fontsize=21) 
        # plott0[0].yticks(fontsize=21) 
        # ax1.set_yticklabels(ax1.get_xticks(), fontsize=21)
        # ax1.set_yticklabels(ax1.get_yticks(), fontsize=21)

        plott1 = dataframe.hist(column=['variability'], ax=ax2, color='teal')
        plott1[0].set_title('')
        plott1[0].set_xlim(0.0,0.23)
        plott1[0].set_xlabel('variability',fontsize=24)
        plott1[0].set_ylabel('density',fontsize=24)
        # plott1[0].xticks(fontsize=21) 
        # plott1[0].yticks(fontsize=21) 
        # ax2.set_xticklabels(ax2.get_xticks(), fontsize=21, rotation=50)
        # ax2.set_yticklabels(ax2.get_yticks(), fontsize=21)

        plott2 = sns.countplot(x="correct.", data=dataframe, order=[0.0,0.2,0.3,0.5,0.7,0.8,1.0], ax=ax3, color='#86bf91')
        ax3.xaxis.grid(True) # Show the vertical gridlines
        # ax3.set_xticklabels(ax3.get_xticks(), size=15)
        plott2.set_title('')
        plott2.set_xlabel('correctness',fontsize=24)
        plott2.set_ylabel('density',fontsize=24)
        # plott2[0].xticks(fontsize=21, rotation=50) 
        # plott2[0].yticks(fontsize=21) 
        # ax3.set_xticklabels(ax3.get_xticks(), fontsize=21, rotation=50)
        # ax3.set_yticklabels(ax3.get_yticks(), fontsize=21)


    fig.tight_layout()
    # filename = f'{plot_dir}/{title}_{model}.pdf' if show_hist else f'figures/compact_{title}_{model}.pdf'
    filename = plot_dir
    fig.savefig(filename, dpi=300)


def plot_data_map_2file(total_dataframe: pd.DataFrame,
                        plot_dir: os.path,
                        hue_metric: str = 'correct.',
                        title: str = '',
                        model: str = 'RoBERTa',
                        show_hist: bool = False,
                        max_instances_to_plot = 55000):
    # Set style.
    sns.set(style='whitegrid', font_scale=1.6, font='Georgia', context='paper')

    # Subsample data to plot, so the plot is not too busy.
    dataframe = total_dataframe["original_metric"]
    dataframe = dataframe.sample(n=max_instances_to_plot if dataframe.shape[0] > max_instances_to_plot else len(dataframe))

    # Normalize correctness to a value between 0 and 1.
    epoch_num = max(list(dataframe["correctness"]))
    print(f"{epoch_num=}")
    dataframe = dataframe.assign(corr_frac = lambda d: d.correctness / d.correctness.max())
    dataframe['correct.'] = [float(f"{x:.1f}") for x in dataframe['corr_frac']]

    main_metric = 'variability'
    other_metric = 'confidence'

    hue = hue_metric
    num_hues = len(dataframe[hue].unique().tolist())
    style = hue_metric if num_hues < 8 else None

    if not show_hist:
        fig, ax0 = plt.subplots(1, 1, figsize=(8, 6))
    else:
        # fig = plt.figure(figsize=(17.5, 10), )
        fig = plt.figure(figsize=(25, 10), )
        # gs = fig.add_gridspec(3, 3, width_ratios=[5, 1.5, 1.5]) # 3 columns, 3 rows
        gs = fig.add_gridspec(3, 3, width_ratios=[5, 1.2, 5]) # 3 columns, 3 rows
        ax0 = fig.add_subplot(gs[:, 0])
        ax00 = fig.add_subplot(gs[:, 2])

    # Make the scatterplot.
    # Choose a palette.
    pal = sns.diverging_palette(260, 15, n=num_hues, sep=10, center="dark")

    plot = sns.scatterplot(x=main_metric,
                           y=other_metric,
                           ax=ax0,
                           data=dataframe,
                           hue=hue,
                           palette=pal,
                           style=style,
                           s=30)

    # Annotate Regions.
    bb = lambda c: dict(boxstyle="round,pad=0.3", ec=c, lw=2, fc="white")
    func_annotate = lambda  text, xyc, bbc : ax0.annotate(text,
                                                          xy=xyc,
                                                          xycoords="axes fraction",
                                                          fontsize=15,
                                                          color='black',
                                                          va="center",
                                                          ha="center",
                                                          rotation=350,
                                                           bbox=bb(bbc))
    an1 = func_annotate("ambiguous", xyc=(0.9, 0.5), bbc='black')
    an2 = func_annotate("easy-to-learn", xyc=(0.27, 0.85), bbc='r')
    an3 = func_annotate("hard-to-learn", xyc=(0.35, 0.25), bbc='b')


    if not show_hist:
        plot.legend(ncol=1, bbox_to_anchor=[0.175, 0.5], loc='right')
    else:
        plot.legend(fancybox=True, shadow=True,  ncol=1)
    plot.set_xlabel('variability')
    plot.set_ylabel('confidence')

    if show_hist:
        plot.set_title(f"{title}-{model} Data Map", fontsize=17)

        # Make the histograms.
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[1, 1])
        ax3 = fig.add_subplot(gs[2, 1])

        plott0 = dataframe.hist(column=['confidence'], ax=ax1, color='#622a87')
        ax1.set_yscale("log")
        plott0[0].set_title('')
        plott0[0].set_xlabel('confidence')
        plott0[0].set_ylabel('density')

        plott1 = dataframe.hist(column=['variability'], ax=ax2, color='teal')
        ax2.set_yscale("log")
        plott1[0].set_title('')
        plott1[0].set_xlabel('variability')
        plott1[0].set_ylabel('density')

        plot2 = sns.countplot(x="correct.", data=dataframe, ax=ax3, color='#86bf91')
        ax3.set_yscale("log")
        ax3.xaxis.grid(True) # Show the vertical gridlines

        plot2.set_title('')
        plot2.set_xlabel('correctness')
        plot2.set_ylabel('density')

        # # Make the histograms.
        # # Subsample data to plot, so the plot is not too busy.
        # dataframe = total_dataframe["diff_metric"]
        # dataframe = dataframe.sample(n=max_instances_to_plot if dataframe.shape[0] > max_instances_to_plot else len(dataframe))
        # # Normalize correctness to a value between 0 and 1.
        # dataframe = dataframe.assign(corr_frac = lambda d: d.correctness / epoch_num)
        # dataframe['correct.'] = [float(f"{x:.1f}") for x in dataframe['corr_frac']]
        # print(dataframe['correct.'])
        # ax1 = fig.add_subplot(gs[0, 2])
        # ax2 = fig.add_subplot(gs[1, 2])
        # ax3 = fig.add_subplot(gs[2, 2])

        # plott0 = dataframe.hist(column=['confidence'], ax=ax1, color='#622a87')
        # ax1.set_yscale("log")
        # plott0[0].set_title('')
        # plott0[0].set_xlabel('confidence')
        # plott0[0].set_ylabel('density')

        # plott1 = dataframe.hist(column=['variability'], ax=ax2, color='teal')
        # ax2.set_yscale("log")
        # plott1[0].set_title('')
        # plott1[0].set_xlabel('variability')
        # plott1[0].set_ylabel('density')

        # plot2 = sns.countplot(x="correct.", data=dataframe, ax=ax3, color='#86bf91')
        # plot2.set_xticklabels(plot2.get_xticklabels(), rotation=60, ha="right")
        # ax3.set_yscale("log")
        # ax3.xaxis.grid(True) # Show the vertical gridlines

        # plot2.set_title('')
        # plot2.set_xlabel('correctness')
        # plot2.set_ylabel('density')

    # Subsample data to plot, so the plot is not too busy.
    dataframe = total_dataframe["diff_metric"]
    dataframe = dataframe.sample(n=max_instances_to_plot if dataframe.shape[0] > max_instances_to_plot else len(dataframe))
    # Normalize correctness to a value between 0 and 1.
    dataframe = dataframe.assign(corr_frac = lambda d: d.correctness / epoch_num)
    dataframe['correct.'] = [float(f"{x:.1f}") for x in dataframe['corr_frac']]
    
    hue = hue_metric
    num_hues = len(dataframe[hue].unique().tolist())
    style = hue_metric if num_hues < 8 else None
    pal = sns.diverging_palette(260, 15, n=num_hues, sep=10, center="dark")
    plot = sns.scatterplot(x=main_metric,
                           y=other_metric,
                           ax=ax00,
                           data=dataframe,
                           hue=hue,
                           palette=pal,
                           style=style,
                           s=30)
    # Annotate Regions.
    bb = lambda c: dict(boxstyle="round,pad=0.3", ec=c, lw=2, fc="white")
    func_annotate = lambda  text, xyc, bbc : ax0.annotate(text,
                                                          xy=xyc,
                                                          xycoords="axes fraction",
                                                          fontsize=15,
                                                          color='black',
                                                          va="center",
                                                          ha="center",
                                                          rotation=350,
                                                           bbox=bb(bbc))
    an1 = func_annotate("ambiguous", xyc=(0.9, 0.5), bbc='black')
    an2 = func_annotate("easy-to-learn", xyc=(0.27, 0.85), bbc='r')
    an3 = func_annotate("hard-to-learn", xyc=(0.35, 0.25), bbc='b')

    # if not show_hist:
    #     plot.legend(ncol=1, bbox_to_anchor=[0.175, 0.5], loc='right')
    # else:
    #     plot.legend(fancybox=True, shadow=True,  ncol=1)
    plot.set_xlabel('variability change')
    plot.set_ylabel('confidence change')


    fig.tight_layout()
    # filename = f'{plot_dir}/{title}_{model}.pdf' if show_hist else f'figures/compact_{title}_{model}.pdf'
    filename = plot_dir
    fig.savefig(filename, dpi=300)


def read_training_dynamics(metric_file):
                        #     , 
                        #    model_dir: os.path,
                        #    strip_last: bool = False,
                        #    id_field: str = "guid",
                        #    burn_out: int = None):
    """
    Given path to logged training dynamics, merge stats across epochs.
    Returns:
    - Dict between ID of a train instances and its gold label, and the list of logits across epochs.
    """
    train_dynamics = {}

    task_name = metric_file.split('/')[-4]
    models_samples = metric_file.split('/')[-3]
    assert '__' in models_samples, f"{models_samples=}"
    models_samples = models_samples.split('__')
    models_samples = [item.split('_') for item in models_samples]
    models = [item[0] for item in models_samples]
    samples = [int(item[1]) for item in models_samples]
    accumulated_sample_count = [0]
    for sample_num in samples:
        accumulated_sample_count.append(accumulated_sample_count[-1]+sample_num)
    loggings = list(torch.load(metric_file, map_location=torch.device('cpu')))
    NUM_MODELS = len(models)

    for model in models:
        train_dynamics[model] = {}

    assert len(models)+1 == len(loggings), f"{len(models)=}, {len(loggings)=}"
    logits = loggings[0]
    labels = loggings[1:]
    assert len(logits) == len(labels) == len(models), f"{len(logits)=}, {len(labels)=}, {len(models)=}"

    for im in range(NUM_MODELS):
        _plm = models[im]
        for _i, (_logits, _label) in enumerate(zip(logits[im].detach(), labels[im])):
            train_dynamics[_plm][_i] = {"gold": _label.item(), "logits": _logits.numpy()}

    # td_dir = os.path.join(model_dir, "training_dynamics")
    # num_epochs = len([f for f in os.listdir(td_dir) if os.path.isfile(os.path.join(td_dir, f))])
    # if burn_out:
    # num_epochs = burn_out

    # logger.info(f"Reading {num_epochs} files from {td_dir} ...")
    # for epoch_num in tqdm.tqdm(range(num_epochs)):
    # epoch_file = os.path.join(td_dir, f"dynamics_epoch_{epoch_num}.jsonl")
    # assert os.path.exists(epoch_file)

    # with open(epoch_file, "r") as infile:
    #     for line in infile:
    #     record = json.loads(line.strip())
    #     guid = record[id_field] if not strip_last else record[id_field][:-1]
    #     if guid not in train_dynamics:
    #         assert epoch_num == 0
    #         train_dynamics[guid] = {"gold": record["gold"], "logits": []}
    #     train_dynamics[guid]["logits"].append(record[f"logits_epoch_{epoch_num}"])

    # logger.info(f"Read training dynamics for {len(train_dynamics)} train instances.")
    return train_dynamics, models, samples, accumulated_sample_count, NUM_MODELS, task_name


def read_training_dynamics_across_model(metric_file, logits_file, label_file):

    train_dynamics = {}

    task_name = file_path.split('/')[-4]
    models_samples = file_path.split('/')[-3]
    assert '__' in models_samples, f"{models_samples=}"
    models_samples = models_samples.split('__')
    models_samples = [item.split('_') for item in models_samples]
    models = [item[0] for item in models_samples]
    samples = [int(item[1]) for item in models_samples]
    accumulated_sample_count = [0]
    for sample_num in samples:
        accumulated_sample_count.append(accumulated_sample_count[-1]+sample_num)

    NUM_MODELS = len(models)

    for model in models:
        train_dynamics[model] = {}

    label_loggings = list(torch.load(metric_file+label_file, map_location=torch.device('cpu')))
    assert len(models)+1 == len(label_loggings), f"{len(models)=}, {len(label_loggings)=}"
    labels = label_loggings[1:]

    logits_loggings = torch.load(metric_file+logits_file, map_location=torch.device('cpu'))
    print(logits_loggings)
    logits = [torch.transpose(logits_loggings[_i,:,:,:],0,1) for _i in range(NUM_MODELS)]
    print(logits[0][0].shape, logits[0][0][:5])
    assert len(logits) == len(labels) == len(models), f"{len(logits)=}, {len(labels)=}, {len(models)=}"

    for im in range(NUM_MODELS):
        _plm = models[im]
        for _i, (_logits, _label) in enumerate(zip(logits[im].detach(), labels[im])):
            train_dynamics[_plm][_i] = {"gold": _label.item(), "logits": _logits.numpy()}

    return train_dynamics, models, samples, accumulated_sample_count, NUM_MODELS, task_name


def read_training_dynamics_2file(original_metric_file, flipped_metric_file):
    
    train_dynamics = {"original": {}, "flipped": {}}
    
    task_name = original_metric_file.split('/')[-4]
    models_samples = original_metric_file.split('/')[-3]
    task_name_flipped = flipped_metric_file.split('/')[-4]
    models_samples_flipped = flipped_metric_file.split('/')[-3]
    assert task_name == task_name_flipped and models_samples == models_samples_flipped, f"unpaired files, {task_name=}, {task_name_flipped=} and {models_samples=}, {models_samples_flipped=}"
    assert '__' in models_samples, f"{models_samples=}"
    models_samples = models_samples.split('__')
    models_samples = [item.split('_') for item in models_samples]
    models = [item[0] for item in models_samples]
    samples = [int(item[1]) for item in models_samples]
    accumulated_sample_count = [0]
    for sample_num in samples:
        accumulated_sample_count.append(accumulated_sample_count[-1]+sample_num)
    loggings = [
        list(torch.load(original_metric_file, map_location=torch.device('cpu'))), 
        list(torch.load(flipped_metric_file, map_location=torch.device('cpu')))
    ]
    NUM_MODELS = len(models)

    for model in models:
        train_dynamics["original"][model] = {}
        train_dynamics["flipped"][model] = {}

    assert len(models)+1 == len(loggings[0]) == len(loggings[1]), f"{len(models)=}, {len(loggings[0])=}, {len(loggings[1])=}"
    logits = [loggings[0][0], loggings[1][0]]
    labels = [loggings[0][1:], loggings[1][1:]]
    assert len(logits[0]) == len(labels[0]) == len(models), f"{len(logits[0])=}, {len(labels[0])=}, {len(models)=}"
    assert len(logits[1]) == len(labels[1]) == len(models), f"{len(logits[1])=}, {len(labels[1])=}, {len(models)=}"

    for key, logits_dataset, labels_dataset in zip(["original", "flipped"], logits, labels):
        for im in range(NUM_MODELS):
            _plm = models[im]
            for _i, (_logits, _label) in enumerate(zip(logits_dataset[im].detach(), labels_dataset[im])):
                train_dynamics[key][_plm][_i] = {"gold": _label.item(), "logits": _logits.numpy()}

    print(f"{train_dynamics.keys()=}")
    for key in train_dynamics.keys():
        print(f"{key=}, {train_dynamics[key].keys()}")
    return train_dynamics, models, samples, accumulated_sample_count, NUM_MODELS, task_name


def read_training_dynamics_2file_across_model(original_metric_file, flipped_metric_file, logits_file, label_file):

    train_dynamics = {"original": {}, "flipped": {}}
    
    task_name = original_metric_file.split('/')[-4]
    models_samples = original_metric_file.split('/')[-3]
    task_name_flipped = flipped_metric_file.split('/')[-4]
    models_samples_flipped = flipped_metric_file.split('/')[-3]
    assert task_name == task_name_flipped and models_samples == models_samples_flipped, f"unpaired files, {task_name=}, {task_name_flipped=} and {models_samples=}, {models_samples_flipped=}"
    assert '__' in models_samples, f"{models_samples=}"
    models_samples = models_samples.split('__')
    models_samples = [item.split('_') for item in models_samples]
    models = [item[0] for item in models_samples]
    samples = [int(item[1]) for item in models_samples]
    accumulated_sample_count = [0]
    for sample_num in samples:
        accumulated_sample_count.append(accumulated_sample_count[-1]+sample_num)
    
    NUM_MODELS = len(models)

    for model in models:
        train_dynamics["original"][model] = {}
        train_dynamics["flipped"][model] = {}

    label_loggings = [
        list(torch.load(original_metric_file+label_file, map_location=torch.device('cpu'))), 
        list(torch.load(flipped_metric_file+label_file, map_location=torch.device('cpu')))
    ]
    assert len(models)+1 == len(label_loggings[0]) == len(label_loggings[1]), f"{len(models)=}, {len(label_loggings[0])=}, {len(label_loggings[1])=}"
    # logits = [loggings[0][0], loggings[1][0]]
    labels = [label_loggings[0][1:], label_loggings[1][1:]]

    logits_loggings = [
        torch.load(original_metric_file+logits_file, map_location=torch.device('cpu')), 
        torch.load(flipped_metric_file+logits_file, map_location=torch.device('cpu'))
    ]
    print(logits_loggings[0], logits_loggings[1])
    logits = [[torch.transpose(logits_loggings[0][_i,:,:,:],0,1) for _i in range(NUM_MODELS)],
              [torch.transpose(logits_loggings[1][_i,:,:,:],0,1) for _i in range(NUM_MODELS)]]
    print(logits[0][0].shape, logits[0][0][:5])
    # assert 1 == 0
    assert len(logits[0]) == len(labels[0]) == len(models), f"{len(logits[0])=}, {len(labels[0])=}, {len(models)=}"
    assert len(logits[1]) == len(labels[1]) == len(models), f"{len(logits[1])=}, {len(labels[1])=}, {len(models)=}"

    for key, logits_dataset, labels_dataset in zip(["original", "flipped"], logits, labels):
        for im in range(NUM_MODELS):
            _plm = models[im]
            for _i, (_logits, _label) in enumerate(zip(logits_dataset[im].detach(), labels_dataset[im])):
                train_dynamics[key][_plm][_i] = {"gold": _label.item(), "logits": _logits.numpy()}

    print(f"{train_dynamics.keys()=}")
    for key in train_dynamics.keys():
        print(f"{key=}, {train_dynamics[key].keys()}")
    return train_dynamics, models, samples, accumulated_sample_count, NUM_MODELS, task_name



def compute_forgetfulness(correctness_trend: List[float]) -> int:
  """
  Given a epoch-wise trend of train predictions, compute frequency with which
  an example is forgotten, i.e. predicted incorrectly _after_ being predicted correctly.
  Based on: https://arxiv.org/abs/1812.05159
  """
  if not any(correctness_trend):  # Example is never predicted correctly, or learnt!
      return 1000
  learnt = False  # Predicted correctly in the current epoch.
  times_forgotten = 0
  for is_correct in correctness_trend:
    if (not learnt and not is_correct) or (learnt and is_correct):
      # nothing changed.
      continue
    elif learnt and not is_correct:
      # Forgot after learning at some point!
      learnt = False
      times_forgotten += 1
    elif not learnt and is_correct:
      # Learnt!
      learnt = True
  return times_forgotten


def compute_correctness(trend: List[float]) -> float:
  """
  Aggregate #times an example is predicted correctly during all training epochs.
  """
  return sum(trend)


def compute_train_dy_metrics(training_dynamics, include_ci=False):
    """
    Given the training dynamics (logits for each training instance across epochs), compute metrics
    based on it, for data map coorodinates.
    Computed metrics are: confidence, variability, correctness, forgetfulness, threshold_closeness---
    the last two being baselines from prior work
    (Example Forgetting: https://arxiv.org/abs/1812.05159 and
    Active Bias: https://arxiv.org/abs/1704.07433 respectively).
    Returns:
    - DataFrame with these metrics.
    - DataFrame with more typical training evaluation metrics, such as accuracy / loss.
    """
    confidence_ = {}
    variability_ = {}
    threshold_closeness_ = {}
    correctness_ = {}
    forgetfulness_ = {}

    # Functions to be applied to the data.
    variability_func = lambda conf: np.std(conf)
    if include_ci:  # Based on prior work on active bias (https://arxiv.org/abs/1704.07433)
        variability_func = lambda conf: np.sqrt(np.var(conf) + np.var(conf) * np.var(conf) / (len(conf)-1))
    threshold_closeness_func = lambda conf: conf * (1 - conf)

    loss = torch.nn.CrossEntropyLoss()

    num_tot_epochs = len(list(training_dynamics.values())[0]["logits"])

    logits = {i: [] for i in range(num_tot_epochs)}
    targets = {i: [] for i in range(num_tot_epochs)}
    training_accuracy = defaultdict(float)

    for guid in training_dynamics:
        correctness_trend = []
        true_probs_trend = []

        record = training_dynamics[guid]
        for i, epoch_logits in enumerate(record["logits"]):
            probs = torch.nn.functional.softmax(torch.Tensor(epoch_logits), dim=-1)
            true_class_prob = float(probs[record["gold"]])
            true_probs_trend.append(true_class_prob)

            prediction = np.argmax(epoch_logits)
            is_correct = (prediction == record["gold"]).item()
            correctness_trend.append(is_correct)

            training_accuracy[i] += is_correct
            logits[i].append(epoch_logits)
            targets[i].append(record["gold"])

        correctness_[guid] = compute_correctness(correctness_trend)
        confidence_[guid] = np.mean(true_probs_trend)
        variability_[guid] = variability_func(true_probs_trend)

        forgetfulness_[guid] = compute_forgetfulness(correctness_trend)
        threshold_closeness_[guid] = threshold_closeness_func(confidence_[guid])

    # # Should not affect ranking, so ignoring.
    # epsilon_var = np.mean(list(variability_.values()))

    column_names = ['guid',
                    'index',
                    'threshold_closeness',
                    'confidence',
                    'variability',
                    'correctness',
                    'forgetfulness',]
    df = pd.DataFrame([[guid,
                        i,
                        threshold_closeness_[guid],
                        confidence_[guid],
                        variability_[guid],
                        correctness_[guid],
                        forgetfulness_[guid],
                        ] for i, guid in enumerate(correctness_)], columns=column_names)

    df_train = pd.DataFrame([[i,
                            loss(torch.Tensor(logits[i]), torch.LongTensor(targets[i])).item() / len(training_dynamics),
                            training_accuracy[i] / len(training_dynamics)
                            ] for i in range(num_tot_epochs)],
                            columns=['epoch', 'loss', 'train_acc'])
    return df, df_train


def compute_train_dy_metrics_2file(training_dynamics, include_ci=False):
    """
    Given the training dynamics (logits for each training instance across epochs), compute metrics
    based on it, for data map coorodinates.
    Computed metrics are: confidence, variability, correctness, forgetfulness, threshold_closeness---
    the last two being baselines from prior work
    (Example Forgetting: https://arxiv.org/abs/1812.05159 and
    Active Bias: https://arxiv.org/abs/1704.07433 respectively).
    Returns:
    - DataFrame with these metrics.
    - DataFrame with more typical training evaluation metrics, such as accuracy / loss.
    """
    confidence_ = {}
    variability_ = {}
    threshold_closeness_ = {}
    correctness_ = {}
    forgetfulness_ = {}
    flipped_confidence_ = {}
    flipped_variability_ = {}
    flipped_threshold_closeness_ = {}
    flipped_correctness_ = {}
    flipped_forgetfulness_ = {}

    # Functions to be applied to the data.
    variability_func = lambda conf: np.std(conf)
    if include_ci:  # Based on prior work on active bias (https://arxiv.org/abs/1704.07433)
        variability_func = lambda conf: np.sqrt(np.var(conf) + np.var(conf) * np.var(conf) / (len(conf)-1))
    threshold_closeness_func = lambda conf: conf * (1 - conf)

    loss = torch.nn.CrossEntropyLoss()


    # for "original" part, large scatter file
    num_tot_epochs = len(list(training_dynamics["original"].values())[0]["logits"])

    logits = {i: [] for i in range(num_tot_epochs)}
    targets = {i: [] for i in range(num_tot_epochs)}
    flipped_logits = {i: [] for i in range(num_tot_epochs)}
    flipped_targets = {i: [] for i in range(num_tot_epochs)}
    training_accuracy = defaultdict(float)

    for guid in training_dynamics["original"]:
        correctness_trend = []
        true_probs_trend = []

        record = training_dynamics["original"][guid]
        for i, epoch_logits in enumerate(record["logits"]):
            probs = torch.nn.functional.softmax(torch.Tensor(epoch_logits), dim=-1)
            true_class_prob = float(probs[record["gold"]])
            true_probs_trend.append(true_class_prob)

            prediction = np.argmax(epoch_logits)
            is_correct = (prediction == record["gold"]).item()
            correctness_trend.append(is_correct)

            training_accuracy[i] += is_correct
            logits[i].append(epoch_logits)
            targets[i].append(record["gold"])

        correctness_[guid] = compute_correctness(correctness_trend)
        confidence_[guid] = np.mean(true_probs_trend)
        variability_[guid] = variability_func(true_probs_trend)

        forgetfulness_[guid] = compute_forgetfulness(correctness_trend)
        threshold_closeness_[guid] = threshold_closeness_func(confidence_[guid])


    for guid in training_dynamics["flipped"]:
        flipped_correctness_trend = []
        flipped_true_probs_trend = []

        record = training_dynamics["flipped"][guid]
        for i, epoch_logits in enumerate(record["logits"]):
            probs = torch.nn.functional.softmax(torch.Tensor(epoch_logits), dim=-1)
            true_class_prob = float(probs[record["gold"]])
            flipped_true_probs_trend.append(true_class_prob)

            prediction = np.argmax(epoch_logits)
            is_correct = (prediction == record["gold"]).item()
            flipped_correctness_trend.append(is_correct)

            training_accuracy[i] += is_correct
            flipped_logits[i].append(epoch_logits)
            flipped_targets[i].append(record["gold"])

        flipped_correctness_[guid] = compute_correctness(flipped_correctness_trend)
        flipped_confidence_[guid] = np.mean(flipped_true_probs_trend)
        flipped_variability_[guid] = variability_func(flipped_true_probs_trend)

        flipped_forgetfulness_[guid] = compute_forgetfulness(flipped_correctness_trend)
        flipped_threshold_closeness_[guid] = threshold_closeness_func(flipped_confidence_[guid])

        # flipped_correctness_[guid] -= correctness_[guid]
        flipped_correctness_[guid] = correctness_[guid]
        flipped_confidence_[guid] -= confidence_[guid]
        flipped_variability_[guid] -= variability_[guid]
        flipped_forgetfulness_[guid] -= forgetfulness_[guid]
        flipped_threshold_closeness_[guid] -= threshold_closeness_[guid]


    # # Should not affect ranking, so ignoring.
    # epsilon_var = np.mean(list(variability_.values()))

    column_names = ['guid',
                    'index',
                    'threshold_closeness',
                    'confidence',
                    'variability',
                    'correctness',
                    'forgetfulness',]
    df = pd.DataFrame([[guid,
                        i,
                        threshold_closeness_[guid],
                        confidence_[guid],
                        variability_[guid],
                        correctness_[guid],
                        forgetfulness_[guid],
                        ] for i, guid in enumerate(correctness_)], columns=column_names)
    df_diff = pd.DataFrame([[guid,
                            i,
                            flipped_threshold_closeness_[guid],
                            flipped_confidence_[guid],
                            flipped_variability_[guid],
                            flipped_correctness_[guid],
                            flipped_forgetfulness_[guid],
                            ] for i, guid in enumerate(flipped_correctness_)], columns=column_names)

    df_train = pd.DataFrame([[i,
                            loss(torch.Tensor(logits[i]), torch.LongTensor(targets[i])).item() / len(training_dynamics["original"]),
                            training_accuracy[i] / len(training_dynamics["original"])
                            ] for i in range(num_tot_epochs)],
                            columns=['epoch', 'loss', 'train_acc'])
    
    return df, df_train, df_diff


if __name__ == '__main__':
    # ################ single plot starts ################
    ###### single model, several epochs ######
    # file_path = {
    #     "imdb": [
    #         # './results/eval_on_real/dynamic_multi_local_data_new/bert-base-uncased/0.9_errorOnlyWrongOnlySelfTestAll_Adjust_all_Entropy_KD0_FuseDataset0/0_1/imdb/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__chatglm3-6b-base_1000__flan-t5-xl_1000/12345/logits_label_of_all_epochs_3epochs.pth',
    #         './results/eval_on_real/dynamic_multi_local_data_new/bert-base-uncased/0.9_errorOnlyWrongOnlySelfTestAll_Adjust_all_Entropy_KD0_FuseDataset0/0_1/imdb/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__chatglm3-6b-base_1000__flan-t5-xl_1000/12345/logits_label_of_all_epochs_6epochs.pth',
    #         # './results/eval_on_real/dynamic_multi_local_data_accumulate/imdb/single_copied/bert-base-uncased/increasedTheta_KD1_FuseDataset1/fewshotK8_5_0.5/1/bert-base-uncased/0.9_errorOnlyWrongOnlySelfTestAll_Adjust_all_Entropy_KD0_FuseDataset0/0_1/imdb/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__chatglm3-6b-base_1000__flan-t5-xl_1000/12345/logits_label_of_all_epochs_3epochs.pth',
    #         './results/eval_on_real/dynamic_multi_local_data_accumulate/imdb/single_copied/bert-base-uncased/increasedTheta_KD1_FuseDataset1/fewshotK8_5_0.5/1/bert-base-uncased/0.9_errorOnlyWrongOnlySelfTestAll_Adjust_all_Entropy_KD0_FuseDataset0/0_1/imdb/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__chatglm3-6b-base_1000__flan-t5-xl_1000/12345/logits_label_of_all_epochs_6epochs.pth',
    #         # './results/eval_on_real/dynamic_multi_local_data_accumulate/imdb/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__flan-t5-xl_1000__chatglm3-6b-base_1000/bert-base-uncased/increasedTheta_KD1_FuseDataset1/fewshotK8_5_0.5/12345/bert-base-uncased/0.9_errorOnlyWrongOnlySelfTestAll_Adjust_all_Entropy_KD0_FuseDataset0/0_1/imdb/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__chatglm3-6b-base_1000__flan-t5-xl_1000/12345/logits_label_of_all_epochs_3epochs.pth',
    #         './results/eval_on_real/dynamic_multi_local_data_accumulate/imdb/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__flan-t5-xl_1000__chatglm3-6b-base_1000/bert-base-uncased/increasedTheta_KD1_FuseDataset1/fewshotK8_5_0.5/12345/bert-base-uncased/0.9_errorOnlyWrongOnlySelfTestAll_Adjust_all_Entropy_KD0_FuseDataset0/0_1/imdb/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__chatglm3-6b-base_1000__flan-t5-xl_1000/12345/logits_label_of_all_epochs_6epochs.pth',
    #     ],
    #     "imdb-close-gpt": [
    #         './results/eval_on_real/dynamic_multi_local_data_new/bert-base-uncased/0.9_errorOnlyWrongOnlySelfTestAll_Adjust_all_Entropy_KD0_FuseDataset0/0_1/imdb/gpt-3.5-turbo-instruct_1000__gpt-4-turbo-preview_1000/12345/logits_label_of_all_epochs_6epochs.pth',
    #         './results/eval_on_real/dynamic_multi_local_data_accumulate/imdb/single_copied/bert-base-uncased/increasedTheta_KD1_FuseDataset1/fewshotK8_5_0.5/12345/bert-base-uncased/0.9_errorOnlyWrongOnlySelfTestAll_Adjust_all_Entropy_KD0_FuseDataset0/0_1/imdb/gpt-3.5-turbo-instruct_1000__gpt-4-turbo-preview_1000/12345/logits_label_of_all_epochs_6epochs.pth',
    #         './results/eval_on_real/dynamic_multi_local_data_accumulate/imdb/gpt-3.5-turbo-instruct_1000__gpt-4-turbo-preview_1000/bert-base-uncased/increasedTheta_KD1_FuseDataset1/fewshotK8_5_0.5/12345/bert-base-uncased/0.9_errorOnlyWrongOnlySelfTestAll_Adjust_all_Entropy_KD0_FuseDataset0/0_1/imdb/gpt-3.5-turbo-instruct_1000__gpt-4-turbo-preview_1000/12345/logits_label_of_all_epochs_6epochs.pth',
    #     ],
    #     "qnli": [
    #         # './results/eval_on_real/dynamic_multi_local_data_new/bert-base-uncased/0.9_errorOnlyWrongOnlySelfTestAll_Adjust_all_Entropy_KD0_FuseDataset0/0_1/qnli/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__chatglm3-6b-base_1000__flan-t5-xl_1000/12345/logits_label_of_all_epochs_3epochs.pth',
    #         './results/eval_on_real/dynamic_multi_local_data_new/bert-base-uncased/0.9_errorOnlyWrongOnlySelfTestAll_Adjust_all_Entropy_KD0_FuseDataset0/0_1/qnli/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__chatglm3-6b-base_1000__flan-t5-xl_1000/12345/logits_label_of_all_epochs_6epochs.pth',
    #         # './results/eval_on_real/dynamic_multi_local_data_accumulate/qnli/single_copied/bert-base-uncased/increasedTheta_KD1_FuseDataset1/fewshotK4_5_0.5/42/bert-base-uncased/0.9_errorOnlyWrongOnlySelfTestAll_Adjust_all_Entropy_KD0_FuseDataset0/0_1/qnli/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__chatglm3-6b-base_1000__flan-t5-xl_1000/12345/logits_label_of_all_epochs_3epochs.pth',
    #         './results/eval_on_real/dynamic_multi_local_data_accumulate/qnli/single_copied/bert-base-uncased/increasedTheta_KD1_FuseDataset1/fewshotK4_5_0.5/42/bert-base-uncased/0.9_errorOnlyWrongOnlySelfTestAll_Adjust_all_Entropy_KD0_FuseDataset0/0_1/qnli/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__chatglm3-6b-base_1000__flan-t5-xl_1000/12345/logits_label_of_all_epochs_6epochs.pth',
    #         # './results/eval_on_real/dynamic_multi_local_data_accumulate/qnli/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__flan-t5-xl_1000__chatglm3-6b-base_1000/bert-base-uncased/increasedTheta_KD1_FuseDataset1/fewshotK4_5_0.5/12345/bert-base-uncased/0.9_errorOnlyWrongOnlySelfTestAll_Adjust_all_Entropy_KD0_FuseDataset0/0_1/qnli/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__chatglm3-6b-base_1000__flan-t5-xl_1000/12345/logits_label_of_all_epochs_3epochs.pth',
    #         './results/eval_on_real/dynamic_multi_local_data_accumulate/qnli/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__flan-t5-xl_1000__chatglm3-6b-base_1000/bert-base-uncased/increasedTheta_KD1_FuseDataset1/fewshotK4_5_0.5/12345/bert-base-uncased/0.9_errorOnlyWrongOnlySelfTestAll_Adjust_all_Entropy_KD0_FuseDataset0/0_1/qnli/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__chatglm3-6b-base_1000__flan-t5-xl_1000/12345/logits_label_of_all_epochs_6epochs.pth',
    #     ],
    # }
    file_path = {
        "imdb": [
            # './results/eval_on_real/dynamic_multi_local_data_new/bert-base-uncased/0.9_errorOnlyWrongOnlySelfTestAll_Adjust_all_Entropy_KD0_FuseDataset0/0_1/imdb/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__chatglm3-6b-base_1000__flan-t5-xl_1000/12345/logits_label_of_all_epochs_3epochs.pth',
            './results/eval_on_real/dynamic_multi_local_data_new/bert-base-uncased/0.9_errorOnlyWrongOnlySelfTestAll_Adjust_all_Entropy_KD0_FuseDataset0/0_1/imdb/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__chatglm3-6b-base_1000__flan-t5-xl_1000/12345/logits_label_of_all_epochs_6epochs.pth',
            # './results/eval_on_real/dynamic_multi_local_data_accumulate/imdb/single_copied/bert-base-uncased/increasedTheta_KD1_FuseDataset1/fewshotK8_5_0.5/1/bert-base-uncased/0.9_errorOnlyWrongOnlySelfTestAll_Adjust_all_Entropy_KD0_FuseDataset0/0_1/imdb/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__chatglm3-6b-base_1000__flan-t5-xl_1000/12345/logits_label_of_all_epochs_3epochs.pth',
            './results/eval_on_real/dynamic_multi_local_data_accumulate/imdb/single_copied/bert-base-uncased/increasedTheta_KD1_FuseDataset1/fewshotK8_5_0.5/1/bert-base-uncased/0.9_errorOnlyWrongOnlySelfTestAll_Adjust_all_Entropy_KD0_FuseDataset0/0_1/imdb/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__chatglm3-6b-base_1000__flan-t5-xl_1000/12345/logits_label_of_all_epochs_6epochs.pth',
            # './results/eval_on_real/dynamic_multi_local_data_accumulate/imdb/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__flan-t5-xl_1000__chatglm3-6b-base_1000/bert-base-uncased/increasedTheta_KD1_FuseDataset1/fewshotK8_5_0.5/12345/bert-base-uncased/0.9_errorOnlyWrongOnlySelfTestAll_Adjust_all_Entropy_KD0_FuseDataset0/0_1/imdb/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__chatglm3-6b-base_1000__flan-t5-xl_1000/12345/logits_label_of_all_epochs_3epochs.pth',
            './results/eval_on_real/dynamic_multi_local_data_accumulate/imdb/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__flan-t5-xl_1000__chatglm3-6b-base_1000/bert-base-uncased/increasedTheta_KD1_FuseDataset1/fewshotK8_5_0.5/12345/bert-base-uncased/0.9_errorOnlyWrongOnlySelfTestAll_Adjust_all_Entropy_KD0_FuseDataset0/0_1/imdb/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__chatglm3-6b-base_1000__flan-t5-xl_1000/12345/logits_label_of_all_epochs_6epochs.pth',
        ],
        "imdb-close-gpt": [
            './results/eval_on_real/dynamic_multi_local_data_new/bert-base-uncased/0.9_errorOnlyWrongOnlySelfTestAll_Adjust_all_Entropy_KD0_FuseDataset0/0_1/imdb/gpt-3.5-turbo-instruct_1000__gpt-4-turbo-preview_1000/12345/logits_label_of_all_epochs_6epochs.pth',
            './results/eval_on_real/dynamic_multi_local_data_accumulate/imdb/single_copied/bert-base-uncased/increasedTheta_KD1_FuseDataset1/fewshotK8_5_0.5/12345/bert-base-uncased/0.9_errorOnlyWrongOnlySelfTestAll_Adjust_all_Entropy_KD0_FuseDataset0/0_1/imdb/gpt-3.5-turbo-instruct_1000__gpt-4-turbo-preview_1000/12345/logits_label_of_all_epochs_6epochs.pth',
            './results/eval_on_real/dynamic_multi_local_data_accumulate/imdb/gpt-3.5-turbo-instruct_1000__gpt-4-turbo-preview_1000/bert-base-uncased/increasedTheta_KD1_FuseDataset1/fewshotK8_5_0.5/12345/bert-base-uncased/0.9_errorOnlyWrongOnlySelfTestAll_Adjust_all_Entropy_KD0_FuseDataset0/0_1/imdb/gpt-3.5-turbo-instruct_1000__gpt-4-turbo-preview_1000/12345/logits_label_of_all_epochs_6epochs.pth',
        ],
        "qnli": [
            # './results/eval_on_real/dynamic_multi_local_data_new/bert-base-uncased/0.9_errorOnlyWrongOnlySelfTestAll_Adjust_all_Entropy_KD0_FuseDataset0/0_1/qnli/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__chatglm3-6b-base_1000__flan-t5-xl_1000/12345/logits_label_of_all_epochs_3epochs.pth',
            './results/eval_on_real/dynamic_multi_local_data_new/bert-base-uncased/0.9_errorOnlyWrongOnlySelfTestAll_Adjust_all_Entropy_KD0_FuseDataset0/0_1/qnli/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__chatglm3-6b-base_1000__flan-t5-xl_1000/12345/logits_label_of_all_epochs_6epochs.pth',
            # './results/eval_on_real/dynamic_multi_local_data_accumulate/qnli/single_copied/bert-base-uncased/increasedTheta_KD1_FuseDataset1/fewshotK4_5_0.5/42/bert-base-uncased/0.9_errorOnlyWrongOnlySelfTestAll_Adjust_all_Entropy_KD0_FuseDataset0/0_1/qnli/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__chatglm3-6b-base_1000__flan-t5-xl_1000/12345/logits_label_of_all_epochs_3epochs.pth',
            './results/eval_on_real/dynamic_multi_local_data_accumulate/qnli/single_copied/bert-base-uncased/increasedTheta_KD1_FuseDataset1/fewshotK4_5_0.5/42/bert-base-uncased/0.9_errorOnlyWrongOnlySelfTestAll_Adjust_all_Entropy_KD0_FuseDataset0/0_1/qnli/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__chatglm3-6b-base_1000__flan-t5-xl_1000/12345/logits_label_of_all_epochs_6epochs.pth',
            # './results/eval_on_real/dynamic_multi_local_data_accumulate/qnli/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__flan-t5-xl_1000__chatglm3-6b-base_1000/bert-base-uncased/increasedTheta_KD1_FuseDataset1/fewshotK4_5_0.5/12345/bert-base-uncased/0.9_errorOnlyWrongOnlySelfTestAll_Adjust_all_Entropy_KD0_FuseDataset0/0_1/qnli/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__chatglm3-6b-base_1000__flan-t5-xl_1000/12345/logits_label_of_all_epochs_3epochs.pth',
            './results/eval_on_real/dynamic_multi_local_data_accumulate/qnli/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__flan-t5-xl_1000__chatglm3-6b-base_1000/bert-base-uncased/increasedTheta_KD1_FuseDataset1/fewshotK4_5_0.5/12345/bert-base-uncased/0.9_errorOnlyWrongOnlySelfTestAll_Adjust_all_Entropy_KD0_FuseDataset0/0_1/qnli/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__chatglm3-6b-base_1000__flan-t5-xl_1000/12345/logits_label_of_all_epochs_6epochs.pth',
        ],
    }

    if isinstance(file_path, str):
        train_dy_metrics, plms, samples, accumulated_sample_count, NUM_MODELS, task_name = read_training_dynamics(metric_file=file_path)
    ###### single model, several epochs ######

    # # ###### several models, single epoch ######
    # file_path = './results/eval_on_real/multi_local_data_new/bert-base-uncased/0.9_errorOnlyWrongOnlySelfTestAll_Adjust_all_Entropy_KD0_FuseDataset0/0_1/imdb/gpt2-xl_10000__llama-2-7b-chat-hf_10000__vicuna-7b-1.5v_10000__opt-6.7b_10000__chatglm3-6b-base_10000__flan-t5-xl_10000/12345/'
    # # file_path = './results/eval_on_real/multi_local_data_flip/bert-base-uncased/0.9_errorOnlyWrongOnlySelfTestAll_Adjust_all_Entropy_KD0_FuseDataset0/0_1/imdb/gpt2-xl_10000__llama-2-7b-chat-hf_10000__vicuna-7b-1.5v_10000__opt-6.7b_10000__chatglm3-6b-base_10000__flan-t5-xl_10000/12345/'
    # train_dy_metrics, plms, samples, accumulated_sample_count, NUM_MODELS, task_name = read_training_dynamics_across_model(
    #                                                                                         metric_file=file_path,
    #                                                                                         logits_file = 'logits_withoutlabel_of_data.pth',
    #                                                                                         label_file = 'logits_label_of_all_epochs.pth',
    #                                                                                         )
    # # ###### several models, single epoch ######

    task_name = 'imdb'
    task_name = 'imdb-close-gpt'
    # task_name = 'qnli'
    file_path = file_path[task_name]

    if isinstance(file_path, str):
        for plm, sample in zip(plms, samples):
            train_dy_metrics[plm], _ = compute_train_dy_metrics(train_dy_metrics[plm])
            
            if 'flip' in file_path:
                dy_metrics_save_dir = f'./figure/dynamics/record/{sample}/{task_name}/flip/'
                plot_dy_save_dir = f'./figure/dynamics/{sample}/{task_name}/flip/'
            else:
                folder_name = 'original' if 'data_new' in file_path else ('single_progen' if 'single' in file_path else 'fusegen')
                num_epochs = 3 if '3epochs' in file_path else (6 if '6epochs' in file_path else (10 if '10epochs' in file_path else 0))
                dy_metrics_save_dir = f'./figure/dynamics/{folder_name}/record/{sample}/{num_epochs}epochs/{task_name}/'
                plot_dy_save_dir = f'./figure/dynamics/{folder_name}/{sample}/{num_epochs}epochs/{task_name}'
            if not os.path.exists(dy_metrics_save_dir):
                os.makedirs(dy_metrics_save_dir)
            if not os.path.exists(plot_dy_save_dir):
                os.makedirs(plot_dy_save_dir)
            
            train_dy_metrics[plm].to_json(f'{dy_metrics_save_dir}/{plm}.jsonl', orient='records', lines=True)
            _mean = np.mean(train_dy_metrics[plm]['variability'])
            _max = np.max(train_dy_metrics[plm]['variability'])
            _min = np.min(train_dy_metrics[plm]['variability'])
            _std = np.std(train_dy_metrics[plm]['variability'])
            print(f"model {plm} with {_mean=}, {_max=}, {_min=}, {_std=}")
            plot_data_map(train_dy_metrics[plm], f'{plot_dy_save_dir}/{plm}.pdf', title=f'{plm}-BERT Data Map ({sample} samples)', show_hist=True, model='bert-base-uncased')
            plot_data_map(train_dy_metrics[plm], f'{plot_dy_save_dir}/{plm}.png', title=f'{plm}-BERT Data Map ({sample} samples)', show_hist=True, model='bert-base-uncased')
    else:
        assert isinstance(file_path, list), "not list and not str"
        PLM_names = {'gpt2-xl':'GPT2', 'llama-2-7b-chat-hf':'Llama2', 'vicuna-7b-1.5v':'Vicuna', 'opt-6.7b':'OPT', 'chatglm3-6b-base':'ChatGLM3', 'flan-t5-xl':'Flan-T5', 'gpt-3.5-turbo-instruct':'GPT3.5', 'gpt-4-turbo-preview':'GPT4'}
        for _file_path in file_path:
            train_dy_metrics, plms, samples, accumulated_sample_count, NUM_MODELS, task_name = read_training_dynamics(metric_file=_file_path)
            for plm, sample in zip(plms, samples):
                train_dy_metrics[plm], _ = compute_train_dy_metrics(train_dy_metrics[plm])
                
                if 'flip' in _file_path:
                    dy_metrics_save_dir = f'./figure/dynamics/record/{sample}/{task_name}/flip/'
                    plot_dy_save_dir = f'./figure/dynamics/{sample}/{task_name}/flip/'
                else:
                    folder_name = 'original' if 'data_new' in _file_path else ('single_progen' if 'single' in _file_path else 'fusegen')
                    num_epochs = 3 if '3epochs' in _file_path else (6 if '6epochs' in _file_path else (10 if '10epochs' in _file_path else 0))
                    dy_metrics_save_dir = f'./figure/dynamics/{folder_name}/record/{sample}/{num_epochs}epochs/{task_name}/'
                    plot_dy_save_dir = f'./figure/dynamics/{folder_name}/{sample}/{num_epochs}epochs/{task_name}'
                if not os.path.exists(dy_metrics_save_dir):
                    os.makedirs(dy_metrics_save_dir)
                if not os.path.exists(plot_dy_save_dir):
                    os.makedirs(plot_dy_save_dir)
                
                train_dy_metrics[plm].to_json(f'{dy_metrics_save_dir}/{plm}.jsonl', orient='records', lines=True)
                _mean = np.mean(train_dy_metrics[plm]['variability'])
                _max = np.max(train_dy_metrics[plm]['variability'])
                _min = np.min(train_dy_metrics[plm]['variability'])
                _std = np.std(train_dy_metrics[plm]['variability'])
                print(f"model {plm} with {_mean=}, {_max=}, {_min=}, {_std=}")
                # plot_data_map(train_dy_metrics[plm], f'{plot_dy_save_dir}/{plm}.pdf', title=f'{plm}-BERT Data Map ({sample} samples)', show_hist=True, model='bert-base-uncased')
                # plot_data_map(train_dy_metrics[plm], f'{plot_dy_save_dir}/{plm}.png', title=f'{plm}-BERT Data Map ({sample} samples)', show_hist=True, model='bert-base-uncased')
                plot_data_map(train_dy_metrics[plm], f'{plot_dy_save_dir}/{plm}_graphOnly.pdf', title=f'{PLM_names[plm]}', show_hist=False, model='bert-base-uncased')
                plot_data_map(train_dy_metrics[plm], f'{plot_dy_save_dir}/{plm}_graphOnly.png', title=f'{PLM_names[plm]}', show_hist=False, model='bert-base-uncased')
                # plot_data_map(train_dy_metrics[plm], f'{plot_dy_save_dir}/{plm}.pdf', title=f'{PLM_names[plm]}', show_hist=True, model='bert-base-uncased')
                # plot_data_map(train_dy_metrics[plm], f'{plot_dy_save_dir}/{plm}.png', title=f'{PLM_names[plm]}', show_hist=True, model='bert-base-uncased')

    # ################# single plot ends #################

    
    # ################ comparison plot starts ################
    # ###### single model, several epochs ######
    # file_path = {
    #     'imdb': {
    #         # "original": './results/eval_on_real/multi_local_data_new/bert-base-uncased/0.9_errorOnlyWrongOnlySelf_new_increasedTheta_Entropy_KD0_FuseDataset0/0_1/imdb/gpt2-xl_100__llama-2-7b-chat-hf_100__vicuna-7b-1.5v_100__opt-6.7b_100__chatglm3-6b-base_100__flan-t5-xl_100/12345/logits_label_of_all_epochs.pth', # for debug, 3 epochs trained
    #         # "original": './results/eval_on_real/multi_local_data_new/bert-base-uncased/0.9_errorOnlyWrongOnlySelf_new_increasedTheta_Entropy_KD0_FuseDataset0/0_1/imdb/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__chatglm3-6b-base_1000__flan-t5-xl_1000/12345/logits_label_of_all_epochs_6epochs.pth',
    #         # "flipped": './results/eval_on_real/multi_local_data_flip/bert-base-uncased/0.9_errorOnlyWrongOnlySelf_new_increasedTheta_Entropy_KD0_FuseDataset0/0_1/imdb/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__chatglm3-6b-base_1000__flan-t5-xl_1000/12345/logits_label_of_all_epochs_6epochs.pth',
    #         "original": './results/eval_on_real/multi_local_data_new/bert-base-uncased/0.9_errorOnlyWrongOnlySelf_new_increasedTheta_Entropy_KD0_FuseDataset0/0_1/imdb/gpt2-xl_10000__llama-2-7b-chat-hf_10000__vicuna-7b-1.5v_10000__opt-6.7b_10000__chatglm3-6b-base_10000__flan-t5-xl_10000/12345/logits_label_of_all_epochs_6epochs.pth',
    #         "flipped": './results/eval_on_real/multi_local_data_flip/bert-base-uncased/0.9_errorOnlyWrongOnlySelf_new_increasedTheta_Entropy_KD0_FuseDataset0/0_1/imdb/gpt2-xl_10000__llama-2-7b-chat-hf_10000__vicuna-7b-1.5v_10000__opt-6.7b_10000__chatglm3-6b-base_10000__flan-t5-xl_10000/12345/logits_label_of_all_epochs_6epochs.pth',
    #     },
    #     'yelp': {
    #         # "original": './results/eval_on_real/multi_local_data_new/bert-base-uncased/0.9_errorOnlyWrongOnlySelf_new_increasedTheta_Entropy_KD0_FuseDataset0/0_1/yelp/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__chatglm3-6b-base_1000__flan-t5-xl_1000/12345/logits_label_of_all_epochs_6epochs.pth',
    #         "original": './results/eval_on_real/multi_local_data_new/bert-base-uncased/0.9_errorOnlyWrongOnlySelf_new_increasedTheta_Entropy_KD0_FuseDataset0/0_1/yelp/gpt2-xl_10000__llama-2-7b-chat-hf_10000__vicuna-7b-1.5v_10000__opt-6.7b_10000__chatglm3-6b-base_10000__flan-t5-xl_10000/12345/logits_label_of_all_epochs_6epochs.pth',
    #         # "flipped": 
    #     }
    # }
    # task = 'imdb'
    # train_dy_metrics, plms, samples, accumulated_sample_count, NUM_MODELS, task_name = read_training_dynamics_2file(
    #     original_metric_file=file_path[task]["original"], 
    #     flipped_metric_file=file_path[task]["flipped"],
    # )
    # ###### single model, several epochs ######

    # # ###### several models, single epoch ######
    # file_path = {
    #     'imdb': {
    #         # "original": './results/eval_on_real/multi_local_data_new/bert-base-uncased/0.9_errorOnlyWrongOnlySelfTestAll_Adjust_all_Entropy_KD0_FuseDataset0/0_1/imdb/gpt2-xl_100__llama-2-7b-chat-hf_100__vicuna-7b-1.5v_100__opt-6.7b_100__chatglm3-6b-base_100__flan-t5-xl_100/12345/',
    #         # "flipped": './results/eval_on_real/multi_local_data_flip/bert-base-uncased/0.9_errorOnlyWrongOnlySelfTestAll_Adjust_all_Entropy_KD0_FuseDataset0/0_1/imdb/gpt2-xl_100__llama-2-7b-chat-hf_100__vicuna-7b-1.5v_100__opt-6.7b_100__chatglm3-6b-base_100__flan-t5-xl_100/12345/',
    #         # "original": './results/eval_on_real/multi_local_data_new/bert-base-uncased/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD0_FuseDataset0/0_1/imdb/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__chatglm3-6b-base_1000__flan-t5-xl_1000/12345/',
    #         # "flipped": './results/eval_on_real/multi_local_data_flip/bert-base-uncased/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD0_FuseDataset0/0_1/imdb/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__chatglm3-6b-base_1000__flan-t5-xl_1000/12345/',
    #         "original": './results/eval_on_real/multi_local_data_new/bert-base-uncased/0.9_errorOnlyWrongOnlySelfTestAll_Adjust_all_Entropy_KD0_FuseDataset0/0_1/imdb/gpt2-xl_10000__llama-2-7b-chat-hf_10000__vicuna-7b-1.5v_10000__opt-6.7b_10000__chatglm3-6b-base_10000__flan-t5-xl_10000/12345/',
    #         "flipped": './results/eval_on_real/multi_local_data_flip/bert-base-uncased/0.9_errorOnlyWrongOnlySelfTestAll_Adjust_all_Entropy_KD0_FuseDataset0/0_1/imdb/gpt2-xl_10000__llama-2-7b-chat-hf_10000__vicuna-7b-1.5v_10000__opt-6.7b_10000__chatglm3-6b-base_10000__flan-t5-xl_10000/12345/',
    #     },
    #     'yelp': {
    #         # "original": './results/eval_on_real/multi_local_data_new/bert-base-uncased/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD0_FuseDataset0/0_1/yelp/gpt2-xl_1000__llama-2-7b-chat-hf_1000__vicuna-7b-1.5v_1000__opt-6.7b_1000__chatglm3-6b-base_1000__flan-t5-xl_1000/12345/',
    #         "original": './results/eval_on_real/multi_local_data_new/bert-base-uncased/0.9_errorOnlyWrongOnlySelf_Adjust_increasedTheta_Entropy_KD0_FuseDataset0/0_1/yelp/gpt2-xl_10000__llama-2-7b-chat-hf_10000__vicuna-7b-1.5v_10000__opt-6.7b_10000__chatglm3-6b-base_10000__flan-t5-xl_10000/12345/',
    #         # "flipped": 
    #     }
    # }
    # task = 'imdb'
    # train_dy_metrics, plms, samples, accumulated_sample_count, NUM_MODELS, task_name = read_training_dynamics_2file_across_model(
    #     original_metric_file=file_path[task]["original"], 
    #     flipped_metric_file=file_path[task]["flipped"],
    #     # logits_file = 'logits_withoutlabel_of_data_1epoch.pth',
    #     # label_file = 'logits_label_of_all_epochs_1epoch.pth',
    #     logits_file = 'logits_withoutlabel_of_data.pth',
    #     label_file = 'logits_label_of_all_epochs.pth',
    # )
    # # ###### several models, single epoch ######

    # train_dy_metrics["original_metric"] = {}
    # train_dy_metrics["diff_metric"] = {}
    # for plm, sample in zip(plms, samples):
    #     print(f"{plm=}")
    #     train_dy_metrics["original_metric"][plm], _, train_dy_metrics["diff_metric"][plm] = compute_train_dy_metrics_2file({"original": train_dy_metrics["original"][plm], "flipped": train_dy_metrics["flipped"][plm]})
    #     dy_metrics_save_dir = f'./figure/dynamics/record/{sample}/{task_name}/diff/'
    #     plot_dy_save_dir = f'./figure/dynamics/{sample}/{task_name}/diff/'
    #     if not os.path.exists(dy_metrics_save_dir):
    #         os.makedirs(dy_metrics_save_dir)
    #     if not os.path.exists(plot_dy_save_dir):
    #         os.makedirs(plot_dy_save_dir)
        
    #     plm_train_dy_metrics = {}
    #     for key in train_dy_metrics.keys():
    #         print(f"{key=}")
    #         plm_train_dy_metrics[key] = train_dy_metrics[key][plm]
    #     print(f"{plm_train_dy_metrics.keys()=}")
    #     plm_train_dy_metrics["original_metric"].to_json(f'{dy_metrics_save_dir}/{plm}_original.jsonl', orient='records', lines=True)
    #     plm_train_dy_metrics["diff_metric"].to_json(f'{dy_metrics_save_dir}/{plm}_diff.jsonl', orient='records', lines=True)

    #     plot_data_map_2file(plm_train_dy_metrics, f'{plot_dy_save_dir}/{plm}.pdf', title=f'{plm}-BERT Data Map ({sample} samples)', show_hist=True, model='bert-base-uncased')
    #     plot_data_map_2file(plm_train_dy_metrics, f'{plot_dy_save_dir}/{plm}.png', title=f'{plm}-BERT Data Map ({sample} samples)', show_hist=True, model='bert-base-uncased')
    # # ################# comparison plot ends #################