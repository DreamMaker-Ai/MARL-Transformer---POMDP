import os.path

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import math


def main():
    filetype = '/history/run-.-tag-mean_num_alive_red_ratio.csv'
    filelist = [
        '1_Baseline/trial-1',
        '1_Baseline/trial-2',
        '2_DeepTransformerBlk/trial-3',
        '../B_MARL Transformer - Decentralized/12_MARL-Transformer-Decentralized-rev/trial-1'
    ]
    legend_list = ['com=2', 'com=8', 'layered=4/com=4', 'MTD']
    colorlist = ['r', 'b', 'g', 'm', 'y']

    x_limit = 1e8
    window = 10

    for f, c, l in zip(filelist, colorlist, legend_list):
        ff = f + filetype
        csv_path = Path(__file__).parent / ff

        csv_df = pd.read_csv(csv_path)

        wall_time = csv_df[csv_df.columns[0]]
        diff_time = np.array(wall_time[1:]) - np.array(wall_time[:-1])
        diff_ids = np.where(diff_time > 1000)

        correct_time = wall_time - wall_time[0]

        for diff_id in diff_ids[0]:
            dt = wall_time[diff_id + 1] - wall_time[diff_id]
            correct_time[diff_id + 1:] = correct_time[diff_id + 1:] - dt

        step = csv_df[csv_df.columns[1]]
        alive_red_ratio = csv_df[csv_df.columns[2]]

        averaged_correct_time = []
        averaged_alive_red_ratio = []

        for idx in range(len(alive_red_ratio) - window + 1):
            averaged_correct_time.append(
                np.mean(correct_time[idx:idx + window])
            )

            averaged_alive_red_ratio.append(
                np.mean(alive_red_ratio[idx:idx + window])
            )

        averaged_correct_time = np.array(averaged_correct_time)
        averaged_alive_red_ratio = np.array(averaged_alive_red_ratio)

        plt.xlabel('learning time [hours]')
        plt.ylabel('mean alive red ratio')

        plt.plot(averaged_correct_time / 3600, averaged_alive_red_ratio, linestyle='solid',
                 color=c, alpha=0.7, linewidth=1, label=l)

        if max(correct_time / 3600) < x_limit:
            x_limit = max(correct_time / 3600)

    # plt.yscale('log')
    plt.title(f'Moving average of alive red ratio vs Learning time, window={window}')
    plt.grid(which="both")
    plt.minorticks_on()
    plt.legend()
    plt.xlim([0, math.ceil(x_limit)])

    savedir = Path(__file__).parent / 'history_plots'
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    savename = filetype.replace('/history/run-.-tag-', '')
    savename = savename.replace('.csv', '')
    plt.savefig(str(savedir) + '/' + savename + '-time.png', dpi=300)

    plt.show()


if __name__ == '__main__':
    main()
