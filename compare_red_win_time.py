import os.path

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import math


def main():
    filetype = '/history/run-.-tag-num_red_win.csv'
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
        diff_ids = np.where(diff_time > 900)

        correct_time = wall_time - wall_time[0]

        for diff_id in diff_ids[0]:
            dt = wall_time[diff_id + 1] - wall_time[diff_id]
            correct_time[diff_id + 1:] = correct_time[diff_id + 1:] - dt

        wall_time = csv_df[csv_df.columns[0]]
        step = csv_df[csv_df.columns[1]]
        num_red_win = csv_df[csv_df.columns[2]]

        averaged_correct_time = []
        averaged_num_red_win = []

        for idx in range(len(num_red_win) - window + 1):
            averaged_correct_time.append(
                np.mean(correct_time[idx:idx + window])
            )

            averaged_num_red_win.append(
                np.mean(num_red_win[idx:idx + window])
            )

        averaged_correct_time = np.array(averaged_correct_time)
        averaged_num_red_win = np.array(averaged_num_red_win)

        plt.xlabel('learning time [hours]')
        plt.ylabel('reds win ratio')

        plt.plot(averaged_correct_time / 3600, averaged_num_red_win / 50, linestyle='solid',
                 color=c, alpha=0.7, linewidth=1, label=l)

        if max(correct_time / 3600) < x_limit:
            x_limit = max(correct_time / 3600)

    # plt.yscale('log')
    plt.title(f'Moving average of reds win ratio vs Learning time, window={window}')
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
