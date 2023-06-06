import os.path

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import math
import numpy as np


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

    x_limit = 1e10
    window = 10

    for f, c, l in zip(filelist, colorlist, legend_list):
        ff = f + filetype
        csv_path = Path(__file__).parent / ff

        csv_df = pd.read_csv(csv_path)

        wall_time = csv_df[csv_df.columns[0]]
        step = csv_df[csv_df.columns[1]]
        num_red_win = csv_df[csv_df.columns[2]]

        averaged_wall_time = []
        averaged_step = []
        averaged_num_red_win = []

        for idx in range(len(num_red_win) - window + 1):
            averaged_wall_time.append(
                np.mean(wall_time[idx:idx + window])
            )

            averaged_step.append(
                np.mean(step[idx:idx + window])
            )

            averaged_num_red_win.append(
                np.mean(num_red_win[idx:idx + window])
            )

        averaged_step = np.array(averaged_step)
        averaged_num_red_win = np.array(averaged_num_red_win)

        plt.xlabel('learning steps [k]')
        plt.ylabel('reds win ratio')

        plt.plot(averaged_step / 1000, averaged_num_red_win / 50, linestyle='solid', color=c,
                 alpha=0.7, linewidth=1, label=l)

        if max(averaged_step / 1000) < x_limit:
            x_limit = max(averaged_step / 1000)

    # plt.yscale('log')
    plt.title(f'Moving Average of reds win ratio vs learning steps, window={window}')
    plt.grid(which="both")
    plt.minorticks_on()
    plt.legend()
    plt.xlim([0, math.ceil(x_limit)])

    savedir = Path(__file__).parent / 'history_plots'
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    savename = filetype.replace('/history/run-.-tag-', '')
    savename = savename.replace('.csv', '')
    plt.savefig(str(savedir) + '/' + savename + '.png', dpi=300)

    plt.show()


if __name__ == '__main__':
    main()
