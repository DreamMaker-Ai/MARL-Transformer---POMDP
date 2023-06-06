import os.path

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import math
import numpy as np


def main():
    filetype = '/history/run-.-tag-mean_num_alive_blue_ratio.csv'
    filelist = [
        '1_Baseline/trial-1',
        '1_Baseline/trial-2',
        '2_DeepTransformerBlk/trial-3',
        '../B_MARL Transformer - Decentralized/12_MARL-Transformer-Decentralized-rev/trial-1'
    ]
    legend_list = ['com=2', 'com=8', 'layered=4/com=4', 'MTD']
    colorlist = ['r', 'b', 'g', 'm', 'y']

    x_limit = 1e8
    window=10

    for f, c, l in zip(filelist, colorlist, legend_list):
        ff = f + filetype
        csv_path = Path(__file__).parent / ff

        csv_df = pd.read_csv(csv_path)

        wall_time = csv_df[csv_df.columns[0]]
        step = csv_df[csv_df.columns[1]]
        alive_blue_ratio = csv_df[csv_df.columns[2]]
        
        
        averaged_wall_time = []
        averaged_step = []
        averaged_blue_ratio = []

        for idx in range(len(alive_blue_ratio) - window + 1):
            averaged_wall_time.append(
                np.mean(wall_time[idx:idx + window])
            )

            averaged_step.append(
                np.mean(step[idx:idx + window])
            )

            averaged_blue_ratio.append(
                np.mean(alive_blue_ratio[idx:idx + window])
            )

        averaged_step = np.array(averaged_step)
        averaged_blue_ratio = np.array(averaged_blue_ratio)



        plt.xlabel('learning steps [k]')
        plt.ylabel('mean alive blue ratio')

        plt.plot(averaged_step / 1000, averaged_blue_ratio, linestyle='solid', color=c, alpha=0.7,
                 linewidth=1, label=l)

        if max(step / 1000) < x_limit:
            x_limit = max(step / 1000)

    # plt.yscale('log')
    plt.title(f'Moving average of alive blue ratio vs Learning steps, window={window}')
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
