import numpy as np
import os
import re
import pickle
import matplotlib.pyplot as plt
from collections import deque
import json
from pathlib import Path


def make_indrease_num_data(agent_type, nominal_data_dir, robust_data_dir):
    num_red_win_list = []
    num_blue_win_list = []
    num_no_contest_list = []

    num_alive_reds_ratio_list = []
    num_alive_blues_ratio_list = []

    remaining_red_effective_force_ratio_list = []
    remaining_blue_effective_force_ratio_list = []

    episode_rewards_list = []
    episode_lens_list = []

    with open(nominal_data_dir) as f:
        json_data = json.load(f)

        num_red_win_list.append(json_data['num_red_win'] / 1000)
        num_blue_win_list.append(json_data['num_blue_win'] / 1000)
        num_no_contest_list.append(json_data['no_contest'] / 1000)

        num_alive_reds_ratio_list.append(json_data['num_alive_reds_ratio'])
        num_alive_blues_ratio_list.append(json_data['num_alive_blues_ratio'])

        remaining_red_effective_force_ratio_list. \
            append(json_data['remaining_red_effective_force_ratio'])
        remaining_blue_effective_force_ratio_list. \
            append(json_data['remaining_blue_effective_force_ratio'])

        episode_rewards_list.append(json_data['episode_rewards'])
        episode_lens_list.append(json_data['episode_lens'])

    # agent_type = 'platoons', 'companies'

    if agent_type == 'platoons' or agent_type == 'blue_platoons' or agent_type == 'red_platoons':
        file_dir = ['(11,20)', '(21,30)', '(31,40)', '(41,50)']
    elif agent_type == 'companies':
        file_dir = ['(6,10)', '(11,20)', '(21,30)', '(31,40)', '(41,50)']
    else:
        raise NotImplementedError()

    for file_name in file_dir:
        child_dir = agent_type + '=' + file_name + '/result.json'

        with open(robust_data_dir + child_dir, 'r') as f:
            json_data = json.load(f)

            num_red_win_list.append(json_data['num_red_win'] / 1000)
            num_blue_win_list.append(json_data['num_blue_win'] / 1000)
            num_no_contest_list.append(json_data['no_contest'] / 1000)

            num_alive_reds_ratio_list.append(json_data['num_alive_reds_ratio'])
            num_alive_blues_ratio_list.append(json_data['num_alive_blues_ratio'])

            remaining_red_effective_force_ratio_list. \
                append(json_data['remaining_red_effective_force_ratio'])
            remaining_blue_effective_force_ratio_list. \
                append(json_data['remaining_blue_effective_force_ratio'])

            episode_rewards_list.append(json_data['episode_rewards'])
            episode_lens_list.append(json_data['episode_lens'])

    return [
        num_red_win_list, num_blue_win_list, num_no_contest_list,
        num_alive_reds_ratio_list, num_alive_blues_ratio_list,
        remaining_red_effective_force_ratio_list, remaining_blue_effective_force_ratio_list,
        episode_rewards_list, episode_lens_list,
    ]


def main():
    """
    agent_type = 'platoons'
    make_test_results_graph_of_increase_number(agent_type)

    agent_type = 'companies'
    make_test_results_graph_of_increase_number(agent_type)

    agent_type = 'blue_platoons'
    make_test_results_graph_of_increase_number(agent_type)
    """

    agent_type = 'platoons'  # 'platoons' or 'blue_platoons'

    num_red_win_lists = []
    num_blue_win_lists = []
    num_no_contest_lists = []

    num_alive_reds_ratio_lists = []
    num_alive_blues_ratio_lists = []

    remaining_red_effective_force_ratio_lists = []
    remaining_blue_effective_force_ratio_lists = []

    episode_rewards_lists = []
    episode_lens_lists = []

    """ trial-1 """
    nominal_data_dir = '1_Baseline/trial-1/nominal/result_1000/result.json'

    parent_dir_1 = '1_Baseline/trial-1' + '/robustness/1_robustness_to_num_agents/'
    robust_data_dir = parent_dir_1 + 'change_num_of_' + agent_type + '/'

    increase_num_data = make_indrease_num_data(agent_type, nominal_data_dir, robust_data_dir)

    num_red_win_lists.append(increase_num_data[0])
    num_blue_win_lists.append(increase_num_data[1])
    num_no_contest_lists.append(increase_num_data[2])

    num_alive_reds_ratio_lists.append(increase_num_data[3])
    num_alive_blues_ratio_lists.append(increase_num_data[4])

    remaining_red_effective_force_ratio_lists.append(increase_num_data[5])
    remaining_blue_effective_force_ratio_lists.append(increase_num_data[6])

    episode_rewards_lists.append(increase_num_data[7])
    episode_lens_lists.append(increase_num_data[8])

    """ trial-2 """
    nominal_data_dir = '1_Baseline/trial-2/nominal/result_1000/result.json'

    parent_dir_1 = '1_Baseline/trial-2' + '/robustness/1_robustness_to_num_agents/'
    robust_data_dir = parent_dir_1 + 'change_num_of_' + agent_type + '/'

    increase_num_data = make_indrease_num_data(agent_type, nominal_data_dir, robust_data_dir)

    num_red_win_lists.append(increase_num_data[0])
    num_blue_win_lists.append(increase_num_data[1])
    num_no_contest_lists.append(increase_num_data[2])

    num_alive_reds_ratio_lists.append(increase_num_data[3])
    num_alive_blues_ratio_lists.append(increase_num_data[4])

    remaining_red_effective_force_ratio_lists.append(increase_num_data[5])
    remaining_blue_effective_force_ratio_lists.append(increase_num_data[6])

    episode_rewards_lists.append(increase_num_data[7])
    episode_lens_lists.append(increase_num_data[8])

    """ trial 3 """
    nominal_data_dir = '2_DeepTransformerBlk/trial-3/nominal/result_1000/result.json'

    parent_dir_1 = '2_DeepTransformerBlk/trial-3' + '/robustness/1_robustness_to_num_agents/'
    robust_data_dir = parent_dir_1 + 'change_num_of_' + agent_type + '/'

    increase_num_data = make_indrease_num_data(agent_type, nominal_data_dir, robust_data_dir)

    num_red_win_lists.append(increase_num_data[0])
    num_blue_win_lists.append(increase_num_data[1])
    num_no_contest_lists.append(increase_num_data[2])

    num_alive_reds_ratio_lists.append(increase_num_data[3])
    num_alive_blues_ratio_lists.append(increase_num_data[4])

    remaining_red_effective_force_ratio_lists.append(increase_num_data[5])
    remaining_blue_effective_force_ratio_lists.append(increase_num_data[6])

    episode_rewards_lists.append(increase_num_data[7])
    episode_lens_lists.append(increase_num_data[8])

    num_red_win_lists = np.array(num_red_win_lists)
    num_blue_win_lists = np.array(num_blue_win_lists)
    num_no_contest_lists = np.array(num_no_contest_lists)

    num_alive_reds_ratio_lists = np.array(num_alive_reds_ratio_lists)
    num_alive_blues_ratio_lists = np.array(num_alive_blues_ratio_lists)

    remaining_red_effective_force_ratio_lists = \
        np.array(remaining_red_effective_force_ratio_lists)
    remaining_blue_effective_force_ratio_lists = \
        np.array(remaining_blue_effective_force_ratio_lists)

    episode_rewards_lists = np.array(episode_rewards_lists)
    episode_lens_lists = np.array(episode_lens_lists)

    savedir = Path(__file__).parent
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    if agent_type == 'platoons' or agent_type == 'blue_platoons' or agent_type == 'red_platoons':
        x = np.array([6.5, 15.5, 25.5, 35.5, 45.5])
    elif agent_type == 'companies':
        x = np.array([3.5, 8.0, 15.5, 25.5, 35.5, 45.5])
    else:
        NotImplementedError()

    for i, l in enumerate(['-', '--', ':']):
        plt.plot(x, num_red_win_lists[i], color='r', marker='o', linestyle=l)
        plt.plot(x, num_blue_win_lists[i], color='b', marker='o', linestyle=l)
        plt.plot(x, num_no_contest_lists[i], color='g', marker='s', linestyle=l)
    plt.title(f'solid:trial-1, dashed: trial-2, dotted: trial-3')
    plt.xlabel('num (bin range center) of ' + agent_type)
    plt.ylabel('win ratio')
    plt.ylim(-0.05, 1.05)
    plt.minorticks_on()
    plt.legend(["red win", "blue win", "no contest"])
    plt.grid()

    savename = 'comparison_win_ratio_of_increase_number-' + agent_type
    plt.savefig(str(savedir) + '/' + savename + '.png', dpi=300)
    plt.show()

    for i, l in enumerate(['-', '--', ':']):
        plt.plot(x, num_alive_reds_ratio_lists[i], color='r', marker='o', linestyle=l)
        plt.plot(x, num_alive_blues_ratio_lists[i], color='b', marker='o', linestyle=l)
    plt.title(f'solid:trial-1, dashed: trial-2, dotted: trial-3')
    plt.xlabel('num (bin range center) of ' + agent_type)
    plt.ylabel('alive agents ratio')
    plt.ylim(-0.05, 1.05)
    plt.minorticks_on()
    plt.legend(["red alive agents ratio", "blue alive agents ratio"])
    plt.grid()
    # plt.yscale('log')

    savename = 'comparison_alive_agents_ratio_of_increase_number-' + agent_type
    plt.savefig(str(savedir) + '/' + savename + '.png', dpi=300)
    plt.show()

    for i, l in enumerate(['-', '--', ':']):
        plt.plot(x, remaining_red_effective_force_ratio_lists[i],
                 color='r', marker='o', linestyle=l)
        plt.plot(x, remaining_blue_effective_force_ratio_lists[i],
                 color='b', marker='o', linestyle=l)
    plt.title(f'solid:trial-1, dashed: trial-2, dotted: trial-3')
    plt.xlabel('num (bin range center) of ' + agent_type)
    plt.ylabel('total remaining force ratio')
    plt.ylim(-0.05, 1.05)
    plt.minorticks_on()
    plt.legend(["red remaining force", "blue remaining force"])
    plt.grid()
    # plt.yscale('log')

    savename = 'comparison_remaining_force_of_increase_number-' + agent_type
    plt.savefig(str(savedir) + '/' + savename + '.png', dpi=300)
    plt.show()

    for i, l in enumerate(['-', '--', ':']):
        plt.plot(x, episode_rewards_lists[i], color='r', marker='o', linestyle=l)
        plt.plot(x, episode_lens_lists[i], color='b', marker='o', linestyle=l)
    plt.title(f'solid:trial-1, dashed: trial-2, dotted: trial-3')
    plt.xlabel('num (bin range center) of ' + agent_type)
    plt.ylabel('rewards / length')
    plt.minorticks_on()
    plt.legend(["rewards", "length"])
    plt.grid()

    savename = 'comparison_rewards_length_of_increase_number-' + agent_type
    plt.savefig(str(savedir) + '/' + savename + '.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    main()
