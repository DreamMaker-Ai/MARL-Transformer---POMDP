import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from utils import compute_current_total_ef_and_force, add_channel_dim, \
    count_alive_agents, count_alive_platoons_and_companies

from utils import compute_red_observation_maps_3, compute_blue_observation_maps_3, \
    compute_engage_observation_maps_3


def movie_generator_test(images):
    """
    Reference code
    :param images: list of  numpy_array
    """
    plts = []
    fig = plt.figure(20)

    for image in images:
        img = plt.imshow(image, vmin=0, vmax=1, animated=True)
        title = plt.title('num_agents')
        plts.append([img] + [title])

    anim = animation.ArtistAnimation(fig, plts, interval=100, blit=True, repeat_delay=3000)
    anim.save('anim.gif', writer='imagemagick')
    # anim.save('anim.mp4', writer='ffmpeg')
    plt.clf()
    plt.cla()
    plt.close(fig)


def rgb_channel_maker(r_channel, g_channel, b_channel):
    """
    Make RGB numpy array: (grid_size,grid_size, 3)
    """
    rgb_channel = np.concatenate([r_channel, g_channel, b_channel], axis=2)

    if np.max(rgb_channel) > 1 or np.min(rgb_channel) < 0:
        raise ValueError

    return rgb_channel


class MakeAnimation:
    """
    normalized (efficiency x force) map and
    normalized (force) map of battlefield, reds, blues
    :return: RGB numpy array (grid_size,grid_size,3)
    """

    def __init__(self, env):
        battlefield = env.battlefield  # (grid_size,grid_size)
        self.battlefield = add_channel_dim(battlefield)  # (grid_size,grid_size,1)

        self.rgb_channel_forces = []  # map of effective forces
        self.rgb_channel_efficiencies = []  # map of efficiencies

        self.rgb_channel_engage_forces = []  # map of engage effective forces

        self.rgb_channel_forces_obs = []  # observation map of effective forces
        self.rgb_channel_efficiencies_obs = []  # observation map of efficiencies

        self.rgb_channel_engage_forces_obs = []  # observation map of engage effective forces

        self.num_alive_reds = []  # number of alive red
        self.num_alive_blues = []

        self.num_alive_reds_platoons = []
        self.num_alive_reds_companies = []

        self.num_alive_blues_platoons = []
        self.num_alive_blues_companies = []

        self.total_efficiency_reds = []  # efficiency r
        self.total_efficiency_blues = []  # effciency b

        self.total_force_reds = []  # effective R
        self.total_force_blues = []  # effective B

    def add_frame(self, env):
        """
        Call this method every time_step
        """
        # 2D map of effective ef and effective force: # (grid_size,grid_size)
        red_normalized_force, red_efficiency = compute_red_observation_maps_3(env)
        blue_normalized_force, blue_efficiency = compute_blue_observation_maps_3(env)

        engage_normalized_force = compute_engage_observation_maps_3(env)

        # Add channel dim : (grid_size,grid_size,1)
        red_normalized_force = add_channel_dim(red_normalized_force)
        red_efficiency = add_channel_dim(red_efficiency)
        blue_normalized_force = add_channel_dim(blue_normalized_force)
        blue_efficiency = add_channel_dim(blue_efficiency)

        engage_normalized_force = add_channel_dim(engage_normalized_force)

        """ normalized(force) map; (grid_size,grid_size,3) """
        r_channel_force = self.battlefield + red_normalized_force
        g_channel_force = self.battlefield
        b_channel_force = self.battlefield + blue_normalized_force

        rgb_channel_force = rgb_channel_maker(r_channel_force, g_channel_force, b_channel_force)

        self.rgb_channel_forces.append(rgb_channel_force)

        """ efficiency map; (grid_size,grid_size,3) """
        r_channel_efficiency = self.battlefield + red_efficiency
        g_channel_efficiency = self.battlefield
        b_channel_efficiency = self.battlefield + blue_efficiency

        rgb_channel_efficiency = \
            rgb_channel_maker(r_channel_efficiency, g_channel_efficiency, b_channel_efficiency)
        self.rgb_channel_efficiencies.append(rgb_channel_efficiency)

        """ normalized(engage force) map; (grid_size,grid_size,3) """
        r_channel_engage_force = self.battlefield
        g_channel_engage_force = self.battlefield + engage_normalized_force
        b_channel_engage_force = self.battlefield

        rgb_channel_engage_force = rgb_channel_maker(
            r_channel_engage_force, g_channel_engage_force, b_channel_engage_force)

        self.rgb_channel_engage_forces.append(rgb_channel_engage_force)

        """ Add number of alive agents """
        num_alive_reds = count_alive_agents(env.reds)
        num_alive_blues = count_alive_agents(env.blues)

        self.num_alive_reds.append(num_alive_reds)
        self.num_alive_blues.append(num_alive_blues)

        num_alive_reds_platoons, num_alive_reds_companies = \
            count_alive_platoons_and_companies(env.reds)
        num_alive_blues_platoons, num_alive_blues_companies = \
            count_alive_platoons_and_companies(env.blues)

        self.num_alive_reds_platoons.append(num_alive_reds_platoons)
        self.num_alive_reds_companies.append(num_alive_reds_companies)

        self.num_alive_blues_platoons.append(num_alive_blues_platoons)
        self.num_alive_blues_companies.append(num_alive_blues_companies)

        """ Add total effective_ef and effective_force """
        # effective rR, effective R, effective bB, effective B
        (total_ef_reds, total_force_reds,
         total_effective_ef_reds, total_effective_force_reds) = \
            compute_current_total_ef_and_force(env.reds)
        (total_ef_blues, total_force_blues,
         total_effective_ef_blues, total_effective_force_blues) = \
            compute_current_total_ef_and_force(env.blues)

        self.total_efficiency_reds.append(total_ef_reds / (total_force_reds + 1e-5))
        self.total_efficiency_blues.append(total_ef_blues / (total_force_blues + 1e-5))

        self.total_force_reds.append(total_effective_force_reds)
        self.total_force_blues.append(total_effective_force_blues)

    def movie_generator(self, fig, ax, content, features_1, features_2, env):
        dir_save = './test_engagement'
        if not os.path.exists(dir_save):
            os.mkdir(dir_save)

        fontsize = 12
        ims = []
        fig.tight_layout()

        ax[0].tick_params(labelbottom=False, bottom=False, labelleft=False, left=False)
        ax[1].tick_params(labelbottom=False, bottom=False, labelleft=False, left=False)

        if content == 'efficiency':
            ax[0].set_title('efficiency', size=fontsize)

        elif content == 'force':
            ax[0].set_title('norm(force)', size=fontsize)

        else:
            raise NotImplementedError()

        ax[1].set_title('engagement', size=fontsize)

        step = 0
        mul = 2  # default=1
        for feature1, feature2 in zip(features_1, features_2):
            txt1 = ax[0].text(0.0, -5.0 * mul,
                              ('time:' + str(np.round(step * env.config.dt, 2)) + ' sec'),
                              fontsize=fontsize)

            txt3 = ax[1].text(0.0, -5.0 * mul, 'num_alive:: reds:' + str(self.num_alive_reds[step]) +
                              ',  blues:' + str(self.num_alive_blues[step]),
                              fontsize=fontsize)

            txt4 = ax[1].text(0.0, -4.0 * mul, '   num_alive_reds:: platoons:' +
                              str(self.num_alive_reds_platoons[step]) +
                              ',  companies:' + str(self.num_alive_reds_companies[step]),
                              fontsize=fontsize)

            txt5 = ax[1].text(0.0, -3.0 * mul, '   num_alive_blues:: platoons:' +
                              str(self.num_alive_blues_platoons[step]) +
                              ',  companies:' + str(self.num_alive_blues_companies[step]),
                              fontsize=fontsize)

            if content == 'efficiency':
                efficiency_reds = np.round(self.total_efficiency_reds[step], 2)
                efficiency_blues = np.round(self.total_efficiency_blues[step], 2)
                txt2 = ax[1].text(
                    0.0, -2.0 * mul, 'efficiency_reds:' + str(efficiency_reds) +
                               ',  efficiency_blues:' + str(efficiency_blues), fontsize=fontsize)

            elif content == 'force':
                force_reds = np.round(self.total_force_reds[step], 2)
                force_blues = np.round(self.total_force_blues[step], 2)
                txt2 = ax[1].text(
                    0.0, -2.0 * mul,
                    'remaining: effective_force_reds:' + str(force_reds) +
                    ',  effective_force_blues:' + str(force_blues), fontsize=fontsize)

            else:
                raise NotImplementedError()

            im1 = ax[0].imshow(feature1, vmin=0, vmax=1, animated=True)
            im2 = ax[1].imshow(feature2, vmin=0, vmax=1, animated=True)

            ims.append([im1] + [txt1] + [txt3] + [txt4] + [txt5] + [txt2] + [im2])

            step += 1

        anim = animation.ArtistAnimation(fig, ims, interval=300, blit=True,
                                         repeat_delay=3000, repeat=True)

        if content == 'efficiency':
            filename = './test_engagement/efficiency'

        elif content == 'force':
            filename = './test_engagement/effective_force'

        else:
            raise NotImplementedError()

        # anim.save(filename + '.gif', writer='imagemagick')

        anim.save(filename + '.mp4', writer='ffmpeg')
        # For mp4, need 'sudo apt install ffmpeg' @ terminal

    def generate_movies(self, env):
        """
        Call this method
        """

        features_1 = self.rgb_channel_efficiencies
        features_2 = self.rgb_channel_engage_forces
        fig_1, ax_1 = plt.subplots(1, 2, figsize=(12.8, 9.6), tight_layout=True)

        self.movie_generator(fig_1, ax_1, 'efficiency', features_1, features_2, env)

        features_1 = self.rgb_channel_forces
        features_2 = self.rgb_channel_engage_forces
        fig_2, ax_2 = plt.subplots(1, 2, figsize=(12.8, 9.6), tight_layout=True)

        self.movie_generator(fig_2, ax_2, 'force', features_1, features_2, env)