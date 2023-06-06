import numpy as np
import gym


class Config:
    def __init__(self):

        # self.model_dir = 'models/model_1500/'  # newest file -> 'ls -ltr'  For training, not test
        self.model_dir = None

        if self.model_dir:  # starting steps for continual training
            self.n0 = None  # learner update cycles. Should be read from tensorboard
            self.actor_cycles = None  # Should be read from tensorboard
        else:
            self.n0 = 0
            self.actor_cycles = 0

        # Define simulation cond.
        self.show_each_episode_result = True  # mainly for debug
        self.draw_win_distributions = False  # mainly for debug
        self.max_episodes_test_play = 1  # default=50 for training

        # Animation setting
        self.make_animation = True  # Use self.max_episodes_test_play=1

        # Time plot of a test setting
        self.make_time_plot = True  # Use self.max_episodes_test_play=1

        # POMDP params
        self.fov = 2  # default=2
        self.com = 8  # default=2 / trial-1:2 / trial-2: 8

        # Define environment parameters
        self.grid_size = 50  # default=15
        self.offset = 0  # blue-team offset from edges

        # Define gym spaces
        self.action_dim = 5
        self.action_space = gym.spaces.Discrete(self.action_dim)

        observation_low = 0.
        observation_high = 1.
        self.observation_channels = 4
        self.n_frames = 4
        self.observation_space = \
            gym.spaces.Box(low=observation_low,
                           high=observation_high,
                           shape=(2 * self.fov + 1,
                                  2 * self.fov + 1,
                                  self.observation_channels)
                           )

        # Buffer
        self.capacity = 10000  # default=100000
        self.compress = True
        self.prioritized_replay = True

        # Neural nets parameters
        self.hidden_dim = 64  # default=64
        self.key_dim = 64  # default=64
        self.num_heads = 2

        self.dropout_rate = 0.2  # default=0.2

        # Training parameters
        self.actor_rollout_steps = 100  # default=100
        self.num_update_cycles = 1000000
        self.actor_rollouts_before_train = 50  # default=50
        self.batch_size = 32  # Default=32
        self.num_minibatchs = 30  # bach_sizeのminibatchの数/1 update_cycle of learner, default=30
        self.tau = 0.01  # Soft update of target network
        self.gamma = 0.96
        self.max_steps = 400  # Default = 100 for training, 200 for the test

        self.learning_rate = 5e-5  # Default = 5e-5
        self.loss_coef = 1.0  # Default = 1.0

        # Define Lanchester simulation parameters
        self.threshold = 5.0  # min of forces R & B
        self.log_threshold = np.log(self.threshold)
        self.mul = 2.0  # Minimum platoon force = threshold * mul
        self.dt = .2  # Default=.2

        # Define possible agent parameters
        self.agent_types = ('platoon', 'company')
        self.agent_forces = (50, 150)

        # Define possible red / blue agent parameters
        self.red_platoons = (150, 150)  # num range of red platoons, default=(3,10)
        self.red_companies = (150, 150)  # num range of red companies, default=(2,5)

        self.blue_platoons = (250, 250)  # num range of blue platoons, default=(3,10)
        self.blue_companies = (250, 250)  # num range of blue companies, default=(2,5)

        self.efficiencies_red = (0.3, 0.5)  # range
        self.efficiencies_blue = (0.3, 0.5)

        # For paddiing of multi-agents, *4 for adding red agents, defaule=*1
        self.max_num_red_agents = (self.red_platoons[1] + self.red_companies[1]) * 1  # nominal
        # self.max_num_red_agents = (self.red_platoons[1] + self.red_companies[1]) * 3  # add agents

        # Red team TBD parameters
        self.R0 = None  # initial total force, set in 'generate_red_team'
        self.log_R0 = None
        self.num_red_agents = None  # set in 'define_red_team'
        self.num_red_platoons = None
        self.num_red_companies = None

        # Blue team TBD parameters
        self.B0 = None  # initial total force, set in 'generate_blue_team'
        self.log_B0 = None
        self.num_blue_agents = None  # set in 'define_blue_team'
        self.num_blue_platoons = None
        self.num_blue_companies = None

    def define_blue_team(self):
        """
        Called from reset
            self.num_blue_agents, self.num_blue_platoons, self.num_blue_companies
            will be allocated.
        """
        self.num_blue_platoons = \
            np.random.randint(
                low=self.blue_platoons[0],
                high=self.blue_platoons[1] + 1)

        self.num_blue_companies = \
            np.random.randint(
                low=self.blue_companies[0],
                high=self.blue_companies[1] + 1)

        self.num_blue_agents = self.num_blue_platoons + self.num_blue_companies

    def define_red_team(self):
        """
        Called from reset
            self.num_red_agents, self.num_red_platoons, self.num_red_companies
            will be allocated.
        """
        self.num_red_platoons = \
            np.random.randint(
                low=self.red_platoons[0],
                high=self.red_platoons[1] + 1)

        self.num_red_companies = \
            np.random.randint(
                low=self.red_companies[0],
                high=self.red_companies[1] + 1)

        self.num_red_agents = self.num_red_platoons + self.num_red_companies

    def reset(self):
        """
        Generate new config for new episode
        """
        self.define_blue_team()
        self.define_red_team()


if __name__ == '__main__':
    config = Config()

    for _ in range(3):
        config.reset()
