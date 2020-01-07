import os
import numpy as np
from text_localization_environment import TextLocEnv
import chainer
import chainerrl
import statistics
import matplotlib.pyplot as plt
from matplotlib import colors
import argparse

from custom_model import CustomModel
from config import CONFIG, print_config


class TestAgent():
    def __init__(self, resets, visualize):
        # class variables
        self.action_meanings = {
            0: 'right',
            1: 'left',
            2: 'up',
            3: 'down',
            4: 'bigger',
            5: 'smaller',
            6: 'fatter',
            7: 'taller',
            8: 'trigger'
        }
        self.actions, self.ious, self.rewards = [], [], []
        self.episode_actions_counts = []
        self.resets = resets

        # initialize Environment
        relative_paths = np.loadtxt(CONFIG['imagefile_path'], dtype=str)
        images_base_path = os.path.dirname(CONFIG['imagefile_path'])
        absolute_paths = [images_base_path + i.strip('.') for i in relative_paths]

        bboxes = np.load(CONFIG['boxfile_path'], allow_pickle=True)

        self.env = TextLocEnv(absolute_paths, bboxes, -1)

        # Initialize Agent
        q_func = chainerrl.q_functions.SingleModelStateQFunctionWithDiscreteAction(CustomModel(9))
        optimizer = chainer.optimizers.Adam(eps=1e-2)
        optimizer.setup(q_func)
        replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=CONFIG['replay_buffer_capacity'])

        explorer = chainerrl.explorers.ConstantEpsilonGreedy(
            epsilon=0,
            random_action_func=self.env.action_space.sample)

        self.agent = chainerrl.agents.DQN(
            q_func,
            optimizer,
            replay_buffer,
            CONFIG['gamma'],
            explorer,
            gpu=CONFIG['gpu_id'],
            replay_start_size=CONFIG['replay_start_size'],
            update_interval=CONFIG['update_interval'],
            target_update_interval=CONFIG['target_update_interval'])
        self.agent.load(CONFIG['agentdir_path'])

        # run test
        if visualize:
            print("Create Visualization")
            visualization_path = ''.join([CONFIG['resultdir_path'], '/visualization'])
            if not os.path.exists(visualization_path):
                os.mkdir(visualization_path)
            self.visualize(visualization_path)
        else:
            print("Create statistical Evaluation")
            self.test()

        # run evaluations
        plots_path = ''.join([CONFIG['resultdir_path'], '/plots'])
        if not os.path.exists(plots_path):
            os.mkdir(plots_path)

        self.print_evaluation()
        self.hist_actions(plots_path)
        self.plot_ious(plots_path)
        self.plot_rewards(plots_path)


    def test(self):
        for n in range(self.resets):
            steps = 0
            print("begin reset")
            state = self.env.reset()
            print("finished reset")

            done = False
            iter_actions = []
            while (not done) and steps < 100:
                action = self.agent.act(state)
                iter_actions.append(action)
                state, reward, done, info = self.env.step(action)

                if done:
                    self.ious.append(self.env.iou)
                    self.rewards.append(reward)
                    self.actions.append(iter_actions)
                steps += 1

    def visualize(self, visualization_path):
        frames = []
        for gif_id in range(self.resets):
            steps = 0
            print("begin reset")
            state = self.env.reset()
            print("finished reset")

            done = False
            iter_actions = []
            while (not done) and steps < 100:
                action = self.agent.act(state)
                iter_actions.append(action)
                state, reward, done, info = self.env.step(action)

                img = self.env.render(mode='human', return_as_file=True)
                frames.append(img)

                if done:
                    self.ious.append(self.env.iou)
                    self.rewards.append(reward)
                    self.actions.append(iter_actions)
                steps += 1

            frames[0].save(
                f'{visualization_path}/visualization_result_{gif_id}.gif',
                format='GIF',
                append_images=frames[1:],
                save_all=True,
                duration=200,
                loop=0)


    def print_evaluation(self):
        print("==== Evaluation of Agent Performance ====")
        print(f"Minimal IoU:\t{self.min_iou()}")
        print(f"Maximal IoU:\t{self.max_iou()}")
        print(f"Average IoU:\t{self.mean_iou()}")
        print(f"Median IoU:\t\t{self.median_iou()}")
        print(f"Minimal Number of Actions:\t{self.min_num_actions()}")
        print(f"Maximal Number of Actions:\t{self.max_num_actions()}")
        print(f"Average Number of Actions:\t{self.mean_num_actions()}")
        print(f"Median Number of Actions:\t{self.median_num_actions()}")

    def min_iou(self):
        return min(self.ious)

    def max_iou(self):
        return max(self.ious)

    def mean_iou(self):
        return statistics.mean(self.ious)

    def median_iou(self):
        sorted_ious = sorted(self.ious)
        return statistics.median(sorted_ious)


    def min_num_actions(self):
        if self.episode_actions_counts == []:
            self.episode_actions_counts = [len(action_list) for action_list in self.actions]

        return min(self.episode_actions_counts)

    def max_num_actions(self):
        if self.episode_actions_counts == []:
            self.episode_actions_counts = [len(action_list) for action_list in self.actions]

        return max(self.episode_actions_counts)

    def mean_num_actions(self):
        if self.episode_actions_counts == []:
            self.episode_actions_counts = [len(action_list) for action_list in self.actions]

        return statistics.mean(self.episode_actions_counts)

    def median_num_actions(self):
        if self.episode_actions_counts == []:
            self.episode_actions_counts = [len(action_list) for action_list in self.actions]

        sorted_actions_counts = sorted(self.episode_actions_counts)
        return statistics.median(sorted_actions_counts)


    def hist_actions(self, plots_path):
        actions = []
        for action_list in self.actions:
            for a in action_list:
                actions.append(a)
        actions.sort()
        action_types = [self.action_meanings[a] for a in set(actions)]
        fig, axs = plt.subplots(1, 1)

        # N is the count in each bin, bins is the lower-limit of the bin
        N, bins, patches = axs.hist(actions)

        # We'll color code by height, but you could use any scalar
        fracs = N / N.max()

        # we need to normalize the data to 0..1 for the full range of the colormap
        norm = colors.Normalize(fracs.min(), fracs.max())

        # Now, we'll loop through our objects and set the color of each accordingly
        for thisfrac, thispatch in zip(fracs, patches):
            color = plt.cm.viridis(norm(thisfrac))
            thispatch.set_facecolor(color)

        axs.set_xticks(list(set(actions)))
        axs.set_xticklabels(action_types, rotation='horizontal', fontsize=10)

        path = '/'.join([plots_path, "histogram_of_chosen_actions.png"])
        plt.savefig(path)
        plt.clf()

    def plot_ious(self, plots_path):
        x = [i for i in range(len(self.ious))]
        y = self.ious

        plt.plot(x, y)
        plt.xticks(x)
        plt.ylim(0.0, 1.0)
        plt.xlabel("resets")

        path = '/'.join([plots_path, "plot_of_ious.png"])
        plt.savefig(path)
        plt.clf()

    def plot_rewards(self, plots_path):
        x = [i for i in range(len(self.rewards))]
        y = self.rewards

        plt.plot(x, y)
        plt.xticks(x)
        plt.ylim(0.0, 100.0)
        plt.xlabel("resets")

        path = '/'.join([plots_path, "plot_of_rewards.png"])
        plt.savefig(path)
        plt.clf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('num_resets', nargs=1, type=int, help='Path to config file', default=100)
    parser.add_argument('-v', '--visualize', action='store_true', help='Indicates if visualization should be saved.')
    args, _ = parser.parse_known_args()

    agent = TestAgent(resets=args.num_resets[0], visualize=args.visualize)
    print(agent.ious)
    print(agent.rewards)
    print(agent.actions)

