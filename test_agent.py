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
    def __init__(self, num_images, visualize):
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
        self.actions_per_image, self.ious_per_image, self.rewards_per_image, self.not_found_words_per_image = [], [], [], []
        self.episode_actions_counts = []
        self.num_images = num_images

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

    def test(self, before_step_callback=False, after_image_callback=False):
        for n in range(self.num_images):
            state = self.env.reset(add_random_iors=False, image_index=18)

            image_ious = []
            image_rewards = []
            image_actions = []
            timeouts = 0

            while timeouts <= 4:  # search for more words
                print(image_ious)
                done = False
                steps = 0

                if timeouts == 0:
                    state = self.env.reset(stay_on_image=True, add_random_iors=False)
                else:
                    # reset initial bbox with 75% size of whole image frame
                    # adjusted corners
                    left = int(self.env.episode_image.width * 0.25)
                    top = int(self.env.episode_image.height * 0.25)
                    right = int(self.env.episode_image.width * 0.75)
                    bottom = int(self.env.episode_image.height * 0.75)
                    if timeouts == 1:
                        bbox = np.array([0, 0, right, bottom])
                    elif timeouts == 2:
                        bbox = np.array([0, top, right, self.env.episode_image.height])
                    elif timeouts == 3:
                        bbox = np.array([left, top, self.env.episode_image.width, self.env.episode_image.height])
                    elif timeouts == 4:
                        bbox = np.array([left, 0, self.env.episode_image.width, bottom])
                    state = self.env.reset(stay_on_image=True, start_bbox=bbox, add_random_iors=False)

                while not done:  # take more steps
                    print(steps)
                    if steps == 40:
                        # timeout when 40 steps reached and no trigger done
                        timeouts += 1
                        break

                    if before_step_callback:
                        before_step_callback()

                    action = self.agent.act(state)
                    image_actions.append(action)
                    state, reward, done, info = self.env.step(action)

                    if done:
                        image_ious.append(self.env.iou)
                        image_rewards.append(reward)
                    steps += 1

            # save all actions on one image together
            self.actions_per_image.append(image_actions)

            self.ious_per_image.append(image_ious)
            self.rewards_per_image.append(image_rewards)
            self.not_found_words_per_image.append(len(self.env.episode_not_found_bboxes))

            if after_image_callback:
                after_image_callback(n)


    def visualize(self, visualization_path):
        frames = []

        def before_step_callback():
            img = self.env.render(mode='human', return_as_file=True)
            frames.append(img)

        def after_image_callback(gif_id):
            frames[0].save(
                f'{visualization_path}/visualization_result_{gif_id}.gif',
                format='GIF',
                append_images=frames[1:],
                save_all=True,
                duration=200,
                loop=0)
            del frames[:]

        self.test(before_step_callback, after_image_callback)

    def print_evaluation(self):
        precisions = self.precision()
        recalls = self.recall()
        print("==== Evaluation of Agent Performance ====")
        print(f"Minimal IoU:\t{self.min_iou()}")
        print(f"Maximal IoU:\t{self.max_iou()}")
        print(f"Average IoU:\t{self.mean_iou()}")
        print(f"Median IoU:\t\t{self.median_iou()}")
        print(f"Minimal precision:\t{min(precisions)}")
        print(f"Maximal precision:\t{max(precisions)}")
        print(f"Average precision:\t{statistics.mean(precisions)}")
        print(f"Median precision:\t\t{statistics.median(precisions)}")
        print(f"Minimal recall:\t{min(recalls)}")
        print(f"Maximal recall:\t{max(recalls)}")
        print(f"Average recall:\t{statistics.mean(recalls)}")
        print(f"Median recall:\t\t{statistics.median(recalls)}")
        print(f"Minimal Number of Actions:\t{self.min_num_actions()}")
        print(f"Maximal Number of Actions:\t{self.max_num_actions()}")
        print(f"Average Number of Actions:\t{self.mean_num_actions()}")
        print(f"Median Number of Actions:\t{self.median_num_actions()}")

    def min_iou(self):
        return min(self.flatten(self.ious_per_image))

    def max_iou(self):
        return max(self.flatten(self.ious_per_image))

    def mean_iou(self):
        return statistics.mean(self.flatten(self.ious_per_image))

    def median_iou(self):
        sorted_ious = sorted(self.flatten(self.ious_per_image))
        return statistics.median(sorted_ious)

    def mean(self, arr):
        return statistics.mean(self.flatten(self.ious_per_image))

    def median(self, arr):
        sorted_ious = sorted(self.flatten(self.ious_per_image))
        return statistics.median(sorted_ious)


    def min_num_actions(self):
        if self.episode_actions_counts == []:
            self.episode_actions_counts = [len(action_list) for action_list in self.actions_per_image]

        return min(self.episode_actions_counts)

    def max_num_actions(self):
        if self.episode_actions_counts == []:
            self.episode_actions_counts = [len(action_list) for action_list in self.actions_per_image]

        return max(self.episode_actions_counts)

    def mean_num_actions(self):
        if self.episode_actions_counts == []:
            self.episode_actions_counts = [len(action_list) for action_list in self.actions_per_image]

        return statistics.mean(self.episode_actions_counts)

    def median_num_actions(self):
        if self.episode_actions_counts == []:
            self.episode_actions_counts = [len(action_list) for action_list in self.actions_per_image]

        sorted_actions_counts = sorted(self.episode_actions_counts)
        return statistics.median(sorted_actions_counts)

    def flatten(self, to_flatten):
        result = []
        for l in to_flatten:
            for i in l:
                result.append(i)
        return result

    def hist_actions(self, plots_path):
        actions = self.flatten(self.actions_per_image)
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
        x = [i for i in range(len(self.flatten(self.ious_per_image)))]
        y = self.flatten(self.ious_per_image)

        plt.plot(x, y)
        plt.xticks(x)
        plt.ylim(0.0, 1.0)
        plt.xlabel("resets")

        path = '/'.join([plots_path, "plot_of_ious.png"])
        plt.savefig(path)
        plt.clf()

    def plot_rewards(self, plots_path):
        x = [i for i in range(len(self.rewards_per_image))]
        y = self.rewards_per_image

        plt.plot(x, y)
        plt.xticks(x)
        plt.ylim(0.0, 100.0)
        plt.xlabel("resets")

        path = '/'.join([plots_path, "plot_of_rewards.png"])
        plt.savefig(path)
        plt.clf()

    def precision(self):
        precisions = []
        for ious in self.ious_per_image:
            tp = len([iou for iou in ious if iou > 0.5])
            fp = len(ious) - tp
            precision = tp / (tp + fp)
            precisions.append(precision)

        return precisions

    def recall(self):
        recalls = []
        for ious, not_found_words in zip(self.ious_per_image, self.not_found_words_per_image):
            tp = len([iou for iou in ious if iou > 0.5])
            fn = not_found_words

            recall = tp / (tp + fn)
            recalls.append(recall)

        return recalls


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('num_images', nargs=1, type=int, help='Path to config file', default=100)
    parser.add_argument('-v', '--visualize', action='store_true', help='Indicates if visualization should be saved.')
    args, _ = parser.parse_known_args()

    agent = TestAgent(num_images=args.num_images[0], visualize=args.visualize)
    print(agent.ious_per_image)
    print(agent.rewards_per_image)
    print(agent.actions_per_image)

