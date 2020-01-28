import os
import numpy as np
from text_localization_environment import TextLocEnv
import chainer
import chainerrl
import statistics
import argparse
import json
import sys

from custom_model import CustomModel
from config import CONFIG, print_config, write_config


class TestAgent():
    def __init__(self, num_images, visualize):
        print_config()
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
        self.image_actions_counts = []
        self.num_images = num_images

        # initialize Environment
        # relative_paths = np.loadtxt(CONFIG['imagefile_path'], dtype=str)
        # images_base_path = os.path.dirname(CONFIG['imagefile_path'])
        # absolute_paths = [images_base_path + i.strip('.') for i in relative_paths]
        with open(CONFIG['imagefile_path'], 'r') as file:
            data = json.loads(file.read())
            relative_paths = [img['file_name'] for img in data]
            images_base_path = os.path.dirname(CONFIG['imagefile_path']) + '/'
            absolute_paths = [images_base_path + i.strip('.') for i in relative_paths]
            bboxes = [[((bbox[0], bbox[1]), (bbox[2], bbox[3])) for bbox in img['bounding_boxes']] for img in data]

        # bboxes = np.load(CONFIG['boxfile_path'], allow_pickle=True)

        self.env = TextLocEnv(absolute_paths, bboxes, CONFIG['gpu_id'], ior_marker=CONFIG['ior_marker'])

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

        # check if Result dir already exists
        if os.path.exists(CONFIG['resultdir_path']):
            print('Result Directory already exists! Exiting...')
            sys.exit(1)

        # create evaluation path
        evaluation_data_path = ''.join([CONFIG['resultdir_path'], '/evaluation_data'])
        os.makedirs(evaluation_data_path)

        write_config()

        if visualize:
            print("Create Visualization")
            visualization_path = ''.join([CONFIG['resultdir_path'], '/visualization'])
            os.mkdir(visualization_path)

            self.visualize(visualization_path)
        else:
            print("Create statistical Evaluation")
            self.test()

        # run evaluations
        self.safe_data(evaluation_data_path + '/ious_per_image', self.ious_per_image)
        self.safe_data(evaluation_data_path + '/rewards_per_image', self.rewards_per_image)
        self.safe_data(evaluation_data_path + '/not_found_words_per_image', self.not_found_words_per_image)
        self.safe_data(evaluation_data_path + '/actions_per_image', self.actions_per_image)

        self.save_evaluation(evaluation_data_path)

    def safe_data(self, path, data):
        with open(path + '.txt', 'w') as f:
            f.write(str(data))

    def test(self, before_step_callback=False, after_image_callback=False):
        for n in range(self.num_images):
            print('starting image with index: ' + str(n))

            state = self.env.reset(image_index=n, training=False)

            image_ious = []
            image_rewards = []
            image_actions = []
            timeouts = 0

            while timeouts <= 4:  # search for more words
                done = False
                steps = 0
                episode_actions = []

                while not done:  # take more steps
                    if steps == CONFIG['timeout']:
                        # timeout when 40 steps reached and no trigger done
                        timeouts += 1
                        break

                    if before_step_callback:
                        before_step_callback()

                    action = self.agent.act(state)
                    episode_actions.append(action)
                    state, reward, done, info = self.env.step(action)

                    if done:
                        image_ious.append(self.env.iou)
                        image_rewards.append(reward)
                        image_actions.append(episode_actions)
                    steps += 1

                if timeouts == 0:
                    state = self.env.reset(stay_on_image=True, training=False)
                else:
                    state = self.env.reset(stay_on_image=True, start_bbox=self.calculate_reset_box(timeouts),
                                           training=False)

            # save all actions on one image together
            self.actions_per_image.append(image_actions)

            self.ious_per_image.append(image_ious)
            self.rewards_per_image.append(image_rewards)
            self.not_found_words_per_image.append(len(self.env.episode_not_found_bboxes))

            if after_image_callback:
                after_image_callback(n)

    def calculate_reset_box(self, timeouts):
        if CONFIG['reset_mode'] == 'corners':
            left = int(self.env.episode_image.width * 0.25)
            top = int(self.env.episode_image.height * 0.25)
            right = int(self.env.episode_image.width * 0.75)
            bottom = int(self.env.episode_image.height * 0.75)
            if timeouts == 1:
                return np.array([0, 0, right, bottom])
            elif timeouts == 2:
                return np.array([0, top, right, self.env.episode_image.height])
            elif timeouts == 3:
                return np.array([left, top, self.env.episode_image.width, self.env.episode_image.height])
            elif timeouts == 4:
                return np.array([left, 0, self.env.episode_image.width, bottom])

        elif CONFIG['reset_mode'] == 'bars':
            height_quarter = int(self.env.episode_image.height / 4)
            height_third = int(self.env.episode_image.height / 3)

            if timeouts == 1:
                return np.array([0, 0, self.env.episode_image.width, height_third])
            elif timeouts == 2:
                return np.array([0, height_quarter, self.env.episode_image.width, height_quarter + height_third])
            elif timeouts == 3:
                return np.array(
                    [0, 2 * height_quarter, self.env.episode_image.width, 2 * height_quarter + height_third])
            elif timeouts == 4:
                return np.array(
                    [0, 3 * height_quarter, self.env.episode_image.width, 3 * height_quarter + height_third])

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

    def save_evaluation(self, path):
        print(f"Save Evaluation Results to {path}")
        out_path = '/'.join([path, "evaluation_summary.txt"])

        with open(out_path, "w") as file:
            file.write("==== Evaluation of Agent Performance ====\n")
            file.write(f"Minimal IoU:\t{self.min_iou()}\n")
            file.write(f"Minimal IoU:\t{self.min_iou()}\n")
            file.write(f"Maximal IoU:\t{self.max_iou()}\n")
            file.write(f"Average IoU:\t{self.mean_iou()}\n")
            file.write(f"Average IoU:\t{self.mean_iou()}\n")
            file.write(f"Median IoU:\t{self.median_iou()}\n")
            file.write(f"Minimal Number of Actions:\t{self.min_num_actions()}\n")
            file.write(f"Maximal Number of Actions:\t{self.max_num_actions()}\n")
            file.write(f"Average Number of Actions:\t{self.mean_num_actions()}\n")
            file.write(f"Median Number of Actions:\t{self.median_num_actions()}\n")

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
        if self.image_actions_counts == []:
            self.image_actions_counts = [sum(len(actions) for actions in reset_actions) for reset_actions in
                                         self.actions_per_image]

        return min(self.image_actions_counts)

    def max_num_actions(self):
        if self.image_actions_counts == []:
            self.image_actions_counts = [sum(len(actions) for actions in reset_actions) for reset_actions in
                                         self.actions_per_image]

        return max(self.image_actions_counts)

    def mean_num_actions(self):
        if self.image_actions_counts == []:
            self.image_actions_counts = [sum(len(actions) for actions in reset_actions) for reset_actions in
                                         self.actions_per_image]

        return statistics.mean(self.image_actions_counts)

    def median_num_actions(self):
        if self.image_actions_counts == []:
            self.image_actions_counts = [sum(len(actions) for actions in reset_actions) for reset_actions in
                                         self.actions_per_image]

        sorted_actions_counts = sorted(self.image_actions_counts)
        return statistics.median(sorted_actions_counts)

    def flatten(self, to_flatten):
        result = []
        for l in to_flatten:
            for i in l:
                result.append(i)
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('num_images', nargs=1, type=int, help='Path to config file', default=100)
    parser.add_argument('-v', '--visualize', action='store_true', help='Indicates if visualization should be saved.')
    args, _ = parser.parse_known_args()

    agent = TestAgent(num_images=args.num_images[0], visualize=args.visualize)
