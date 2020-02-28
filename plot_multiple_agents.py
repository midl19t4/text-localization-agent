import os
import numpy as np
import statistics
import sys
import matplotlib.pyplot as plt

from config import CONFIG, print_config

"""
Creates combined plots of multiple agents.
usage:
- put evaluation results of multiple agents into one directory:
    - evaluation_dir
        - agent1
            - evaluation_data
            - ...
        - agent2
            - evaluation_data
            - ...
- set evaluation_data_path to the location of evaluation_dir
- run: python plot_multiple_agents.py
"""
class PlotMultipleAgents():
    def __init__(self):
        evaluation_data_path = '/home/joshua/Documents/HPI/Master/semester1/midl/final_evaluation/testting_1000_finish'
        agents = ['single_box', 'single_cross', 'single_cross_24', 'sum_box', 'sum_box_24', 'sum_cross', 'sum_cross_bars', 'sum_cross_wide']
        files = ['actions_per_image', 'ious_per_image', 'not_found_words_per_image']

        self.data = {}
        for key in agents:
            path = '/'.join([evaluation_data_path, key, 'evaluation_data'])
            self.data[key] = {}
            for f in files:
                filepath = ''.join([path, '/', f, '.txt'])
                try:
                    content = eval(self.read_file(filepath))
                except:
                    del self.data[key]
                    break
                self.data[key][f] = content

        self.plots_path = '/home/joshua/Documents/HPI/Master/semester1/midl/final_evaluation/testting_1000_finish/aggregated_plots'
        if not os.path.exists(self.plots_path):
            os.mkdir(self.plots_path)

        self.plot_ious()
        self.plot_action_counts_per_reset()
        self.boxplot('ious_per_image')
        self.boxplot('precision')

    def read_file(self, path):
        with open(path, 'r') as file:
            return file.read().strip()


    def flatten(self, to_flatten):
        result = []
        for l in to_flatten:
            for i in l:
                result.append(i)
        return result


    def plot_ious(self):
        fig = plt.figure(num=None, figsize=(18, 6), dpi=200, facecolor='w', edgecolor='k')
        plt.title('Sorted IoU\'s per Reset')
        for k, v in self.data.items():
            ious = sorted(self.flatten(v['ious_per_image']))

            x = [i for i in range(len(ious))]

            plt.plot(x, ious, label=k, figure=fig)
            plt.ylim(0.0, 1.0)
            plt.xlabel("reset")
            plt.ylabel("IoU")

        plt.legend()

        path = '/'.join([self.plots_path, "plot_of_ious.png"])
        plt.savefig(path)
        plt.clf()

    def plot_action_counts_per_reset(self):
        fig = plt.figure(num=None, figsize=(18, 6), dpi=200, facecolor='w', edgecolor='k')
        plt.title('Action Counts per Reset')
        for k, v in self.data.items():
            flattend_actions = self.flatten(v['actions_per_image'])
            action_counts = [len(actions) for actions in flattend_actions]
            x = [i for i in range(len(action_counts))]
            y = sorted(action_counts)

            plt.plot(x, y, label=k, figure=fig)
            plt.ylim(0.0, 45.0)
            plt.xlabel("Resets")
            plt.ylabel("Actions taken")

        plt.legend()

        path = '/'.join([self.plots_path, "plot_of_action_counts_per_reset.png"])
        plt.savefig(path)
        plt.clf()


    def boxplot(self, kind):
        title = f'{kind} Boxplot Evaluation'

        if kind == 'ious_per_image':
            data = [self.flatten(v['ious_per_image']) for v in self.data.values()]
        elif kind == 'precision':
            data = [self.precision(v['ious_per_image']) for v in self.data.values()]
        else:
            return 0

        path = ''.join([self.plots_path, '/', kind, '_boxplot.png'])
        labels = self.data.keys()
        num_boxes = len(labels)

        fig, ax1 = plt.subplots(figsize=(10, 8))
        fig.canvas.set_window_title(title)

        bp = ax1.boxplot(data, notch=0, sym='+', vert=1, whis=1.5)
        plt.setp(bp['boxes'], color='black')
        plt.setp(bp['whiskers'], color='black')
        plt.setp(bp['fliers'], color='red', marker='+')

        # Add a horizontal grid to the plot, but make it very light in color
        # so we can use it for reading data values but not be distracting
        ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                       alpha=0.5)

        # Hide these grid behind plot objects
        ax1.set_axisbelow(True)
        ax1.set_title(title)
        ax1.set_xlabel('Distribution')
        ax1.set_ylabel('Value')

        # Set the axes ranges and axes labels
        ax1.set_xlim(0.5, num_boxes + 0.5)
        top = 1
        bottom = 0
        ax1.set_ylim(bottom, top)
        ax1.set_xticklabels(labels,
                            rotation=45, fontsize=8)

        path = ''.join([self.plots_path, '/', title, '.png'])
        plt.savefig(path)
        plt.clf()


    def precision(self, ious_per_image):
        precisions = []
        for ious in ious_per_image:
            tp = len([iou for iou in ious if iou > 0.5])
            fp = len(ious) - tp
            precision = tp / (tp + fp) if tp + fp > 0 else 0
            precisions.append(precision)

        return precisions


    def recall(self):
        recalls = []
        for ious, not_found_words in zip(self.ious_per_image, self.not_found_words_per_image):
            tp = len([iou for iou in ious if iou > 0.5])
            fn = not_found_words

            recall = tp / (tp + fn) if tp + fn > 0 else 0
            recalls.append(recall)

        return recalls


if __name__ == '__main__':
    PlotMultipleAgents()
