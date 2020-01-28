import os
import numpy as np
import statistics
import matplotlib.pyplot as plt

from config import CONFIG, print_config


class VisualizeAgent():
    def __init__(self):
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

        evaluation_data_path = ''.join([CONFIG['resultdir_path'], '/evaluation_data'])

        self.ious_per_image = eval(self.read_file(evaluation_data_path + '/ious_per_image.txt'))
        self.rewards_per_image = eval(self.read_file(evaluation_data_path + '/rewards_per_image.txt'))
        self.not_found_words_per_image = eval(self.read_file(evaluation_data_path + '/not_found_words_per_image.txt'))
        self.actions_per_image = eval(self.read_file(evaluation_data_path + '/actions_per_image.txt'))

        plots_path = ''.join([CONFIG['resultdir_path'], '/plots'])
        if not os.path.exists(plots_path):
            os.mkdir(plots_path)

        self.hist_actions(plots_path)
        self.plot_action_counts_per_reset(plots_path)
        self.plot_ious(plots_path)
        self.plot_rewards(plots_path)
        self.roc_curve(plots_path)

        data = [self.flatten(self.ious_per_image), self.precision(), self.recall()]
        path = '/'.join([plots_path, 'boxplot_overview.png'])
        labels = ["IoU", "Precision", "Recall"]
        self.boxplot(path, data, labels)


    def read_file(self, path):
        with open(path, 'r') as file:
            return file.read().strip()


    def flatten(self, to_flatten):
        result = []
        for l in to_flatten:
            for i in l:
                result.append(i)
        return result


    def hist_actions(self, plots_path):
        actions = self.flatten(self.flatten(self.actions_per_image))
        actions.sort()
        counted_actions = [actions.count(i) for i in self.action_meanings.keys()]
        action_types = self.action_meanings.values()
        index = np.arange(len(self.action_meanings.keys()))
        fig, axs = plt.subplots(1, 1)

        axs.bar(index, counted_actions, 0.9)
        plt.xticks(index, action_types)

        path = '/'.join([plots_path, "histogram_of_chosen_actions.png"])
        plt.savefig(path)
        plt.clf()


    def plot_ious(self, plots_path):
        ious = self.flatten(self.ious_per_image)
        x = [i for i in range(len(ious))]
        y = sorted(ious)

        plt.plot(x, y)
        plt.ylim(0.0, 1.0)
        plt.xlabel("resets")

        path = '/'.join([plots_path, "plot_of_ious.png"])
        plt.savefig(path)
        plt.clf()

    def plot_action_counts_per_reset(self, plots_path):
        flattend_actions = self.flatten(self.actions_per_image)
        action_counts = [len(actions) for actions in flattend_actions]
        x = [i for i in range(len(action_counts))]
        y = sorted(action_counts)

        plt.plot(x, y)
        plt.ylim(0.0, 40.0)
        plt.xlabel("resets")
        plt.ylabel("Actions taken")

        path = '/'.join([plots_path, "plot_of_action_counts_per_reset.png"])
        plt.savefig(path)
        plt.clf()


    def plot_rewards(self, plots_path):
        rewards = self.flatten(self.rewards_per_image)
        x = [i for i in range(len(rewards))]
        y = sorted(rewards)

        plt.plot(x, y)
        plt.ylim(0.0, 100.0)
        plt.xlabel("resets")

        path = '/'.join([plots_path, "plot_of_rewards.png"])
        plt.savefig(path)
        plt.clf()


    def roc_curve(self, plots_path):
        precisions = sorted(self.precision())
        tpr = sorted(self.recall())

        roc_auc = np.trapz(precisions, tpr)

        plt.figure()
        lw = 2
        plt.plot(precisions, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        path = '/'.join([plots_path, "adapted_roc_curve.png"])
        plt.savefig(path)
        plt.clf()


    def boxplot(self, path, data, labels):
        num_boxes = len(data)

        fig, ax1 = plt.subplots(figsize=(5, 5))
        fig.canvas.set_window_title('Boxplot Evaluation')

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
        ax1.set_title('Boxplot Evaluation of captured metrics')
        ax1.set_xlabel('Distribution')
        ax1.set_ylabel('Value')

        # Set the axes ranges and axes labels
        ax1.set_xlim(0.5, num_boxes + 0.5)
        top = 1
        bottom = 0
        ax1.set_ylim(bottom, top)
        ax1.set_xticklabels(labels,
                            rotation=45, fontsize=8)

        plt.savefig(path)
        plt.clf()


    def precision(self):
        precisions = []
        for ious in self.ious_per_image:
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

    def append_evaluation(self, path):
        out_path = '/'.join([path, "evaluation_summary.txt"])

        with open(out_path, "a") as file:
            file.write(f"Minimal precision:\t{min(precisions)}\n")
            file.write(f"Maximal precision:\t{max(precisions)}\n")
            file.write(f"Average precision:\t{statistics.mean(precisions)}\n")
            file.write(f"Average precision:\t{statistics.mean(precisions)}\n")
            file.write(f"Median precision:\t{statistics.median(precisions)}\n")
            file.write(f"Minimal recall:\t{min(recalls)}\n")
            file.write(f"Maximal recall:\t{max(recalls)}\n")
            file.write(f"Average recall:\t{statistics.mean(recalls)}\n")
            file.write(f"Median recall:\t{statistics.median(recalls)}\n")

if __name__ == '__main__':
    VisualizeAgent()
