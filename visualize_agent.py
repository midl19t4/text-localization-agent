import click
import os
import numpy as np
from text_localization_environment import TextLocEnv
from chainerrl.links.mlp import MLP
from chainerrl.links import Sequence
from chainerrl.experiments.train_agent import train_agent_with_evaluation
import chainer
import chainerrl
import sys
from collections import defaultdict
from PIL import Image

from custom_model import CustomModel

ACTION_MEANINGS = {0: 'right',
                   1: 'left',
                   2: 'up,',
                   3: 'down',
                   4: 'bigger',
                   5: 'smaller',
                   6: 'fatter',
                   7: 'taller',
                   8: 'trigger'
                   }

@click.command()
@click.option("--imagefile", "-i", default='image_locations.txt', help="Path to the file containing the image locations.", type=click.Path(exists=True))
@click.option("--boxfile", "-b", default='bounding_boxes.npy', help="Path to the bounding boxes.", type=click.Path(exists=True))
@click.option("--directory", "-d", default='agent', help="Path to the agent.", type=click.Path(exists=True))
@click.option("--gpu", "-g", default=-1, help="GPU to use.")
def main(imagefile, boxfile, directory, gpu):
    print(imagefile)
    print(boxfile)

    relative_paths = np.loadtxt(imagefile, dtype=str)
    images_base_path = os.path.dirname(imagefile)
    absolute_paths = [images_base_path + i.strip('.') for i in relative_paths]
    bboxes = np.load(boxfile, allow_pickle=True)

    env = TextLocEnv(absolute_paths, bboxes, -1)
    q_func = chainerrl.q_functions.SingleModelStateQFunctionWithDiscreteAction(CustomModel(9))
    optimizer = chainer.optimizers.Adam(eps=1e-2)
    optimizer.setup(q_func)
    replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)
    gamma = 0.95
    explorer = chainerrl.explorers.ConstantEpsilonGreedy(
        epsilon=0, random_action_func=env.action_space.sample)

    agent = chainerrl.agents.DQN(
        q_func, optimizer, replay_buffer, gamma, explorer,
        gpu=gpu,
        replay_start_size=500, update_interval=1,
        target_update_interval=100)
    agent.load(directory)
    actions = defaultdict(int)
    obs = env.reset()
    done = False
    i = 0
    while (not done) and i < 15:
        #print(i,j)
        action = agent.act(obs)
        actions[ACTION_MEANINGS[action]] += 1
        obs, reward, done, info = env.step(action)
        #j -= 1
        img = env.render(mode='human', return_as_file=True)
        img.save(f'img/{i}',"bmp")

        print(ACTION_MEANINGS[action], reward, done, info)
        #input()
        i += 1



if __name__ == '__main__':
    main()
