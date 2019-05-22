import click
import os
import numpy as np
from text_localization_environment import TextLocEnv
from chainerrl.links.mlp import MLP
from chainer import links as L
from chainerrl.links import Sequence
from chainerrl.experiments.train_agent import train_agent_with_evaluation
import chainer
from chainer import Chain
from chainer import functions as F
import chainerrl
import sys
from collections import defaultdict

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


class CustomModel(Chain):
    def __init__(self, n_actions):
        super(CustomModel, self).__init__()
        with self.init_scope():
            self.resNet=L.ResNet50Layers()
            self.l1=L.Linear(2139, 1024)
            self.l2=L.Linear(1024, 1024)
            self.l3=L.Linear(1024, n_actions)


    def forward(self, x):
        image, history, penalty = x[0], x[1], x[2]
        image = F.reshape(image, (-1,3,224,224))
        history = F.reshape(history.astype('float32'),(-1,90))
        penalty = F.reshape(penalty,(-1,1))
        h1 = F.relu(self.resNet(image, layers=['pool5'])['pool5'])
        h1 = F.reshape(F.concat((h1, history, penalty), axis=1), (-1,2139))
        h2 = F.relu(self.l1(h1))
        h3 = F.relu(self.l2(h2))
        return F.relu(self.l3(h3))


@click.command()
@click.option("--imagefile", "-i", default='image_locations.txt', help="Path to the file containing the image locations.", type=click.Path(exists=True))
@click.option("--boxfile", "-b", default='bounding_boxes.npy', help="Path to the bounding boxes.", type=click.Path(exists=True))
@click.option("--directory", "-d", default='agent', help="Path to the agent.", type=click.Path(exists=True))
def main(imagefile, boxfile, directory):
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
        gpu=-1,
        replay_start_size=500, update_interval=1,
        target_update_interval=100)
    agent.load(directory)

    actions = defaultdict(int)
    for i in range(10):
        done = False
        obs = env.reset()
        j = 50
        while (not done) and (j > 0) :
            print(i,j)
            action = agent.act(obs)
            actions[action] += 1
            obs, reward, done, info = env.step(action)
            j -= 1
    print(actions)
            #env.render()
            #print(ACTION_MEANINGS[action], reward, done, info)
            #input()



if __name__ == '__main__':
    main()
