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
import logging
import sys
from tb_chainer import SummaryWriter
import time
import re
import chainer.computational_graph as c

class CustomModel(Chain):
    def __init__(self, n_actions):
        super(CustomModel, self).__init__()
        with self.init_scope():
            self.resNet=L.ResNet50Layers()
            self.l1=L.Linear(None, 1024)
            self.l2=L.Linear(None, 1024)
            self.l3=L.Linear(None, n_actions)


    def forward(self, x):
        #import pdb; pdb.set_trace()
        image, history, penalty = x[0], x[1], x[2]
        image = F.reshape(image, (1,3,224,224))
        history = F.reshape(history.astype('float32'),(1,-1))
        penalty = F.reshape(penalty,(1,-1))
        h1 = F.relu(self.resNet(image, layers=['pool5'])['pool5'])
        h1 = F.reshape(F.concat((h1, history, penalty), axis=1), (1,-1))
        h2 = F.relu(self.l1(h1))
        h3 = F.relu(self.l2(h2))
        return F.relu(self.l3(h3))


@click.command()
@click.option("--imagefile", "-i", default='image_locations.txt', help="Path to the file containing the image locations.", type=click.Path(exists=True))
@click.option("--boxfile", "-b", default='bounding_boxes.npy', help="Path to the bounding boxes.", type=click.Path(exists=True))
def main(imagefile, boxfile):
    print(imagefile)
    print(boxfile)

    relative_paths = np.loadtxt(imagefile, dtype=str)
    images_base_path = os.path.dirname(imagefile)
    absolute_paths = [images_base_path + i.strip('.') for i in relative_paths]
    bboxes = np.load(boxfile, allow_pickle=True)

    env = TextLocEnv(absolute_paths, bboxes, -1)
    m = CustomModel(10)
    print('asfasdf')
    vs = [m(env.reset())]
    g = c.build_computational_graph(vs)
    with open('graph.dot', 'w') as o:
        o.write(g.dump())

if __name__ == '__main__':
    main()
