import chainer
import chainer.functions as F
import chainer.links as L


class ResNet50Alpha(chainer.Chain):

    def __init__(self):
        super(ResNet50Alpha, self).__init__()
        w = chainer.initializers.HeNormal()
        block = [3, 4, 6, 3]

        with self.init_scope():
            self.conv1 = L.Convolution2D(4, 64, 7, 2, 3, initialW=w, nobias=True)
            self.bn1 = L.GroupNormalization(32)
            self.res2 = BottleNeckBlock(block[0], 64, 64, 256, 1)
            self.res3 = BottleNeckBlock(block[1], 256, 128, 512)
            self.res4 = BottleNeckBlock(block[2], 512, 256, 1024)
            self.res5 = BottleNeckBlock(block[3], 1024, 512, 2048)

    def __call__(self, x):
        h = self.bn1(self.conv1(x))
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        h = self.res5(h)
        return h


class BottleNeckBlock(chainer.ChainList):

    def __init__(self, layer, in_size, ch, out_size, stride=2):
        super(BottleNeckBlock, self).__init__()
        self.add_link(BottleNeckA(in_size, ch, out_size, stride))
        for i in range(layer - 1):
            self.add_link(BottleNeckB(out_size, ch))

    def __call__(self, x):
        for f in self.children():
            x = f(x)
        return x


class BottleNeckA(chainer.Chain):

    def __init__(self, in_size, ch, out_size, stride=2):
        super(BottleNeckA, self).__init__()
        w = chainer.initializers.HeNormal()

        with self.init_scope():
            self.conv1 = L.Convolution2D(
                in_size, ch, 1, stride, 0, initialW=w, nobias=True)
            self.bn1 = L.GroupNormalization(32)
            self.conv2 = L.Convolution2D(
                ch, ch, 3, 1, 1, initialW=w, nobias=True)
            self.bn2 = L.GroupNormalization(32)
            self.conv3 = L.Convolution2D(
                ch, out_size, 1, 1, 0, initialW=w, nobias=True)
            self.bn3 = L.GroupNormalization(32)

            self.conv4 = L.Convolution2D(
                in_size, out_size, 1, stride, 0,
                initialW=w, nobias=True)
            self.bn4 = L.GroupNormalization(32)

    def __call__(self, x):
        h1 = F.relu(self.bn1(self.conv1(x)))
        h1 = F.relu(self.bn2(self.conv2(h1)))
        h1 = self.bn3(self.conv3(h1))
        h2 = self.bn4(self.conv4(x))

        return F.relu(h1 + h2)


class BottleNeckB(chainer.Chain):

    def __init__(self, in_size, ch):
        super(BottleNeckB, self).__init__()
        w = chainer.initializers.HeNormal()

        with self.init_scope():
            self.conv1 = L.Convolution2D(
                in_size, ch, 1, 1, 0, initialW=w, nobias=True)
            self.bn1 = L.GroupNormalization(32)
            self.conv2 = L.Convolution2D(
                ch, ch, 3, 1, 1, initialW=w, nobias=True)
            self.bn2 = L.GroupNormalization(32)
            self.conv3 = L.Convolution2D(
                ch, in_size, 1, 1, 0, initialW=w, nobias=True)
            self.bn3 = L.GroupNormalization(32)

    def __call__(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = self.bn3(self.conv3(h))

        return F.relu(h + x)
