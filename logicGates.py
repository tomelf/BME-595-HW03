from NeuralNetwork import NeuralNetwork
from mnist import MNIST
import torch

class AND(NeuralNetwork):
    def __init__(self):
        super(AND, self).__init__()

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        b1 = torch.ByteTensor([1 if x else 0])
        b2 = torch.ByteTensor([1 if y else 0])
        return torch.equal(torch.add(b1,b2).ge(2), torch.ByteTensor([1]))

    def train(self):
        mndata = MNIST('./python-mnist/data')
        training = mndata.load_training()
        x = torch.DoubleTensor(training[0]).transpose(0, 1)
        labels = []
        for i in training[1]:
            l = [0] * 10
            l[i] = 1
            labels.append(l)
        y = torch.DoubleTensor(labels).transpose(0, 1)

        self.build(x.size()[0], x.size()[0]/2, 2*y.size()[0], y.size()[0])

        eta = 0.8

        super(AND, self).forward(x)
        super(AND, self).backward(y)
        self.updateParams(eta)

        for i in range(len(self.Theta)):
            th = self.Theta[i]
            dE_dTheta = self.dE_dTheta[i]
            self.Theta[i] = th - self.eta * dE_dTheta

class OR(NeuralNetwork):
    def __init__(self):
        super(OR, self).__init__()

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        b1 = torch.ByteTensor([1 if x else 0])
        b2 = torch.ByteTensor([1 if y else 0])
        return torch.equal(torch.add(b1,b2).ge(1), torch.ByteTensor([1]))

    def train(self):
        mndata = MNIST('./python-mnist/data')
        training = mndata.load_training()
        x = torch.DoubleTensor(training[0]).transpose(0, 1)
        labels = []
        for i in training[1]:
            l = [0] * 10
            l[i] = 1
            labels.append(l)
        y = torch.DoubleTensor(labels).transpose(0, 1)

        self.build(x.size()[0], x.size()[0]/2, 2*y.size()[0], y.size()[0])

        eta = 0.8

        super(OR, self).forward(x)
        super(OR, self).backward(y)
        self.updateParams(eta)

        for i in range(len(self.Theta)):
            th = self.Theta[i]
            dE_dTheta = self.dE_dTheta[i]
            self.Theta[i] = th - self.eta * dE_dTheta

class NOT(NeuralNetwork):
    def __init__(self):
        super(NOT, self).__init__()

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        b1 = torch.ByteTensor([1 if x else 0])
        return torch.equal(~b1, torch.ByteTensor([1]))

    def train(self):
        mndata = MNIST('./python-mnist/data')
        training = mndata.load_training()
        x = torch.DoubleTensor(training[0]).transpose(0, 1)
        labels = []
        for i in training[1]:
            l = [0] * 10
            l[i] = 1
            labels.append(l)
        y = torch.DoubleTensor(labels).transpose(0, 1)

        self.build(x.size()[0], x.size()[0]/2, 2*y.size()[0], y.size()[0])

        eta = 0.8

        super(NOT, self).forward(x)
        super(NOT, self).backward(y)
        self.updateParams(eta)

        for i in range(len(self.Theta)):
            th = self.Theta[i]
            dE_dTheta = self.dE_dTheta[i]
            self.Theta[i] = th - self.eta * dE_dTheta

class XOR(NeuralNetwork):
    def __init__(self):
        super(XOR, self).__init__()

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        b1 = torch.ByteTensor([1 if x else 0])
        b2 = torch.ByteTensor([1 if y else 0])
        return torch.equal(torch.add(b1*~b2, ~b1*b2).ge(1), torch.ByteTensor([1]))

    def train(self):
        mndata = MNIST('./python-mnist/data')
        training = mndata.load_training()
        x = torch.DoubleTensor(training[0]).transpose(0, 1)
        labels = []
        for i in training[1]:
            l = [0] * 10
            l[i] = 1
            labels.append(l)
        y = torch.DoubleTensor(labels).transpose(0, 1)

        self.build(x.size()[0], x.size()[0]/2, 2*y.size()[0], y.size()[0])

        eta = 0.8

        super(XOR, self).forward(x)
        super(XOR, self).backward(y)
        self.updateParams(eta)

        for i in range(len(self.Theta)):
            th = self.Theta[i]
            dE_dTheta = self.dE_dTheta[i]
            self.Theta[i] = th - self.eta * dE_dTheta
