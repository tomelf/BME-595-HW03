from NeuralNetwork import NeuralNetwork
from mnist import MNIST
import torch

class AND(NeuralNetwork):
    def __init__(self):
        super(AND, self).__init__()
        self.build(2, 2, 1)

    def __call__(self, x, y):
        output = self.forward(x, y)
        self.train()
        return output

    def forward(self, x, y):
        self.x = 1 if x else 0
        self.y = 1 if y else 0
        output = super(AND, self).forward(torch.FloatTensor([self.x, self.y]))
        return output[0][0] > 0.5

    def train(self):
        label = (self.x and self.y)
        super(AND, self).backward(torch.FloatTensor([label]))
        super(AND, self).updateParams(eta=0.2)

class OR(NeuralNetwork):
    def __init__(self):
        super(OR, self).__init__()
        self.build(2, 2, 1)

    def __call__(self, x, y):
        output = self.forward(x, y)
        self.train()
        return output

    def forward(self, x, y):
        self.x = 1 if x else 0
        self.y = 1 if y else 0
        output = super(OR, self).forward(torch.FloatTensor([self.x, self.y]))
        return output[0][0] > 0.5

    def train(self):
        label = (self.x or self.y)
        super(OR, self).backward(torch.FloatTensor([label]))
        super(OR, self).updateParams(eta=0.2)

class NOT(NeuralNetwork):
    def __init__(self):
        super(NOT, self).__init__()
        self.build(1, 2, 1)

    def __call__(self, x):
        output = self.forward(x)
        self.train()
        return output

    def forward(self, x):
        self.x = 1 if x else 0
        output = super(NOT, self).forward(torch.FloatTensor([self.x]))
        return output[0][0] > 0.5

    def train(self):
        label = 1 if not self.x else 0
        super(NOT, self).backward(torch.FloatTensor([label]))
        super(NOT, self).updateParams(eta=0.2)

class XOR(NeuralNetwork):
    def __init__(self):
        super(XOR, self).__init__()
        self.build(2, 2, 1)

    def __call__(self, x, y):
        output = self.forward(x, y)
        self.train()
        return output

    def forward(self, x, y):
        self.x = 1 if x else 0
        self.y = 1 if y else 0
        output = super(XOR, self).forward(torch.FloatTensor([self.x, self.y]))
        return output[0][0] > 0.5

    def train(self):
        label = (self.x ^ self.y)
        super(XOR, self).backward(torch.FloatTensor([label]))
        super(XOR, self).updateParams(eta=0.2)
