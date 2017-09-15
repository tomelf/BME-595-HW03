from NeuralNetwork import NeuralNetwork
from mnist import MNIST
import torch

class AND(NeuralNetwork):
    def __init__(self):
        super(AND, self).__init__()
        self.build(2, 2, 1)

    def __call__(self, x, y):
        self.x = x
        self.y = y
        output = self.forward(self.x, self.y)
        return output

    def forward(self, x, y):
        x = 1 if x else 0
        y = 1 if y else 0
        output = super(AND, self).forward(torch.FloatTensor([x, y]))
        return output[0][0] > 0.5

    def train(self):
        output = self.forward(self.x, self.y)
        x = 1 if self.x else 0
        y = 1 if self.y else 0
        label = (x and y)
        super(AND, self).backward(torch.FloatTensor([label]))
        super(AND, self).updateParams(eta=0.2)

class OR(NeuralNetwork):
    def __init__(self):
        super(OR, self).__init__()
        self.build(2, 2, 1)

    def __call__(self, x, y):
        self.x = x
        self.y = y
        output = self.forward(self.x, self.y)
        return output

    def forward(self, x, y):
        x = 1 if x else 0
        y = 1 if y else 0
        output = super(OR, self).forward(torch.FloatTensor([x, y]))
        return output[0][0] > 0.5

    def train(self):
        output = self.forward(self.x, self.y)
        x = 1 if self.x else 0
        y = 1 if self.y else 0
        label = (x or y)
        super(OR, self).backward(torch.FloatTensor([label]))
        super(OR, self).updateParams(eta=0.2)

class NOT(NeuralNetwork):
    def __init__(self):
        super(NOT, self).__init__()
        self.build(1, 2, 1)

    def __call__(self, x):
        self.x = x
        output = self.forward(self.x)
        return output

    def forward(self, x):
        x = 1 if x else 0
        output = super(NOT, self).forward(torch.FloatTensor([x]))
        return output[0][0] > 0.5

    def train(self):
        output = self.forward(self.x)
        x = 1 if self.x else 0
        label = 1 if not x else 0
        super(NOT, self).backward(torch.FloatTensor([label]))
        super(NOT, self).updateParams(eta=0.2)

class XOR(NeuralNetwork):
    def __init__(self):
        super(XOR, self).__init__()
        self.build(2, 2, 1)

    def __call__(self, x, y):
        self.x = x
        self.y = y
        output = self.forward(self.x, self.y)
        return output

    def forward(self, x, y):
        x = 1 if x else 0
        y = 1 if y else 0
        output = super(XOR, self).forward(torch.FloatTensor([x, y]))
        return output[0][0] > 0.5

    def train(self):
        output = self.forward(self.x, self.y)
        x = 1 if self.x else 0
        y = 1 if self.y else 0
        label = (x ^ y)
        super(XOR, self).backward(torch.FloatTensor([label]))
        super(XOR, self).updateParams(eta=0.2)
