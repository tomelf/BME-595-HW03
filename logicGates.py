from neural_network import NeuralNetwork
import torch

class AND(NeuralNetwork):
    def __init__(self):
        super(AND, self).__init__(3, 1)

    def forward(self, x, y):
        self.forward(torch.randn(3).type(torch.DoubleTensor))
        b1 = torch.ByteTensor([1 if x else 0])
        b2 = torch.ByteTensor([1 if y else 0])
        return torch.equal(torch.add(b1,b2).ge(2), torch.ByteTensor([1]))

    def train():
        pass

class OR(NeuralNetwork):
    def __init__(self):
        super(OR, self).__init__(3, 1)

    def forward(self, x, y):
        self.forward(torch.randn(3).type(torch.DoubleTensor))
        b1 = torch.ByteTensor([1 if x else 0])
        b2 = torch.ByteTensor([1 if y else 0])
        return torch.equal(torch.add(b1,b2).ge(1), torch.ByteTensor([1]))

    def train():
        pass

class NOT(NeuralNetwork):
    def __init__(self):
        super(NOT, self).__init__(3, 1)

    def forward(self, x):
        self.forward(torch.randn(3).type(torch.DoubleTensor))
        b1 = torch.ByteTensor([1 if x else 0])
        return torch.equal(~b1, torch.ByteTensor([1]))

    def train():
        pass

class XOR(NeuralNetwork):
    def __init__(self):
        super(XOR, self).__init__(3, 1)

    def forward(self, x, y):
        self.forward(torch.randn(3).type(torch.DoubleTensor))
        b1 = torch.ByteTensor([1 if x else 0])
        b2 = torch.ByteTensor([1 if y else 0])
        return torch.equal(torch.add(b1*~b2, ~b1*b2).ge(1), torch.ByteTensor([1]))

    def train():
        pass
