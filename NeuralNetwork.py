import torch
import math

class NeuralNetwork(object):
    def __init__(self, in_layer, *h_arr):
        self.Theta = dict()
        self.dE_dTheta = dict()

        self.build(in_layer, h_arr)

    def build(self, in_layer, *h_arr):
        self.layers = []
        self.in_layer = in_layer
        self.out_layer = h_arr[-1]
        for i in range(len(h_arr)):
            s = in_layer if i==0 else h_arr[i-1]
            e = h_arr[i]
            l = torch.normal(means=torch.zeros(e, s+1), std=torch.zeros(e, s+1).fill_(1/math.sqrt(e))).type(torch.DoubleTensor)
            self.layers.append(l)
        return self.layers

    def getLayer(self, layer):
        return self.layers[layer-1]

    def forward(self, input):
        output = input.view(input.size()[0], 1) if len(input.size()) == 1 else input
        for idx, layer in enumerate(self.layers):
            bias = torch.randn(1, output.size()[1]).type(torch.DoubleTensor)
            output = torch.cat((bias, output), 0)
            output = torch.sigmoid(layer.mm(output))
        return output

    def backward(self, target):
        pass

    def updateParams(self, eta):
        pass
