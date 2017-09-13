import torch
import math
import numpy as np

class NeuralNetwork(object):
    def __init__(self):
        self.eta = 0.5
        self.Theta = []
        self.dE_dTheta = []
        self.a = []
        self.a_hat = []
        self.z = []
        self.delta = []

    def build(self, in_layer, *h_arr):
        for i in range(len(h_arr)):
            s = in_layer if i==0 else h_arr[i-1]
            e = h_arr[i]
            th = torch.normal(means=torch.zeros(e, s+1), std=torch.zeros(e, s+1).fill_(1/math.sqrt(e))).type(torch.DoubleTensor)
            self.Theta.append(th)

    def getLayer(self, layer):
        return self.Theta[layer-1]

    def forward(self, input):
        x = input.view(input.size()[0], 1) if len(input.size()) == 1 else input

        self.a.append(x.clone()) # a(1) = x
        bias = torch.randn(1, x.size()[1]).type(torch.DoubleTensor)
        self.a_hat.append(torch.cat((bias, x), 0)) # a_hat(1) = [bias, a(1)]

        for idx, th in enumerate(self.Theta):
            z = th.mm(self.a_hat[idx]) # z(l+1)  = theta(l) * a_hat(l)
            self.a.append(torch.sigmoid(z)) # a(l+1) = activate(z(l+1))
            bias = torch.randn(1, self.a[idx].size()[1]).type(torch.DoubleTensor)
            self.a_hat.append(torch.cat((bias, self.a[idx+1]), 0)) # a_hat(l) = [bias, a(l)]

        return self.a[-1]

    def backward(self, target):
        target = target.view(target.size()[0], 1) if len(target.size()) == 1 else target
        bias = torch.zeros(1, target.size()[1]).type(torch.DoubleTensor)
        target = torch.cat((bias, target), 0)

        for i in reversed(range(len(self.a))):
            # print "======== Round", i, "============="
            if i == len(self.a_hat)-1:
                # delta(L) = (a_hat(L)-y) * (a_hat(L) * (1-a_hat(L)))
                self.delta.insert(0, (self.a_hat[i]-target) * (self.a_hat[i]*(1-self.a_hat[i])))
                # print "delta[", i+1, "]", self.delta[0]
            else:
                th = self.Theta[i]
                last_delta = self.delta[0][1:]
                # print "th", th
                # print "a_hat[i]", self.a_hat[i]
                # print "last_delta", last_delta

                # delta(l) = (theta(l)^T x delta(l+1)) * (a_hat(l) * (1-a_hat(l)))
                delta = (th.transpose(0,1).mm(last_delta)) * (self.a_hat[i] * (1-self.a_hat[i]))
                # print "delta", delta

                # dE_dTheta(l) = a(l) x (delta(l+1)^T)
                dE_dTheta = self.a_hat[i].mm(last_delta.transpose(0, 1)).transpose(0, 1)
                # print "dE_dTheta", dE_dTheta

                self.delta.insert(0, delta)
                self.dE_dTheta.insert(0, dE_dTheta)
                # print "delta[", i+1, "]", self.delta[0]
                # print "dE_dTheta[", i+1, "]", self.dE_dTheta[0]

    def updateParams(self, eta):
        self.eta = eta
        pass
