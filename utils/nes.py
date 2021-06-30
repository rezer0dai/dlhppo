import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

import numpy as np

import config # TODO KICK OUT

def nes_ibounds(layer):
    b = np.sqrt(3. / layer.weight.data.size(1))
    return (-b, +b)

def initialize_weights(layer):
    if type(layer) not in [nn.Linear, ]:
        return
    nn.init.uniform_(layer.weight.data, *nes_ibounds(layer))

class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=False)

        self.register_buffer('noise', torch.zeros(out_features, in_features))

        self.apply(initialize_weights)

    def forward(self, sigma, data):
        #print("\n wow", sigma.device, data.device, self.noise.device)
        return F.linear(data, self.weight + sigma * Variable(self.noise).to(sigma.device))

    def sample_noise(self):
        torch.randn(self.noise.size(), out=self.noise)

    def remove_noise(self):
        #  assert not self.noise.sum()
        torch.zeros(self.noise.size(), out=self.noise)

class NoisyNet(nn.Module):
    def __init__(self, layers):
        super().__init__()
        # TODO : properly expose interface to remove/select noise in selective way
        self.layers = layers

        self.relu = config.RELU#max([l.out_features for l in layers]) == 64
        print("RELU", self.relu, layers)

        self.sigma = []
        for i, layer in enumerate(self.layers):
            self.sigma.append(
                nn.Parameter(torch.Tensor(layer.out_features, layer.in_features).fill_(.017)))
            self.register_parameter("sigma_%i"%(i), self.sigma[-1])

            assert len(list(layer.parameters())) == 1, "unexpected layer in NoisyNet {}".format(layer)

            for j, p in enumerate(layer.parameters()):
                self.register_parameter("neslayer_%i_%i"%(i, j), p)

    def sample_noise(self, i):
        self.layers[i % len(self.layers)].sample_noise()

    def remove_noise(self):
        for layer in self.layers:
            layer.remove_noise()

    def forward(self, data):
        for i, layer in enumerate(self.layers[:-1]):
            if self.relu:
                data = F.selu(layer(self.sigma[i], data))
#                x = layer(self.sigma[i], data)
#                data = x * torch.tanh(F.softplus(x))
            else:
                data = torch.tanh(layer(self.sigma[i], data))
        return self.layers[-1](self.sigma[-1], data)

class NoisyNetFactory:
    def __init__(self, layers):
        self.net = [ NoisyLinear(layer, layers[i+1]) for i, layer in enumerate(layers[:-1]) ]

    def head(self):
        return NoisyNet(self.net)
