import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

import numpy as np
import random

def nes_ibounds(layer):
    b = np.sqrt(3. / layer.weight.data.size(1))
    return (-b, +b)

def initialize_weights(layer):
    if type(layer) not in [nn.Linear, ]:
        return
    nn.init.uniform_(layer.weight.data, *nes_ibounds(layer))

import numpy as np
class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=False)

        self.register_buffer("noise", torch.zeros(out_features, in_features))

        self.apply(initialize_weights)

    def forward(self, sigma, data):
        return F.linear(data, self.weight + sigma * Variable(self.noise).to(sigma.device))

    def sample_noise(self):
        #assert False
        torch.randn(self.noise.size(), out=self.noise)

    def remove_noise(self):
        torch.zeros(self.noise.size(), out=self.noise)

class NoisyNet(nn.Module):
    def __init__(self, layers, noise_interval, relu):
        super().__init__()
        self.count = 0
        self.noise_interval = noise_interval

        # TODO : properly expose interface to remove/select noise in selective way
        self.layers = layers

        self.relu = relu

        sigma = []
        for i, layer in enumerate(self.layers):
            sigma.append( 
                nn.Parameter(torch.Tensor(layer.out_features, layer.in_features).fill_(.017))
            )
#            self.register_parameter("sigma_%i"%(i), sigma)

            #assert len(list(layer.parameters())) == 1, "unexpected layer in NoisyNet {}".format(layer)


#        for i, layer in enumerate(self.layers):
            for j, p in enumerate(layer.parameters()):
                self.register_parameter("neslayer_%i_%i"%(i, j), p)

        self.sigma = nn.ParameterList(sigma)

    def sample_noise(self):
        #return
        self.count += 1
        if not self.noise_interval:
            return
        if self.count % self.noise_interval:
            return
        l = random.randint(0, len(self.layers) - 1)
        self.layers[l].sample_noise()

#        for l in self.layers:
#            l.sample_noise()

    def remove_noise(self):
        for layer in self.layers:
            layer.remove_noise()

    def forward(self, data):
        #print("\n ..... ", self.sigma_0.device)
        for i, layer in enumerate(self.layers[:-1]):
            if self.relu:
                data = F.selu(layer(self.sigma[i], data))
#                x = layer(self.sigma[i], data)
#                data = x * torch.tanh(F.softplus(x))
            else:
                data = torch.tanh(layer(self.sigma[i], data))
        return self.layers[-1](self.sigma[-1], data)

class NoisyNetFactory:
    def __init__(self, layers, noise_interval, relu):
        self.net = [ NoisyLinear(layer, layers[i+1]) for i, layer in enumerate(layers[:-1]) ]
        self.relu = relu
        self.noise_interval = noise_interval

    def head(self):
        return NoisyNet(self.net, self.noise_interval, self.relu)
