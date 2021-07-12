import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

import numpy as np
import random


def orthogonal_init(tensor, gain=1):
    '''
    Fills the input `Tensor` using the orthogonal initialization scheme from OpenAI
    Args:
        tensor: an n-dimensional `torch.Tensor`, where :math:`n \geq 2`
        gain: optional scaling factor
    Examples:
        >>> w = torch.empty(3, 5)
        >>> orthogonal_init(w)
    '''
    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    rows = tensor.size(0)
    cols = tensor[0].numel()
    flattened = tensor.new(rows, cols).normal_(0, 1)

    if rows < cols:
        flattened.t_()

    # Compute the qr factorization
    u, s, v = torch.svd(flattened, some=True)
    if rows < cols:
        u.t_()
    q = u if tuple(u.shape) == (rows, cols) else v
    with torch.no_grad():
        tensor.view_as(q).copy_(q)
        tensor.mul_(gain)
    return tensor

def nes_ibounds(layer):
    b = np.sqrt(3. / layer.weight.data.size(1))
    return (-b, +b)

def initialize_weights(layer):
    if type(layer) not in [nn.Linear, ]:
        return
        
    orthogonal_init(layer.weight.data, gain=2**.5)
    #nn.init.uniform_(layer.weight.data, *nes_ibounds(layer))

import numpy as np
class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, last):
        super().__init__(in_features, out_features, bias=False)

        self.register_buffer("noise", torch.zeros(out_features, in_features))

        if not last: self.apply(initialize_weights)
        else: orthogonal_init(self.weight.data, 1.)

    def forward(self, sigma, data):
        return F.linear(data, self.weight + sigma * Variable(self.noise).to(sigma.device))

    def sample_noise(self):
        torch.randn(self.noise.size(), out=self.noise)

    def remove_noise(self):
        assert not self.noise.sum()
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
        self.count += 1
        if self.noise_interval:
            return
        if self.count % self.noise_interval:
            return
        l = random.randint(0, len(self.layers) - 1)
        self.layers[l].sample_noise()

    def remove_noise(self):
        assert False
        for layer in self.layers:
            layer.remove_noise()

    def forward(self, data):
        self.sample_noise()

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
        self.net = [ NoisyLinear(layer, layers[i+1], len(layers)-2== i ) for i, layer in enumerate(layers[:-1]) ]
        self.relu = relu
        self.noise_interval = noise_interval

    def head(self):
        return NoisyNet(self.net, self.noise_interval, self.relu)
