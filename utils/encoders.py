import abc
import numpy as np

import torch

try:
    from utils.rbf import *
except:
    print("RBF sampler will not work, probably due missing sklearn package")

from utils.normalizer import *

class IEncoder(nn.Module):
    def __init__(self, size_in, size_out, n_features):
        super().__init__()
        self.size_in = size_in
        self.size_out = size_out
        self.n_features = n_features

        self.register_parameter("dummy_param", nn.Parameter(torch.empty(0)))
    def device(self):
        return self.dummy_param.device

    @abc.abstractmethod
    def forward(self, states, memory):
        pass

    def out_size(self):
        return self.size_out

    def in_size(self):
        return self.size_in

    def features_n(self):
        return self.n_features

    def has_features(self):
        return False

    def extract_features(self, states):
        feats = torch.zeros( # states can come in group, but features is per state not group
                states.view(-1, self.size_in).size(0), self.n_features).reshape(
                        states.size(0), -1).to(states.device)
        return self.forward(states, feats)

# better to rethink design of this ~ beacuse of RNN ~ features, multiple ? dont over-engineer though...
class StackedEncoder(IEncoder):
    def __init__(self, encoder_a, encoder_b):
        super().__init__(size_in=encoder_a.in_size(), size_out=encoder_b.out_size(), n_features=encoder_b.features_n())
        self.encoder_a = encoder_a
        self.encoder_b = encoder_b
        assert not self.encoder_a.has_features() or not self.encoder_a.has_features(), "only one RNN is allowed in encoder!"
        assert not self.encoder_a.has_features(), "Currently RNN can be only *last* layer of encoder!!"

    def features_n(self):
        return self.encoder_b.features_n()

    def has_features(self):
        return self.encoder_a.has_features() or self.encoder_a.has_features()

    def forward(self, states, memory):
        states, memory = self.encoder_a(states, memory)
        return self.encoder_b(states, memory)

    def extract_features(self, states):
        states, features_a = self.encoder_a.extract_features(states)
        states, features_b = self.encoder_b.extract_features(states)
        return states, features_b if self.encoder_b.has_features() else features_a

class IdentityEncoder(IEncoder):
    def __init__(self, size_in):
        super().__init__(size_in=size_in, size_out=size_in, n_features=1)
    def forward(self, states, memory):
        return states, memory

class RBFEncoder(IEncoder):
    def __init__(self, size_in, env, gamas, components, sampler = None):
        self.encoder = RbfState(env, gamas, components, sampler)
        super().__init__(size_in=size_in, size_out=self.encoder.size, n_features=1)
    def forward(self, states, memory=None):
        device = states.device
        states = states.view(-1, self.size_in)
        states = self.encoder.transform(states).reshape(-1, self.size_out)
        if memory is None:
            return torch.from_numpy(states).float().to(device)
        return torch.from_numpy(states).float().to(memory.device), memory

class BatchNormalizer2D(IEncoder):
    def __init__(self, size_in, n_history):
        super().__init__(size_in=size_in*n_history, size_out=size_in*n_history, n_features=1)
        self.bn = nn.BatchNorm1d(self.size_in)
    def forward(self, states, memory):
        states = states.view(-1, self.size_in)
        if states.size(0) == 1: self.eval()
        out = self.bn(states)
        if states.size(0) == 1: self.train()
        return out, memory
class BatchNormalizer3D(IEncoder):
    def __init__(self, size_in, n_history):
        super().__init__(size_in=size_in, size_out=size_in*n_history, n_features=1)
        self.n_history = n_history
        self.bn = nn.BatchNorm1d(self.n_history)
    def forward(self, states, memory):
        states = states.view(-1, self.n_history, self.size_in)
        if states.size(0) == 1: self.eval()
        out = self.bn(states).view(-1, self.size_out)
        if states.size(0) == 1: self.train()
        return out, memory
    def extract_features(self, states):
        assert states.size(1) == self.size_out
        feats = torch.zeros(
                states.view(-1, self.size_out).size(0), self.n_features).to(states.device)
        return self.forward(states, feats)

class GlobalNormalizerWGrads(IEncoder):
    def __init__(self, size_in, n_history):
        super().__init__(size_in=size_in*n_history, size_out=size_in*n_history, n_features=1)
        self.norm = Normalizer(self.size_in)
        self.add_module("norm", self.norm)
    def forward(self, states, memory):
        states = states.to(self.device())
        self.norm.update(states)
        return self.norm.normalize(states).view(-1, self.size_out), memory.to(self.device())
class GlobalNormalizer(GlobalNormalizerWGrads):
    def __init__(self, size_in, n_history):
        super().__init__(size_in, n_history)
        for p in self.norm.parameters():
            p.requires_grad = False

class GoalGlobalNorm(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.cnt = 0
        self.size = size
        self.norm = Normalizer(self.size)
        self.add_module("norm", self.norm)
        for p in self.norm.parameters():
            p.requires_grad = False
        self.stop = False

        self.dummy_param = nn.Parameter(torch.empty(0))
    def device(self):
        return self.dummy_param.device

    def stop_norm(self):
        self.stop = True
    def forward(self, states):
        self.cnt += 1
        shape = states.size()
        states = states.reshape(-1, self.size).to(self.device())
        if not self.stop:
            self.norm.update(states)
        return self.norm.normalize(states).view(shape)

class GoalIdentity(nn.Module):
    def __init__(self, _):
        super().__init__()

        self.dummy_param = nn.Parameter(torch.empty(0))
    def device(self):
        return self.dummy_param.device
        
    def forward(self, states):
        return states.to(self.device())
