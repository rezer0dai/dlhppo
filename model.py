import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from utils.nes import *
from utils.policy import PPOHead

import random

from collections import OrderedDict

from config import CORE_GOAL_SIZE, CORE_STATE_SIZE, HRL_ACTION_SIZE, CORE_ORIGINAL_GOAL_SIZE
import config

def move_tensor(x):
    return x

class Actor(nn.Module): # decorator
    def __init__(self, net, action_size, f_mean_clip, f_scale_clip, ibottleneck, noise_scale=False):
        super().__init__()
        self.add_module("net", net)
        self.algo = PPOHead(action_size, f_scale_clip, noise_scale)
        self.f_mean_clip = f_mean_clip
        if ibottleneck is not None:
            self.add_module("ibottleneck", ibottleneck)
        else:
            self.ibottleneck = move_tensor

        self.register_parameter("dummy_param", nn.Parameter(torch.empty(0)))
    def device(self):
        return self.dummy_param.device

    def forward(self, goals, states):
        #assert torch.float32 == goals.dtype
        with torch.no_grad():
            goal = self.ibottleneck(goals.to(self.device()))

        state = states[:, 3:-config.CORE_ORIGINAL_GOAL_SIZE-config.TIMEFEAT] # SKIP timefeature, duplicates achieved goal + goal
        if config.BLIND:
            if config.HRL_GOAL_SIZE == goal.shape[-1]:
                state = state[:, :config.LL_STATE_SIZE] # only arm + gripper
#                if not config.ERGOJR:
#                    state = state[:, :10] # only arm + gripper
#                else:
#                    state = state[:, :config.CORE_ORIGINAL_GOAL_SIZE * 2 + config.ACTION_SIZE * 2]

        state = state.to(self.device())
        state = torch.cat([goal, state], 1)
        x = self.net(state)
        x = self.f_mean_clip(x)
        return self.algo(x)

    def sample_noise(self, _):
        self.net.sample_noise(random.randint(0, len(self.net.layers) - 1))
    def remove_noise(self):
        self.net.remove_noise()
class ActorFactory: # proxy
    def __init__(self, layers, action_size, f_mean_clip, f_scale_clip, ibottleneck, noise_scale=.3):

        ll = [(l - config.CORE_ORIGINAL_GOAL_SIZE - config.TIMEFEAT - config.CORE_ORIGINAL_GOAL_SIZE) if 0 == i else l for i, l in enumerate(layers)]
#SHAREDMININETGOAL
#        ll = [33 if 0 == i else l for i, l in enumerate(layers)]

        if config.BLIND:
            if config.ACTION_SIZE == action_size:
                ll[0] = config.LL_STATE_SIZE + config.HRL_GOAL_SIZE
#                if not config.ERGOJR:
#                    ll[0] = 10+config.HRL_GOAL_SIZE
#                else:
#                    ll[0] = config.CORE_ORIGINAL_GOAL_SIZE * 2 + config.ACTION_SIZE * 2 + config.HRL_GOAL_SIZE
#
#                ll[0] = 10+config.HRL_ACTION_SIZE
#                ll[0] = 13+config.HRL_GOAL_SIZE
#                ll[0] = 6 + config.HRL_GOAL_SIZE


        net = nn.Sequential(*[ nn.Linear(layer, ll[i+1]) for i, layer in enumerate(ll[:-1]) ])
        self.actor = lambda : Actor(net, action_size, f_mean_clip, f_scale_clip, ibottleneck, noise_scale)
#        self.factory = NoisyNetFactory(ll)
#        self.actor = lambda head: Actor(head, action_size, f_mean_clip, f_scale_clip, ibottleneck, noise_scale)
    def head(self):
        return self.actor()
        return self.actor(self.factory.head())

import numpy as np

def rl_ibounds(layer):
    b = 1. / np.sqrt(layer.weight.data.size(0))
    return (-b, +b)

def initialize_weights(layer):
    if type(layer) not in [nn.Linear, ]:
        return
    nn.init.uniform_(layer.weight.data, *rl_ibounds(layer))

class Mish(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class Critic(nn.Module):
    def __init__(self, n_actors, n_rewards, state_size, action_size, layers, ibottleneck):
        super().__init__()

        self.add_module("ibottleneck", ibottleneck)

#        state_size = config.CORE_STATE_SIZE + config.CORE_ORIGINAL_GOAL_SIZE*(not config.NO_GOAL) - 3#achieved goal duplicate
        state_size = config.CORE_STATE_SIZE - config.CORE_ORIGINAL_GOAL_SIZE + config.HRL_GOAL_SIZE#achieved goal duplicate

#SHAREDMININETGOAL
#        state_size += config.HRL_GOAL_SIZE - 2 - 3

        action_size = config.ACTION_SIZE# + config.HRL_ACTION_SIZE * config.NO_GOAL

        layers = [l for layer in [state_size + action_size] + layers + [n_rewards]  for l in [layer, 0] ]
        self.net = nn.Sequential(
                OrderedDict([
                    *[("layer_%i"%i, (nn.Linear(layer, layers[i+2], bias=(i!=0) and (i!=len(layers)-4)) if layer else (nn.Tanh() if i != 1 else nn.ReLU()))) for i, layer in enumerate(layers[:-3])]

#                    *[("layer_%i"%i, (nn.Linear(layer, layers[i+2], bias=(i!=0) and (i!=len(layers)-4)) if layer else nn.Tanh())) for i, layer in enumerate(layers[:-3])]
#                    *[("layer_%i"%i, (nn.Linear(layer, layers[i+2], bias=(i!=0) and (i!=len(layers)-4)) if layer else Mish())) for i, layer in enumerate(layers[:-3])]

                    ])
                )
        self.apply(initialize_weights)
        self.net[-1].weight.data.uniform_(-3e-3, 3e-3) # seems this works better ? TODO : proper tests!!

        self.register_parameter("dummy_param", nn.Parameter(torch.empty(0)))
    def device(self):
        return self.dummy_param.device

    def forward(self, goals, states, actions):
        #assert torch.float32 == goals.dtype
        goals = self.ibottleneck(goals.to(self.device()))
        states = states[:, 3:-config.CORE_ORIGINAL_GOAL_SIZE]#config.TIMEFEAT] # SKIP GOAL and timefeature from value, leaving timefeature only for LL Actor

        if not config.NO_GOAL:
            states = torch.cat([goals, states], 1)
        else:
            assert False

        x = torch.cat((states.to(self.device()), actions.to(self.device())), dim=1)
        return self.net(x)
