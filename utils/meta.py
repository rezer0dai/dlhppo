import os, random
import torch
import numpy as np

import torch.optim as optim
from torch.autograd import Variable

from utils.polyak import POLYAK

class META(POLYAK):
    def init_meta(self, lra, lrc):
        self.optim_actor = optim.SGD(self.ac_target.actor_parameters(), lr=lra, momentum=.7, nesterov=True)
        self.optim_critic = [ optim.Adam(self.ac_target.critic_parameters(i), lr=lrc) for i in range(len(self.ac_target.critic)) ]

        self.sigmas = [key for key in self.ac_explorer.state_dict().keys() if "sigma" in key]

        for p in self.ac_target.parameters():
            p.grad = Variable(torch.zeros(p.size()))

# obsolete as we no longer use this, therefore need to update just for actor learning
    def meta_update_mean(self, explorer_params, targets, tau):
        if not tau:
            return

        self._algo(explorer_params, targets, self.optim_actor)

        # we want to skip updating explorers sigma, as that is exploration factor!!
        sigmas = [ self.ac_explorer.state_dict()[sigma] for sigma in self.sigmas ]
        for actor in explorers: # copy targets one
            for target_w, explorer_w in zip(targets.parameters(), actor.parameters()):
                match = sum([torch.all(sigma==explorer_w) for sigma in sigmas if sigma.shape == explorer_w.shape])
                if match: # we left sigma specific per noise head!!
                    if match > 1: # DEBUG
                        print("\n")
                        for sigma, name in zip(sigmas, self.sigmas):
                            if sigma.shape == explorer_w.shape and torch.all(sigma==explorer_w):
                                print("META MATCH!:", match.item(), name, sigma.sum().item())
                        # for test purposes we want using reptile + foml only one actor head ...
                    #assert 1 == match
                    continue
                explorer_w.data.copy_(target_w.data)

    def meta_update(self, i, explorer_params, targets, tau):
        if not tau:
            return

        self._algo(explorer_params, targets, self.optim_critic[i])

        for target_w, explorer_w in zip(targets.parameters(), explorer_params):
            explorer_w.data.copy_(target_w.data)

    def _algo(self, explorer_params, targets, optim):
        assert False
