import os
import torch
import numpy as np

import torch.optim as optim
from torch.autograd import Variable

from utils.meta import META

# we build here on premmise, that batch contains multi-task experience!!
# therefore we will take few grad steps in 'inner-loop' and then one FOMAML step in here
class FOML(META):
    def init_meta(self, lra, lrc):
        super().init_meta(lra/2, lrc/2)
        self.c = 0

    def _algo(self, explorer_params, targets, optim):
        for target_w, explorer_w in zip(targets.parameters(), explorer_params):
            if explorer_w.grad is None:
                continue
            #  assert explorer_w.grad.sum()
            target_w.grad.data.add_(explorer_w.grad.data)

        self.c += 1
        if self.c % 5:
            return # acumulate few different gradients over gradients

        optim.step() # backprop trigger
        optim.zero_grad() # learn leftovers
        for target_w in targets.parameters():
            assert not target_w.grad.sum()
