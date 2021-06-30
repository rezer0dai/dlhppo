import os
import torch
import numpy as np

import torch.optim as optim
from torch.autograd import Variable

from utils.meta import META

class REPTILE(META):
    def init_meta(self, lra, lrc):
        super().init_meta(lra/1e-1, lrc/1e-1)

    def _algo(self, explorer_params, targets, optim):
        optim.zero_grad() # scatter previous optimizer leftovers

        for target_w, explorer_w in zip(targets.parameters(), explorer_params):
            assert not target_w.grad.sum()
#            if explorer_w.grad is None:
#                continue
            target_w.grad.data = target_w.data - explorer_w.data

        optim.step() # backprop trigger
