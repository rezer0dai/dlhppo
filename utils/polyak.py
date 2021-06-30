import os
import torch
import numpy as np

import torch.optim as optim
from torch.autograd import Variable

from utils.syncnet import SyncNetwork

class POLYAK(SyncNetwork):
    def init_meta(self, lra, lrc):
        pass

    def polyak_update(self, explorer_params, targets, tau):
        if not tau:
            return

        for target_w, explorer_w in zip(targets.parameters(), explorer_params):
            target_w.data.copy_(
                target_w.data * (1. - tau) + explorer_w.data * tau)

    def meta_update(self, _, explorer_params, targets, tau):
        return self.polyak_update(explorer_params, targets, tau)
