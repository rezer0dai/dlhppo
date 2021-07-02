import torch
torch.set_default_dtype(torch.float32)
torch.autograd.set_detect_anomaly(True)

import numpy as np
import random, timebudget

import config

from dlppoh import *

from lightning import get_ready, DLPPOHLightning

import pytorch_lightning as pl

if '__main__' == __name__:
    fm = get_ready()
    algo = DLPPOHLightning(fm)

    trainer = pl.Trainer(
        gpus=0,
        distributed_backend='ddp_cpu',
        num_processes=2,
        max_epochs = 1000 * 100,
    )

    trainer.fit(algo)

    print("we are done here maybe")
