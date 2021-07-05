import torch
torch.set_default_dtype(torch.float32)
torch.autograd.set_detect_anomaly(True)

import timebudget
timebudget.set_quiet()

from lightning import get_ready, DLPPOHLightning

import pytorch_lightning as pl

import config

if '__main__' == __name__:
    fm = get_ready(config.PREFIX)
    algo = DLPPOHLightning(fm, config.PREFIX, config.MIN_N_SIM)

    trainer = pl.Trainer(
        gpus=0,
        distributed_backend="ddp_cpu" if 40 // config.MIN_N_SIM > 1 else "dp",
        num_processes=40 // config.MIN_N_SIM,
        max_epochs = 1000 * 100,
    )

    trainer.fit(algo)

    print("we are done here maybe")
