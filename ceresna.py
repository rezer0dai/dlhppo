import torch
torch.set_default_dtype(torch.float32)
torch.autograd.set_detect_anomaly(True)

import timebudget
timebudget.set_quiet()

from lightning import get_ready, DLPPOHLightning

import pytorch_lightning as pl

import config

if '__main__' == __name__:
    fm, env, task = get_ready(config.PREFIX)
    algo = DLPPOHLightning(fm, config.PREFIX, config.MIN_N_SIM, env, task)

#    trainer = pl.Trainer(
#        gpus=0,
#        distributed_backend="ddp_cpu" if config.MIN_N_SIM // 40 > 1 else "dp",
#        num_processes=config.MIN_N_SIM // 40,
#        max_epochs = 1000 * 100,
#    )
#
#    trainer.fit(algo)

    algo.delay_load(0)
    print("\n", "*"*30, "\n")
    opts = algo.configure_optimizers()
    i = 0
    while -1 != algo.on_train_batch_start(None, None, None):
        i += 1
        info = algo.training_step(None, i, -1)

        loss = info["loss"]
        algo.model.zero_grad()
        loss.backward()
        opts[info["oid"]].step()
        opts[info["oid"] + 2].step()

    trainer.save_checkpoint(config.ENV_NAME+".ckt")

    print("we are done here maybe")
