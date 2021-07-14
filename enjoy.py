import torch
torch.set_default_dtype(torch.float32)
torch.autograd.set_detect_anomaly(True)

import timebudget
timebudget.set_quiet()

from lightning import get_ready, DLPPOHLightning

import pytorch_lightning as pl

import config

if '__main__' == __name__:
    from tasks.oaiproc import GymGroup, GymRender
    gym = GymRender(config.ENV_NAME, config.TOTAL_ENV)

    fm = get_ready(config.PREFIX, False)
    algo = DLPPOHLightning.load_from_checkpoint(config.ENV_NAME+".ckt", model=fm, prefix=config.PREFIX, n_env=1)

    algo.delay_load(1)

    for seed in range(1):
        z = 0
        s, g, r, d = gym.reset([seed])
        s = torch.cat([s, g], 1)
        import time
        for _ in range(1000):
            d,_,_ = algo.env.agent.exploit(g, s, torch.zeros(0), 0)
#            g_g = torch.randn(d.sample().shape)
            g_g = d.sample()
            for _ in range(10):
                d,m,_ = algo.task.ENV.ll_env.agent.exploit(g_g, s, torch.zeros(0), 0)
                z += 1
                a = d.sample()
                s, g, r, d = gym.step(a)
                s = torch.cat([s, g], 1)
                if 0 == r:
                    print("GOAL REACHED", z)
#                    time.sleep(10)
                    break
    print("EXPERIMENT DONE")

