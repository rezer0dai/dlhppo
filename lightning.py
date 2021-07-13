import os, time

import torch 
from torch import nn
from torch import optim

from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset

class RLDummyDataSet(IterableDataset):
    def __iter__(self):
        yield []
    def __len__(self):
        return 1024 * 8
            
import pytorch_lightning as pl

from collections import OrderedDict, deque, namedtuple
from typing import Tuple, List
   
class FullModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.idx = {}
        self.mods = []
        self.done = False
    def register(self, name, module):
        if self.done:
            return
        assert not(name in self.idx), name
        self.idx[name] = len(self.mods)
        self.mods.append(module)
        print("appended module", name, self.idx[name])
    def ready(self):
        self.mods = nn.ModuleList(self.mods)
        self.share_memory()
        self.done = True
    def __getitem__(self, name):
        assert name in self.idx, name
        return self.mods[self.idx[name]]
    
    def _params(self, cond):
        for name in self.idx:
            if not cond(name):
                continue
            for p in self.mods[self.idx[name]].parameters():
                yield p
                
    def enc_parameters(self):
        return self._params(lambda name: "encoder" in name)

    def explorer_parameters(self):
        return self._params(lambda name: "explorer" in name)
    
    def critic_explorer_parameters(self, i, pi):
        return self._params(lambda name: "explorer" in name and "critic" in name and pi in name and "_%i_"%i in name)
    
    def critic_target_parameters(self, i, pi):
        return self._params(lambda name: "target" in name and "critic" in name and pi in name and "_%i_"%i in name)
    
    def actor_explorer_parameters(self, i, pi):
        return self._params(lambda name: "explorer" in name and "actor" in name and pi in name and "_%i_"%i in name)
    
    def actor_target_parameters(self, i, pi):
        return self._params(lambda name: "target" in name and "actor" in name and pi in name and "_%i_"%i in name)
         
from torch.multiprocessing import Queue, Process
from head import install_highlevel
from dlppoh import HighLevelCtrlTask

import numpy as np

class DLPPOHLightning(pl.LightningModule):
    def __init__(self, model, prefix, n_env):
        super().__init__()
        model.ready()
        self.add_module("model", model)
        self.count = 0
        self.finished = False
        self.ready = False
        self.n_env = n_env
        self.prefix = prefix
        
    def delay_load(self, rank):
        if self.ready:
            return

        self.env_start = time.time()

        KEYID = self.prefix+"_hl"

        high_level_task = HighLevelCtrlTask("dlppoh_dock_%i"%rank, KEYID, self.model)

        #self.print("dlppoh_dock_%i"%rank)

        self.env, self.task = install_highlevel(high_level_task, KEYID, self.model)
        self.ready = True
        self.playground = self.env.step(self.task, self.count * 100 + np.arange(self.n_env), 1)
        self.env.debug_stats = False#True
        
    def _training_step(self, rank):
        
        self.delay_load(rank)

#        print("\n ---> ", torch.cat([ep.view(-1) for ep in self.model.enc_parameters()]).sum(), os.getpid())

        steping = False
        while True:

            loss = self.task.ENV.ll_env.agent.step(True, steping)
            if loss is not None:
                break

            loss = self.env.agent.step(None, steping)
            if loss is not None:
                break

            steping = True

            data, acu_reward = next(self.playground, (None, None))
            if data is not None:
                self.reward = data[6]
                continue

            self.count += 1
            if 0 == self.count % 4:
                test = next(self.env.evaluate(self.task, None))
                self.finished = test[0]
                msg = "\n\n  <{} min> #{} > ENV TEST : {}".format("%.2f"%((time.time()-self.env_start) / 60), self.count, test)
                self.print(msg)

            self.print("\n[ <{}min> new ep -> #{} last_reward = {} ]".format(
              "%.2f"%((time.time()-self.env_start) / 60), self.count, self.reward.mean()))
            self.playground = self.env.step(self.task, self.count * 100 + np.arange(self.n_env), 1)
        return loss
        
    def on_train_batch_start(self, _batch, _batch_idx, _dataloader_idx):
        if self.finished:
            return -1
        return None

        #if self.model.loss.requires_grad:
    def training_step(self, batch, nb_batch, optimizer_idx=-1) -> OrderedDict:
        loss = None
        while loss is None:
            loss = self._training_step(os.getpid())
        return OrderedDict({'loss': loss, 'log': "testlog%i"%nb_batch, 'progress_bar': "pb%i"%nb_batch})
            
    def configure_optimizers(self) -> List[Optimizer]:
        return [optim.Adam(self.model.explorer_parameters(), lr=3e-4)]
        return [
                optim.Adam(self.model.actor_explorer_parameters(0, "lowpi"), lr=1e-3),
                optim.Adam(self.model.actor_explorer_parameters(0, "highpi"), lr=3e-4),
                optim.Adam([p for p in self.model.critic_explorer_parameters(i, "lowpi") for i in range(2)], lr=1e-3),
                ]
    
    def train_dataloader(self) -> DataLoader:
        dataset = RLDummyDataSet()
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=1,
            sampler=None,
        )
        return dataloader

def get_ready(prefix, do_sampling=True):
    from head import install_highlevel

    fm = FullModel()

    KEYID = prefix+"_hl"
    high_level_task = HighLevelCtrlTask("dlppoh_dock", KEYID, fm, do_sampling=do_sampling)

    env, task = install_highlevel(high_level_task, KEYID, fm, do_sampling=do_sampling)
    fm.ready()
    return fm
