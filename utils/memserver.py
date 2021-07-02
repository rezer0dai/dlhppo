import torch
import numpy as np

import time, threading, collections

from torch.multiprocessing import Queue, Process
#from timebudget import timebudget

from utils.fastmem import FastMemory

class MemoryServer(Process):#threading.Thread):#
    def __init__(self, descs, memory, credit_assign, brain, n_step, good_reach):
        super().__init__()

        # must be shared!! double call should be ok ?!
        brain.share_memory()

        self.memory = memory

        self.descs = descs

        self.credit = credit_assign
        self.brain = brain
        self.n_step = n_step
        self.good_reach = good_reach

        self.channel = Queue()
        self.sampler = [ Queue() for _ in descs ]

        self.sleep = .1
        self.lock = threading.Lock()

    def run(self):
        self.thread_pool = []
        self.storage = collections.deque(maxlen=100)
        self.cmd = { "push" : self._push, "switch" : self._switch }

        worker = threading.Thread(target=self._worker)
        worker.start()

        while True: # single thread is fine
            data = self.channel.get()
            cmd, data = data
            self.cmd[cmd](*data)

        worker.join()

    def _switch(self, sleep):
        self.sleep = sleep

    def _push(self, ep, chunks, e_i, goods):
        max_allowed = len(ep) - self.n_step - 1
        allowed_mask = [ bool(sum(goods[i:i+self.good_reach, e_i])) for i in range(max_allowed)
                ] + [False] * (len(ep) - max_allowed)

        _, episode = self.credit( # it is double
                    *[ep[:, sum(chunks[:i+1]):sum(chunks[:i+2])] for i in range(len(chunks[:-1]))],
                    brain=self.brain,
                    recalc=True)
        idx = np.arange(len(episode))[allowed_mask]

        with self.lock:
            self.storage.append(episode[idx].share_memory_())
            assert self.storage[-1].is_shared()

            for sampler in self.sampler:
                sampler.put(self.storage[-1])

            self.memory.push(episode, allowed_mask)
        return episode[idx]

    def _worker(self):
        while not len(self.memory):
            time.sleep(.1)

        def update(ind): # curry curry
            def _update(recalc, indices, allowed_mask, episode):
                return self._update(ind, recalc, indices, allowed_mask, episode)
            return _update

        while True:
            while not self.sleep:
                time.sleep(.1)
            time.sleep(self.sleep)
            with self.lock: # lock only one sample, should be fast enough
                for i, desc in enumerate(self.descs):
                    episode_batch = self.memory.sample(update(i), desc.batch_size, desc.memory_size)

    def _update(self, ind, recalc, indices, allowed_mask, episode):
        if recalc:
            return self._push_to_fast(ind, recalc, indices, allowed_mask, episode)

        # though i dont like many threads with python environment...
        self.thread_pool.append(
                threading.Thread(target=self._push_to_fast, args=(# do async as most expensive work
                    ind, recalc, indices, allowed_mask, episode, )))

        self.thread_pool[-1].start()
        self._clean_threads()
        return torch.ones(len(allowed_mask), sum(self.memory.chunks))

    def _push_to_fast(self, ind, recalc, indices, allowed_mask, episode):
        goals, states, memory, actions, probs, rewards, _, _, _, _, _ = episode

        _, episode = self.credit(goals, states, memory, actions, probs, rewards,
                self.brain, recalc=recalc, indices=indices)

        idx = np.arange(len(episode))[allowed_mask]

        self.storage.append(episode[idx].share_memory_())
        assert self.storage[-1].is_shared()
        assert not episode[idx].is_shared()
        assert not episode[idx[0]].is_shared()
        self.sampler[ind].put(self.storage[-1])

        return episode

    def _clean_threads(self):
        if len(self.thread_pool) < 20:
            return
        self.thread_pool[0].join()
        del self.thread_pool[0]

#from timebudget import timebudget

class MemoryBoost:
    def __init__(self, descs, memory, credit_assign, brain, n_step, good_reach):
        self.fast_m = [ FastMemory(
            desc, memory.chunks, memory.device) for desc in descs ]

        self.server = [
                MemoryServer(descs, memory, credit_assign, brain, n_step, good_reach
                    ) for _ in range(7) ]
        for server in self.server:
            server.start()

        memory.device = 'cpu'

        self.storage = collections.deque(maxlen=100)
        self.total = 0

    def __len__(self):
        assert False

    def push(self, ep, chunks, e_i, goods):
        self.total = 0
        self.storage.append(ep.share_memory_())
        #  assert self.storage[-1].is_shared()
        for server in self.server:
            server.channel.put(("push", (self.storage[-1], chunks, e_i, goods)))

    #@timebudget
    def step(self, ind, desc):
        if not len(self.fast_m[0]) and self.server[0].sampler[0].empty():
            return

        for server in self.server:
            server.channel.put(("switch", (.01,)))

        for server in self.server:
            min_draw = 2 # should be brain description property
            while min_draw > 0 or not server.sampler[ind].empty():
                episode = server.sampler[ind].get()
                self.fast_m[ind].push(episode.clone())
                del episode
                min_draw -= 1

        for server in self.server:
            server.channel.put(("switch", (.07,)))

    def sample(self, ind, _desc):
        for server in self.server:
            server.channel.put(("switch", (0.,)))
        return self.fast_m[ind].sample()

    #@timebudget
    def shuffle(self):
        for fast_m in self.fast_m:
            fast_m.shuffle()
