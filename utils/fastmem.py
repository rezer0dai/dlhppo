import torch

import random
from timebudget import timebudget

import sys

class FastMemory:
    def __init__(self, desc, chunks, device):
        self.xmemory = []
        self._memory = []
        self.desc = desc
        self.capacity = self.desc.optim_pool_size
        self.chunks = chunks

        assert self.capacity
        self.ind = -1
        self.sentinel = 0
        self.memory = torch.zeros(self.capacity, sum(chunks), device=device)

#        self.chunkies = [ torch.zeros(self.capacity, sum(self.chunks[:i+2])-sum(self.chunks[:i+1])) for i in range(len(self.chunks)-1) ]

    def push(self, episode, _):
        assert episode[0].shape[0] == sum(self.chunks), "--> {} {}".format(episode.shape, self.chunks)
        self._memory.extend(episode)

#        self._clean()
#        self.xmemory.extend(episode)

    def shuffle(self):
        if not len(self._memory):
            return

#        if self.desc.batch_size > 2000 : print("PER DREAM >>", len(self._memory), "<<")
#        else: print("XXX-PER DREAM >>", len(self._memory), "<<")
        self.rebuff = 0

        indices = random.sample(range(len(self._memory)), len(self._memory))
        for i in indices:
            self.ind = (self.ind + 1) % self.capacity
            self.memory[self.ind] = self._memory[i]
            if self.sentinel < len(self.memory) - 1:
                self.sentinel = self.ind

#            for c in range(len(self.chunks)-1):
#                self.chunkies[c][self.ind] = self._memory[i][sum(self.chunks[:c+1]):sum(self.chunks[:c+2])]

        del self._memory
        self._memory = []

    def sample(self, beg=-1, epoch=0):
        if 1 == epoch:
            return ( None, None, None )

        self.shuffle()

#        with timebudget("[1]FastMemory-sample"):
#            with timebudget("[1]FastMemory-sample::STACKING"):
#                samples = torch.stack(random.sample(
#                    self.xmemory, min(len(self.xmemory)-1, self.desc.optim_batch_size))
#                        ).to(self.device)#.contiguous()
#
#            for _ in range(self.desc.optim_epochs):
#                idx = random.sample(range(len(samples)), min(len(samples)-1, self.desc.batch_size))
#                with timebudget("[1]FastMemory-sample::INDEXING"):
#                    return (None, [
#                            samples[idx, sum(self.chunks[:i+1]):sum(self.chunks[:i+2])
#                        ] for i in range(len(self.chunks)-1) ],
#                        [-1, epoch - 1 if 0 != epoch else self.desc.optim_epochs ])

        with timebudget("FastMemory-sample"):
            if -1 == beg:
                beg = random.randint(0, self.sentinel - 1 - min(self.sentinel-1-1, self.desc.optim_batch_size))

            space = min(self.desc.optim_batch_size, self.sentinel-1 - beg)
            delta = random.randint(0, space - min(space-1, self.desc.batch_size))

            start = beg + delta
            end = start + min(self.desc.batch_size, space - 2)

            with timebudget("FastMemory-sample::INDEXING"):
                return (None, [
                    self.memory[start:end, sum(self.chunks[:i+1]):sum(self.chunks[:i+2])
                    ] for i in range(len(self.chunks)-1) ], 
                    [beg, epoch - 1 if 0 != epoch else self.desc.optim_epochs ])

#            with timebudget("FastMemory-sample::INDEXING[CHUNKIES]"):
#                return (None, [start, end, self],
#                [beg, epoch - 1 if 0 != epoch else self.desc.optim_epochs ])

#                    yield (None, [
#                        self.chunkies[i][beg:end
#                            ] for i in range(len(self.chunks)-1) ])
    def __len__(self):
        return max(self.sentinel, len(self._memory))

#    def _clean(self):
#        if len(self.xmemory) < self.capacity:
#            return
#
#        n_clean = len(self.xmemory) // 4
#        del self.xmemory[:n_clean]

