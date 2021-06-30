import torch
import numpy as np
import random
import itertools

# we want to store tensors as CPU only i think ..
class Memory:
    def __init__(self, capacity, chunks, ep_draw, device='cpu'):
        self.memory = []

        self.ep_draw = ep_draw
        self.chunks = [0] + chunks
        self.back_pointers = []
        self.capacity = capacity
        self.device = device

    def push(self, experience_batch):
        self._clip()

        experience_batch = torch.stack(experience_batch).transpose(0, 1).contiguous()
        self._sync(experience_batch)

        experience_batch = experience_batch.view(-1, sum(self.chunks)).cpu()
        self.memory = torch.cat([self.memory, experience_batch]) if len(self.memory) else experience_batch

        assert len(self.memory) == len(self.back_pointers)

    def sample(self, update, batch_size, back_view=0):
        idx, eps = self._select(batch_size, back_view)

        samples = []
        if update is not None: # now we getting hairy computationaly
            for i, e in enumerate(eps):
                experience = self._decompose(self.memory[range(*e)])
                do_update, experience = update(experience)
                experience = torch.cat(experience, 1)

                if do_update: self.memory[range(*e)] = experience

                for j in idx[i]:
                    assert j in range(*e)
                    samples.append(experience[j - e[0]])

            samples = torch.stack(samples)
        else:
            samples = self.memory[list(itertools.chain.from_iterable(idx))]

        return self._decompose(samples.to(self.device))

    def __len__(self):
        if not len(self.back_pointers):
            return 0
        return self.back_pointers[-1][1]

    def _decompose(self, samples):
        return [ # indexing inside len(idx) sized batch is cheaper than query chunk for len(idx) samples
                samples[:, sum(self.chunks[:i+1]):sum(self.chunks[:i+2])
            ] for i in range(len(self.chunks)-1) ]

    def _select(self, batch_size, back_view):
        assert len(self)
        idx = []
        bck = []
        size = 0
        start = 0 if not back_view else (len(self) - back_view)
        while size < batch_size:
            i = random.randint(start, len(self) - 1)
            to_draw = random.randint(1, self.ep_draw)
            ep_idx = range(*self.back_pointers[i])
            idx.append(random.sample(ep_idx, to_draw))
            size += len(idx[-1])
            bck.append(self.back_pointers[i])
        return idx, bck

    def _sync(self, experience_batch):
        cum_size = len(self)
        for e in experience_batch:
            size = len(e)
            self_ptr = torch.tensor([[cum_size, cum_size + size]] * size)
            self.back_pointers = torch.cat([self.back_pointers, self_ptr]) if len(self.back_pointers) else self_ptr
            cum_size += size
        assert self.back_pointers[-1][1] == len(self.back_pointers)

    def _clip(self):
        if len(self) < self.capacity:
            return
        index = self.memory.shape[0] // 3
        new_bottom = self.back_pointers[index][1]
        self.memory = self.memory[new_bottom:]
        self.back_pointers = self.back_pointers[new_bottom:]
        self.back_pointers -= new_bottom
        assert self.back_pointers[-1][1] == len(self.back_pointers)
        assert self.back_pointers[0][0] == 0
class Memory2:
    def __init__(self, capacity, chunks, ep_draw):
        self.memory = [[] for _ in range(len(chunks))]

        self.ep_draw = ep_draw
        self.chunks = [0] + chunks#[ b[0].numel() for b in experience_batch ]
        self.back_pointers = []
        self.capacity = capacity

    def push(self, experience_batch):
        self._clip()

        for i, e_b in enumerate(experience_batch):
            e_b = torch.stack(e_b).transpose(0, 1)
            if not len(self.back_pointers) or self.back_pointers[-1][1] < len(self.memory[0]): self._sync(e_b)
            e_b = e_b.contiguous().view(-1, len(e_b[0][0]))
            self.memory[i] = torch.cat([self.memory[i], e_b]) if len(self.memory[i]) else e_b

        assert len(self.memory[0]) == len(self.back_pointers)

    def sample(self, update, batch_size, back_view=0):
        idx, eps = self._select(batch_size, back_view)

        samples = [[] for _ in range(len(self.memory))]
        if update is not None: # now we getting hairy computationaly
            for i, e in enumerate(eps):
                experience = [ mem[range(*e)] for mem in self.memory ]
                do_update, experience = update(experience)

                if do_update:
                    for j, exp in enumerate(experience):
                        self.memory[j][range(*e)] = exp

                for j in idx[i]:
                    assert j in range(*e)
                    for s in range(len(experience)):
                        samples[s].append(experience[s][j - e[0]])

            samples = [ torch.stack(s) for s in samples ]
        else:
            samples = [ [mem[i] for i in idx] for mem in self.memory ]

        return samples

    def __len__(self):
        if not len(self.back_pointers):
            return 0
        return self.back_pointers[-1][1]

    def _select(self, batch_size, back_view):
        assert len(self)
        idx = []
        bck = []
        size = 0
        start = 0 if not back_view else (len(self) - back_view)
        while size < batch_size:
            i = random.randint(start, len(self) - 1)
            to_draw = random.randint(1, self.ep_draw)
            ep_idx = range(*self.back_pointers[i])
            idx.append(random.sample(ep_idx, to_draw))
            size += len(idx[-1])
            bck.append(self.back_pointers[i])
        return idx, bck

    def _sync(self, experience_batch):
        cum_size = len(self)
        for e in experience_batch:
            size = len(e)
            self_ptr = torch.tensor([[cum_size, cum_size + size]] * size)
            self.back_pointers = torch.cat([self.back_pointers, self_ptr]) if len(self.back_pointers) else self_ptr
            cum_size += size
        assert self.back_pointers[-1][1] == len(self.back_pointers)

    def _clip(self):
        if len(self) < self.capacity:
            return
        index = self.memory[0].shape[1] // 3
        new_bottom = self.back_pointers[index][1]
        for i in range(len(self.memory)):
            self.memory[i] = self.memory[i][new_bottom:]
        self.back_pointers = self.back_pointers[new_bottom:]
        self.back_pointers -= new_bottom
        assert self.back_pointers[-1][1] == len(self.back_pointers)
        assert self.back_pointers[0][0] == 0

out = """
import time
chunks = [3, 30, 30, 12, 9]
mem = Memory(1e+5, chunks, 20)
mem2 = Memory2(1e+5, chunks, 20)
for i in range(1000):
    batch = []
    for j in range(50):
        data = torch.cat([j + torch.ones(20, c) for c in chunks ], 1)
        batch.append(data)
#    batch = torch.stack(batch)
#    print(batch.shape)
    mem.push(batch)
    start = time.time()
    b = mem.sample(lambda e: (True, e), 1024)
    end = time.time()
    print(b[0].sum(), len(b[0]))
    t1 = end - start
    batch = [ [torch.ones(20, c)] * 50 for c in chunks ]
    mem2.push(batch)
    start = time.time()
    b = mem2.sample(lambda e: (True, e), 1024)
    end = time.time()
    print(b[0].sum(), len(b[0]))
    t2 = end - start
    print("TIMES : ", t1, t2)
print("DONE")
"""
