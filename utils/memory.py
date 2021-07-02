import torch
import random
import itertools

# we want to store tensors as CPU only i think ..
class Memory:
    def __init__(self, capacity, recalc_feats_delay, chunks, ep_draw, ep_dream):
        self.memory = []

        self.ep_draw = ep_draw
        self.ep_dream = ep_dream
        self.chunks = [0] + chunks
        self.back_pointers = []
        self.capacity = capacity
        self.allowed_mask = []

        self.recalc_feats_delay = recalc_feats_delay

    def push(self, experience, allowed_mask):
        self._clip()
        self.allowed_mask.extend(allowed_mask)
        self._sync(experience)

#        self.memory = torch.cat([self.memory, experience]).to(experience.device) if len(self.memory) else experience
        self.memory = torch.cat([self.memory, experience]).type_as(experience) if len(self.memory) else experience

        assert len(self.memory) == len(self.back_pointers)
        assert len(self.memory) == len(self.allowed_mask)

    def sample(self, update, batch_size, back_view=0):
        idx, eps = self._select(batch_size, back_view)

        if update is not None: # now we getting hairy computationaly
            samples = self._sample_with_update(idx, eps, update)
        else:
#            samples = self.memory[list(itertools.chain.from_iterable(idx))]

            return (None, [ list(itertools.chain.from_iterable(idx)), [ range(sum(self.chunks[:i+1]),sum(self.chunks[:i+2]))
                    for i in range(len(self.chunks)-1) ], self ])

        return (None, self._decompose(samples[:batch_size]))

    def dream(self, update, back_view=0):
        idx, eps = self._select(self.ep_draw * self.ep_dream // 2, back_view)
        self._sample_with_update(idx, eps, update, do_sample=False)

    def __len__(self):
        if not len(self.back_pointers):
            return 0
        return self.back_pointers[-1][1]

    def _sample_with_update(self, idx, eps, update, do_sample=True):
        samples = []
        for i, e in enumerate(eps):
            experience = self._decompose(self.memory[range(*e)])# indexing will copy tensor out

            recalc = 0 == random.randint(0, self.recalc_feats_delay)
            experience = update(
                    recalc,
                    [(j - e[0]).item() for j in idx[i]],
                    [self.allowed_mask[j] for j in range(*e)],
                    experience)
            #  experience = torch.cat(experience, 1)

#            experience = self.memory[range(*e)]
            if recalc: self.memory[range(*e)] = experience

            if not do_sample:
                continue
            for j in idx[i]:
                assert j in range(*e)
                samples.append(experience[j - e[0]])
        if not do_sample:
            return
        return torch.stack(samples)

    def _decompose(self, samples):
        return [ # indexing inside len(idx) sized batch is cheaper than query chunk for len(idx) samples
                samples[:, sum(self.chunks[:i+1]):sum(self.chunks[:i+2])
            ] for i in range(len(self.chunks)-1) ]

    def _select(self, batch_size, back_view):
        assert len(self)
        idx = []
        bck = []
        size = 0
        start = 0 if not back_view else max(0, len(self) - back_view)
        while size < batch_size:
            i = random.randint(start, len(self) - 1)
            to_draw = random.randint(1, self.ep_draw)
            ep_idx = range(*self.back_pointers[i])
            selection = random.sample(ep_idx, min(len(ep_idx)-1, to_draw))

            selection = [s for s in selection if self.allowed_mask[s]]
            if not len(selection):
                continue

            idx.append(selection)
            size += len(idx[-1])
            bck.append(self.back_pointers[i])
        return idx, bck

    def _sync(self, e):
        cum_size = len(self)
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

#        del self.memory[:new_bottom]
        self.memory = self.memory[new_bottom:].clone()

#        del self.back_pointers[:new_bottom]
        self.back_pointers = self.back_pointers[new_bottom:].clone()

        self.back_pointers -= new_bottom

        del self.allowed_mask[:new_bottom]
#        self.allowed_mask = self.allowed_mask[new_bottom:]

        assert self.back_pointers[-1][1] == len(self.back_pointers)
        assert self.back_pointers[0][0] == 0
        assert len(self.allowed_mask) == len(self.back_pointers)
