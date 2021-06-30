import random
import numpy as np
from utils.credit import CreditAssignment

class HER(CreditAssignment):
    def __init__(self, cind, her_delay, gae, n_step, floating_step, gamma, gae_tau, her_select_ratio=.4, resampling=False, kstep_ir=False, clip=None):
        super().__init__(cind, gae, n_step, floating_step, gamma, gae_tau, resampling, kstep_ir, clip)
        self.her_delay = her_delay
        self.her_select_ratio = her_select_ratio

    def _random_n_step(self, length, _indices, recalc):
        if recalc or random.randint(0, self.her_delay):
            return super()._random_n_step(length, None, recalc)

        n_steps, indices = self._do_random_n_step(length, self._do_n_step)
        her_step_inds = self._her_indices(length, n_steps, indices)
        return (True, *her_step_inds)

    def _her_indices(self, ep_len, n_steps, indices):
        cache = np.zeros(ep_len)
        h = self._her_select_idx(n_steps)
        n, i = self._select_steps(h, n_steps, indices)
        if sum(h): cache[ h ] = 1
        cache[0] = 0
        return cache, n, i

    def _her_select_idx(self, n_steps):
#        in_scope = random.randint(0, max(n_steps))
#        hers = [ i if (i < len(n_steps) - 1 and s in range(1, in_scope) and random.random() < self.her_select_ratio) else 0 for i, s in enumerate(n_steps) ]
        hers = [ i if random.random() < self.her_select_ratio else 0 for i, s in enumerate(n_steps[:-1]) ]
        return hers + [0]

    def _select_steps(self, hers, n_steps, inds):
        # AVOID COLLISIONS !!!

        if not sum(hers):
            return n_steps, inds

        n_steps[hers] = 1
        inds[hers] = np.array(hers) + 1

        # avoid O(n^2) as n is length of ep, force O(n) instead
        her = 0
        for i, ind in enumerate(inds):

            j = 0
            while i >= her:
                j += 1
                if her + j >= len(hers):
                    break
                if not hers[her+j]:
                    continue
                her += j
                j = 0
            if her + j >= len(hers):
                break

            if ind > her:
                n_steps[i] = her-1-i
                inds[i] = i + n_steps[i]
                assert inds[i] == her-1
                assert inds[i] == n_steps[i]+i

        assert all(n_steps[hers] == 1)

        return n_steps, inds
