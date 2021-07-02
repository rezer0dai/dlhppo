import numpy as np
import random

from utils.fastmem import *
#from timebudget import timebudget

class MemoryBoost:
    def __init__(self, descs, memory, credit_assign, brain, good_reach, recalc_per_episode, recalc_per_push):
        self.fast_m = [ IRFastMemory( # TODO add branch for PER ( Priotized Experience Replay )
            desc, memory.chunks, memory.device) if cred.resampling else FastMemory(
                desc, memory.chunks) for cred, desc in zip(credit_assign, descs) ]

        self.memory = memory

        self.credit = credit_assign
        self.brain = brain
        self.good_reach = good_reach
        self.recalc_per_episode = recalc_per_episode
        self.recalc_per_push = recalc_per_push

    def __len__(self):
        assert False

    def push(self, ep, chunks, e_i, goods):
        for i in range(len(self.fast_m)):
            self._push(i, ep, chunks, e_i, goods)

    def step(self, ind, desc):
        pass

    def sample(self, ind, desc):
        def update(ind): # curry curry
            def _update(recalc_feats, indices, allowed_mask, episode):
                return self._push_to_fast(recalc_feats, ind, indices, allowed_mask, episode)
            return _update

        def dream():
            if True:#with timebudget("FullMemory-dream"):
                self.memory.dream(update(ind), desc.memory_size)
                #print("DREAM SHUFFLING")
                self.fast_m[ind].shuffle()

#        if 0 == random.randint(0, desc.recalc_delay) or desc.batch_size > len(self.fast_m[ind]):
#            #print("START DREAM")
#            dream()

        return self.fast_m[ind].sample

    #@timebudget
    def _push_to_fast(self, recalc_feats, ind, indices, allowed_mask, episode):
        goals, states, memory, actions, probs, rewards, _, _, _, _, _ = episode
        values = self.brain.qa_future(goals, states, memory, actions, ind)

        for i in range(self.recalc_per_episode):
            recalc = recalc_feats and i == self.recalc_per_episode-1

            _, ir, episode_, allowed_mask_ = self.credit[ind](goals, states, memory, actions, probs, rewards,
                    allowed_mask,
                    self.brain, recalc=recalc, indices=indices,
                    values=values)

            idx = np.arange(len(episode_))[allowed_mask_]
            if not len(idx):
                continue
            if not recalc: 
                self.fast_m[ind].push(episode_[idx], ir)

        return episode_

    #@timebudget
    def _push(self, ind, ep, chunks, e_i, goods):
        max_allowed = len(ep) - 1
        allowed_mask = [ bool(sum(goods[
            i:i+self.good_reach, e_i
            ])) for i in range(max_allowed)
                ] + [False] * (len(ep) - max_allowed)

        goals, states, memory, actions, probs, rewards = *[ep[:, sum(chunks[:i+1]):sum(chunks[:i+2])] for i in range(len(chunks[:-1]))],
        values = self.brain.qa_future(goals, states, memory, actions, ind)

        recalc = False
        for j in range(self.recalc_per_push):
            recalc = (0 == ind and self.recalc_per_push - 1 == j)
            if True:#with timebudget("credit-assign"):
                _, ir, episode, allowed_mask_ = self.credit[ind]( # it is double
                            goals, states, memory, actions, probs, rewards,
                            allowed_mask,
                            brain=self.brain,
                            recalc=recalc, 
                            values=values)

                idx = np.arange(len(episode))[allowed_mask_]
                if not len(idx):
                    continue
                if not recalc:
                    self.fast_m[ind].push(episode[idx], ir[idx])

        if 0 != ind:
            return
        assert recalc
        self.memory.push(episode, allowed_mask)

    #@timebudget
    def shuffle(self):
#        print("AFTER PUSH")
        for fast_m in self.fast_m:
            fast_m.shuffle()
