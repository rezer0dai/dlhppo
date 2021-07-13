import torch
import numpy as np

from alchemy.brain import Brain
from utils.memory import Memory
from utils.rl_algos import BrainOptimizer

import time, random

#from timebudget import timebudget

class BrainDescription:
    def __init__(self,
            memory_size, batch_size,
            optim_pool_size, optim_epochs, optim_batch_size, recalc_delay,
            lr_actor, learning_delay, learning_repeat, warmup,
            sync_delta_a, sync_delta_c, tau_actor, tau_critic,
            bellman, ppo_eps, natural, mean_only, separate_actors, prio_schedule=None
            ):
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.optim_epochs = optim_epochs
        self.optim_batch_size = optim_batch_size
        self.optim_pool_size = optim_pool_size
        self.recalc_delay = recalc_delay
        self.sync_delta_a = sync_delta_a
        self.sync_delta_c = sync_delta_c
        self.learning_delay = learning_delay
        self.learning_repeat = learning_repeat
        self.warmup = warmup
        self.lr_actor = lr_actor
        self.tau_actor = tau_actor
        self.tau_critic = tau_critic
        self.bellman = bellman
        self.ppo_eps = ppo_eps
        self.natural = natural
        self.mean_only = mean_only
        self.separate_actors = separate_actors

        self.prio_schedule = prio_schedule

        self.counter = 0
        self.info = None

    def __repr__(self):
        return str([
            self.memory_size, "<- memory_size;",
            self.batch_size, "<- batch_size;",
            self.optim_epochs, "<- optim_epochs;",
            self.optim_batch_size, "<- optim_batch_size;",
            self.optim_pool_size, "<- optim_pool_size;",
            self.recalc_delay, "<- recalc_delay;",
            self.sync_delta_a, "<- sync_delta_a;",
            self.sync_delta_c, "<- sync_delta_c;",
            self.learning_delay, "<- learning_delay;",
            self.learning_repeat, "<- learning_repeat;",
            self.lr_actor, "<- lr_actor;",
            self.tau_actor, "<- tau_actor;",
            self.tau_critic, "<- tau_critic;",
            self.ppo_eps, "<- ppo_eps;",
            self.natural, "<- natural;",
            self.mean_only, "<- mean_only;",
            ])

class Agent:
    def __init__(self, 
            brains, experience,
            Actor, Critic, goal_encoder, encoder,
            n_agents, 
            n_actors, detach_actors, n_critics, detach_critics,
            stable_probs,
            resample_delay, min_step,
            state_size, action_size,
            freeze_delta, freeze_count,
            lr_critic, clip_norm, q_clip,
            model_path, save, load, delay,
            min_n_sim=None,
            loss_callback=None,
            full_model=None
            ):

        self.freeze_delta = freeze_delta
        self.freeze_count = freeze_count

        self.n_targets = 1 if not detach_actors else n_actors

        self.brain = Brain(
            Actor, Critic, encoder, goal_encoder,
            n_agents, 
            n_actors, detach_actors, n_critics, detach_critics,
            stable_probs,
            resample_delay,
            lr_critic, clip_norm, q_clip,
            model_path=model_path, save=save, load=load, delay=delay,
            loss_callback=loss_callback,
            full_model=full_model
            )
        #self.brain.share_memory() # at this point we dont know if this is needed

        self.bd_desc = brains
        self.algo = [ BrainOptimizer(self.brain, desc) for desc in self.bd_desc ]

        self.counter = 0
        self.freeze_d = 0
        self.freeze_c = 0
        self.exps = experience(self.bd_desc, self.brain)

        self.min_step = min_step
        self.warmups = [ b.warmup for b in brains ]

        self.n_simulations = None
        self.n_approved_simulations = 0
        self.min_n_sim = min_n_sim

    def step(self, info, steping):
        for i, desc in enumerate(self.bd_desc):
            self.exps.step(i, desc)

        return self._clocked_step(info, steping)

#    @timebudget
    def _clocked_step(self, info, steping):
        if self.n_simulations is None:
            return None

        if self.n_approved_simulations < self.n_simulations:
            return None

        for a_i, bd in self._select_algo(steping):
            if not bd.learning_repeat:
                continue
            bd.counter += 1
            while bd.counter < bd.learning_repeat:
                
                self._encoder_freeze_schedule()

                if True:#with timebudget("learn-round"):
                    batcher = self.exps.sample(a_i, bd)

                    bd.info = batcher(*bd.info[-1])
                    if bd.info[-1] is None:
                        bd.info = (None, None, [-1, 0])
                        continue
                    loss = self.brain.learn(
                        bd.info[:-1],
                        0 if bd.counter % bd.sync_delta_a else bd.tau_actor,
                        0 if bd.counter % bd.sync_delta_c else bd.tau_critic,
                        self.algo[a_i], a_i, bd.mean_only, bd.separate_actors)
                    return loss
            bd.counter = 0
            bd.info = None
        return None

    def save(self, goals, states, memory, actions, probs, rewards, goods, finished):
        self.n_simulations = len(states[0]) if self.min_n_sim is None else self.min_n_sim
        if self.n_approved_simulations >= self.n_simulations:
            self.n_approved_simulations = 0

        n_pushed = 0

        episode_batch = (goals, states, memory, actions, probs, rewards)
        goods = np.asarray(goods)
        if len(goods) < self.min_step:
            return

        #  goals, states, memory, actions, probs, rewards
        chunks = [0] + [e[0].shape[1] for e in episode_batch]

        full_batch = []
        for i in range(len(episode_batch[0])):
            data = torch.cat([chunk[i].type_as(probs[0]) for chunk in episode_batch], 1)
            full_batch.append(data)
        full_batch = torch.stack(full_batch).transpose(0, 1).contiguous()
        for e_i, ep in enumerate(full_batch):
            if not sum(goods[:, e_i]):
                #assert False
                continue
            
            # TODO : DELETE
#            sx = ep[:, sum(chunks[:2]):sum(chunks[:3])]
#            if sx.max() > 4. or sx.min() < -4.:
#                print("OUT OF XXX : ", sx, sx.shape, sx.max(), sx.min())
##                assert False
#                continue
#

            last_good = len(goods)-1#0 # as we can have multi env, and some eps can end faster
#            # therefore last good is also indicator of real end in episode per env
#            # as we need to walk in group, we can not let one env done before others!
#            for i, g in enumerate(reversed(goods[:, e_i])):
#                last_good = len(goods) - i
#                if g: break

#            if self.n_approved_simulations >= 3 * self.n_simulations / 2:
#                continue # experimental

            self.exps.push(ep[:last_good], chunks, e_i, goods[:last_good])

            self.n_approved_simulations += 1
            n_pushed += 1
#            print("NEXT", e_i, sum(goods[:, e_i]), len(goods), self.n_approved_simulations)

        if len(full_batch) != n_pushed: print("\n PUSH ", n_pushed, len(full_batch))

        self.exps.shuffle()

    def _select_algo(self, steping):
        if 0 == sum([0 != bd.counter for bd in self.bd_desc]):
            self.counter += steping
        # bug here, infinite loop possible if multiple brains
        for i, bd in enumerate(self.bd_desc):
            if self.warmups[i] > 0:
                self.warmups[i] -= 1
                continue

            if not bd.learning_delay:
                continue
            if self.counter % bd.learning_delay:
                continue
#            if len(self.exps) < bd.optim_batch_size:
#                continue # out of process for memserver !!
#            if 0 == random.randint(0, 10): print("BATCH STATS", len(self.exps.fast_m[i]), bd.batch_size, bd.optim_batch_size)
#            if len(self.exps.fast_m[i]) < bd.optim_batch_size:
            if len(self.exps.fast_m[i]) < bd.batch_size:
                continue
            if bd.info is None:
                bd.info = (None, None, [-1, 0])
            yield i, bd

    def _encoder_freeze_schedule(self):
        if not self.freeze_delta:
            return
        self.freeze_d += (0 == self.freeze_c)
        if self.freeze_d % self.freeze_delta:
            return
        if not self.freeze_c:
            self.brain.freeze_encoders()
        self.freeze_c += 1
        if self.freeze_c <= self.freeze_count:
            return
        self.freeze_c = 0
        self.brain.unfreeze_encoders()

    def exploit(self, goal, state, history, tind):
        return self.brain.exploit(goal, state, history, tind)

    def explore(self, goal, state, history, t):
        return self.brain.explore(goal, state, history, t)

    def sync_target(self, b, blacklist):
        self.brain.sync_target(b, blacklist)

    def sync_explorer(self, b, blacklist):
        self.brain.sync_explorer(b, blacklist)
