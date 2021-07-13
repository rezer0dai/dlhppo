import numpy as np
import random, copy, sys
import itertools

from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim

#torch.set_default_dtype(torch.float16)

from utils.ac import *

from utils.polyak import POLYAK as META
#from utils.foml import FOML as META
#from utils.reptile import REPTILE as META

#from timebudget import timebudget
import logging

def def_loss_callback(pi_loss, critic_loss, *x): return pi_loss, critic_loss

class Brain(META):
    def __init__(self,
            Actor, Critic, encoder, goal_encoder,
            n_agents, 
            n_actors, detach_actors, n_critics, detach_critics,
            stable_probs,
            resample_delay,
            lr_critic, clip_norm, q_clip,
            model_path, save, load, delay,
            loss_callback,
            full_model
            ):
        super().__init__(model_path, save, load, delay)

        self.tpu_callback = None
        self.loss_callback = loss_callback if loss_callback is not None else def_loss_callback

        self.mp = model_path

        self.q_clip = q_clip
        self.clip_norm = clip_norm
        self.resample_delay = resample_delay

        self.losses = []

        self.n_actors = n_actors
        self.stable_probs = stable_probs

        #encoder.share_memory()
        #goal_encoder.share_memory()

        if save:
            Path(model_path).mkdir(parents=True, exist_ok=True)

        self.n_agents = 0
        self.global_id = "_hl_" in self.mp
        self.full_model = full_model

        nes = Actor()
# TODO i reversed here detached logic from target to behaviour
        self.ac_explorer = ActorCritic(full_model, encoder, goal_encoder,
                    [ nes.head() for _ in range(n_actors) ],
                    [ Critic() for _ in range(n_critics) ], n_agents, False, self.global_id)

        self.ac_target = ActorCritic(full_model, encoder, goal_encoder,
                    [ Actor().head() for _ in range(1 if not detach_actors else n_actors) ],
                    [ Critic() for _ in range(1 if not detach_critics else n_critics) ], n_agents, True, self.global_id)

        #print(self.ac_target)
        #print(self.ac_explorer)
        # sync
        for i in range(self.ac_explorer.n_critics):
            self.polyak_update(self.tcp(0), self.ecp(i), 1.)
#
        for i in range(self.ac_explorer.n_actors):
            self.polyak_update(self.tap(0), self.eap(i), 1.)

        return
        #  self.init_meta(lr=1e-3)

        self.load_models(0, "eac")
        self.save_models_ex(0, "eac")


        print("--->", n_critics, n_actors, len(self.ac_explorer.critic), len(self.ac_explorer.actor), len(self.ac_target.critic), len(self.ac_target.actor))

#        self.full_optimizer = Ranger(self.ac_explorer.parameters(), lr=lr_critic, eps=1e-5, weight_decay=1e-3)
        self.full_optimizer = optim.Adam(self.ac_explorer.parameters(), lr=lr_critic)

        self.resample(0)

    def tap(self, ind): return self.full_model.actor_target_parameters(ind, "highpi" if self.global_id else "lowpi")
    def eap(self, ind): return self.full_model.actor_explorer_parameters(ind, "highpi" if self.global_id else "lowpi")

    def tcp(self, ind):
        if config.N_CRITICS == 1:
            assert 0 == ind
            if self.global_id:
                return self.full_model.critic_explorer_parameters(0, "lowpi")
            return self.full_model.critic_target_parameters(0, "lowpi")
        return self.full_model.critic_target_parameters(self.global_id % (config.N_CRITICS - (not config.DETACH_CRITICS)), "lowpi")

    def ecp(self, ind): 
        if config.N_CRITICS == 1:
            assert 0 == ind
            if self.global_id:
                return self.full_model.critic_target_parameters(0, "lowpi")
            return self.full_model.critic_explorer_parameters(0, "lowpi")
        return self.full_model.critic_explorer_parameters(self.global_id, "lowpi")

    #@timebudget
    def learn(self, batch, tau_actor, tau_critic, backward_policy, tind, mean_only, separate_actors):
        loss = self._learn(batch, tau_actor, tau_critic, backward_policy, tind, mean_only, separate_actors)
        return loss

    #@timebudget
    def _learn(self, batch, tau_actor, tau_critic, backward_policy, tind, mean_only, separate_actors):
        if True:#with timebudget("_learn_debatch"):
            w_is, (goals, states, memory, actions, old_probs, r, n_goals, n_states, n_memory, n_rewards, n_discounts) = batch

        if not len(goals):
            return
        assert len(goals)

        #print("LEARNNNN->", len(goals))

        probs = old_probs.mean(1)

        self.losses.append([])

        assert all(n_discounts != 0.)

        clip = 2e-1

        if True:#with timebudget("_learn_future"):
    # SELF-play ~ get baseline
            with torch.no_grad():
                n_qa, n_dist = self.ac_target.suboptimal_qa(n_goals, n_states, n_memory)

                qa_stable = self.ac_explorer.qa_stable(goals, states, memory, actions, -1)

            # TD(0) with k-step estimators
            td_targets = n_rewards + n_discounts * n_qa

        sync_delta = 3
        loss = 0.
        if True:#with timebudget("_learn_backprop"):
    #        if "lowlevel" in self.mp and random.random() < .1:
            for s in range(sync_delta):

    # activate gradients ~ SELF-play
                pi_loss = []
                
                q_replay, dists, probs_, offline_actions, offline_goals = self.ac_explorer(
                        goals, states, memory, self.global_id, 0, mean_only, 
                        probs=probs, old_pis=actions)

                if actions.shape[-1] != offline_goals.shape[-1] * 3:
                    mask = 0. == r
#                    print("\n\n???", actions.shape, sum(mask), r)
                else:
                    mask = torch.ones(len(offline_goals))
                # learn ACTOR ~ explorer
                pi_loss, optimizer = backward_policy(
#                        qa_stable, td_targets, w_is,
                        q_replay, td_targets, w_is,
                        probs_, actions, dists,
                        offline_actions=offline_actions, offline_goals=offline_goals, mask=mask,
                        _eval=None, retain_graph=False)#(sync_delta-1 != s))#surrogate_loss)

                cl_clip = qa_stable + torch.clamp(q_replay - qa_stable, -clip, clip)
                cl_clip = (cl_clip - td_targets).pow(2).mean(1)

                cl_raw = (q_replay - td_targets).pow(2).mean(1)
                critic_loss = (torch.max(cl_raw, cl_clip) * (w_is if w_is is not None else 1.)).mean()

                pi_loss, critic_loss = self.loss_callback(pi_loss, critic_loss, self, actions, goals, states, memory, qa_stable, n_dist)
                loss += (pi_loss + critic_loss) * .5

        loss = loss / sync_delta

        target = self.tcp(self.global_id * config.DETACH_CRITICS)
        explorer = self.ecp(random.randint(0, self.ac_explorer.n_critics-1))

        self.meta_update(
                None,
                explorer,
                target,
                tau_critic)

        for aitd in range(self.ac_target.n_actors):
            aied = random.randint(0, self.ac_explorer.n_actors-1)

            target = self.tap(aitd)
            explorer = self.eap(aied)

            self.meta_update(
                    None,
                    explorer,
                    target,
                    tau_actor)

        return loss

        self.save_models(0, "eac")

        if random.random() < .1 and sync_delta_c:
            aa = torch.cat([p.view(-1) for p in config.AGENT[0].brain.ac_explorer.critic_parameters(-1)]).sum()
            ab = torch.cat([p.view(-1) for p in config.AGENT[0].brain.ac_target.critic_parameters(0)]).sum()

            ba = torch.cat([p.view(-1) for p in config.AGENT[1].brain.ac_explorer.critic_parameters(-1)]).sum()
            bb = torch.cat([p.view(-1) for p in config.AGENT[1].brain.ac_target.critic_parameters(0)]).sum()

#            msg = "\n{}\n[{}]LOSSES=>{}:{}:{}\n\tRAW:{}\nQ:{}\nN_Q:{}\nSUMS:{}:{}--{}:{}\n".format("*"*80, len(goals), pi_loss, cl_clip.mean(), cl_raw.mean(), cl_raw[:3], n_qa[:3], q_replay[:3],
            msg = "\n{}\n[{}]LOSSES=>{}:{}:{}\n\tRAW:{}||CLIP{}\nQ:{}\nN_Q:{}\nSUMS:{}:{}--{}:{}\n".format("*"*80, len(goals), 
                    pi_loss, critic_loss, cl_raw.mean(), 
                    cl_raw[:3], cl_clip[:3], n_qa[:3], q_replay[:3],
                    aa, ab, ba, bb)
#            print(msg)
            logging.warning(msg)

    def resample(self, t):
        return#TODO
        if 0 != t % self.resample_delay:
            return
        for actor in self.ac_explorer.actor:
            actor.sample_noise(t // self.resample_delay)

    def explore(self, goal, state, memory, t): # exploration action
        self.resample(t)
        with torch.no_grad(): # should run trough all explorers i guess, random one to choose ?
            e_dist, mem = self.ac_explorer.act(goal, state, memory, -1)

            if not self.stable_probs:
                t_dist = e_dist
            else:
                t_dist, _ = self.ac_target.act(goal, state, memory, -1)

        return e_dist, mem, t_dist

    def exploit(self, goal, state, memory, tind): # exploitation action
        with torch.no_grad():
            dist, mem = self.ac_target.act(goal, state, memory, -1)#tind % len(self.ac_target.actor))
        return dist, mem, dist

    def qa_future(self, goals, states, memory, actions, cind):
        with torch.no_grad():
            return self.ac_target.qa_stable(goals, states, memory, actions, -1)

    #@timebudget
    def backprop(self, optim, loss, params, callback=None, just_grads=False, retain_graph=False):
        # learn
        optim.zero_grad() # scatter previous optimizer leftovers
        loss.backward(retain_graph=retain_graph) # propagate gradients
        torch.nn.utils.clip_grad_norm_(params, self.clip_norm) # avoid (inf, nan) stuffs

        if just_grads:
            return # we want to have active grads but not to do backprop!

        if callback is not None:
            optim.step(callback) # trigger backprop with natural gradient
        else:
            optim.step() # trigger backprop

    #@timebudget
    def recalc_feats(self, goals, states, actions, e_log_probs, n_steps, resampling, kstep_ir, tind, clip):
        return torch.zeros(len(goals), 1).type_as(goals), torch.ones(len(goals), 1).type_as(goals)

        with torch.no_grad():
            _, f = self.ac_target.encoder.extract_features(states)

            if not resampling:
                return f, torch.ones(len(f))

            if True:#not self.stable_probs:#
#                assert False
                e_dist, _ = self.ac_explorer.act(goals, states, f, 0)
                e_log_probs = e_dist.log_prob(actions)

            t_dist, _ = self.ac_target.act(goals, states, f, tind)
            t_log_probs = t_dist.log_prob(actions)

            ir_ratio = (t_log_probs - e_log_probs).exp().mean(1)
            if kstep_ir:
                ir_ratio = torch.tensor([ir_ratio[i:i+k].mean() for i, k in enumerate(n_steps)])

            ir_ratio[-1] = ir_ratio[1:-1].mean()

#            print("R", ir_ratio.mean())
#            if ir_ratio.mean() > 1.+clip:
#                return f, 1.-clip + torch.zeros(ir_ratio.shape), e_log_probs

            if 0 == random.randint(0, 200): print("\nIRSTEP : ", ir_ratio[:-1].mean(), ir_ratio.median(), ir_ratio, t_log_probs.mean(), e_log_probs.mean(), kstep_ir)

            ir_ratio = torch.clamp(ir_ratio, min=1.-clip, max=1.+clip)
        return f, ir_ratio

    def freeze_encoders(self):
        return
        self.ac_explorer.freeze_encoders()
        self.ac_target.freeze_encoders()

    def unfreeze_encoders(self):
        return
        self.ac_explorer.unfreeze_encoders()
        self.ac_target.unfreeze_encoders()

