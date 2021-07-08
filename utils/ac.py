import numpy as np
import torch
import torch.nn as nn

import random
# TODO
import config

class ActorCritic(nn.Module): # share common preprocessing layer!
    # encoder could be : RNN, CNN, RBF, BatchNorm / GlobalNorm, others.. and combination of those
    def __init__(self, full_model, encoder, goal_encoder, actor, critic, n_agents, target, global_id):
        super().__init__()

        self.n_actors = len(actor)
        self.n_critics = len(critic)

        self.global_id = global_id

        self.target = target

#        self.actor = actor
#TODO oldsxuul->False:#
        if len(config.AGENT) > 0 and len(config.AGENT) != 2:
            #TODO REVERSED
#            if target:#not target:#
            if (1 != config.N_CRITICS and target) or (1 == config.N_CRITICS and not target):#not target:#
#                self.critic = config.AGENT[0].brain.ac_target.critic
                critic = config.AGENT[0].brain.ac_target.critic
            else:
#                self.critic = config.AGENT[0].brain.ac_explorer.critic
                critic = config.AGENT[0].brain.ac_explorer.critic
#        else:
#            self.critic = critic

        self.n_agents = n_agents

#        if goal_encoder is not None:
#            self.add_module("goal_encoder", goal_encoder)

#        self.add_module("encoder", encoder)

#        for i, a in enumerate(self.actor):
#            a.share_memory()
#            self.add_module("actor_%i"%i, a)
#        for i, c in enumerate(self.critic):
#            c.share_memory()
#            self.add_module("critic_%i"%i, c)

#        self.goal_grads = [] if self.goal_encoder is None else [ p.requires_grad for p in self.goal_encoder.parameters() ]
#        self.encoder_grads = [ p.requires_grad for p in self.encoder.parameters() ]

        for i, a in enumerate(actor):
            full_model.register(self.decorate("actor_%i"%i), a)
#        self.actor = lambda i: full_model[self.decorate("actor_%i"%i)]


        if not len(config.AGENT):
            for i, c in enumerate(critic):
                full_model.register(self.decorate("critic_%i"%i), c)
#            self.critic = lambda i: full_model[self.decorate("critic_%i"%i)]
#        else:
#            self.critic = critic

        full_model.register(self.decorate("encoder"), encoder)
        full_model.register(self.decorate("goal_encoder"), goal_encoder)

        self.full_model = full_model

    def decorate(self, sid, global_id=None, target=None):
        if global_id is None:
            global_id = self.global_id

        if config.N_CRITICS != 1 or target is None:
            target = self.target # we want this swap only for critic and only if we target exchanged ( HL<->LL ) critic trainig

        return sid+"_%s_%s"%("target" if target else "explorer", "highpi" if global_id else "lowpi")

    def critic(self, i):
        return self.full_model[self.decorate("critic_%i"%i, 0, self.target if not self.global_id else not self.target)]

    def actor(self, i):
        return self.full_model[self.decorate("actor_%i"%i)]

    def encoder(self, x, m):
        with torch.no_grad(): 
            return self.full_model[self.decorate("encoder")](x, m)

    def goal_encoder(self, x):
        with torch.no_grad(): 
            return self.full_model[self.decorate("goal_encoder")](x)


#    def parameters(self):
#        for p in self.actor_parameters():
#            yield p
#        for p in self.critic_parameters(-1):
#            yield p
#
## TODO : where to make sense to train encoder -> at Actor, Critic, or both ??
#    def actor_parameters(self):
#        for actor in self.actor:
#            for p in actor.parameters():
#                yield p
#
#        if config.DOUBLE_LEARNING and self.global_id:#False:#
#            eid = 0#(2 == len(config.AGENT))
#            for actor in (config.AGENT[eid].brain.ac_explorer.actor if config.DL_EXPLORER else config.AGENT[eid].brain.ac_target.actor):
#                for p in actor.parameters():
#                    yield p
#
#        if self.goal_encoder is not None:
#            for p in self.goal_encoder.parameters():
#                if p.requires_grad:
#                    yield p
#
#    def critic_parameters(self, ind):
#        c_i = ind if ind < len(self.critic) else 0
#        for p in self.encoder.parameters():
#            if p.requires_grad:
#                yield p
#        if -1 == ind:
#            for p in self.critic[self.global_id % len(self.critic)].parameters():
#                yield p
##            for critic in self.critic:
##                for p in critic.parameters():
##                    yield p
#        else:
#            for p in self.critic[c_i].parameters():
#                yield p

    def forward_impl(self, goals, states, memory, a_i, mean_only, probs = None):# = 0):
        assert not mean_only

        a_i = a_i % self.n_actors
        if config.LLACTOR_UNOMRED and 3 != goals.shape[-1]:
            dist = self.actor(a_i)(
                    goals.view(-1, goals.shape[1] // self.n_agents), 
                    states.view(-1, states.shape[1] // self.n_agents))
        else:
            states_, _ = self.encoder(states, memory)
            goals_ = self.goal_encoder(goals)

            dist = self.actor(a_i)(
                    goals_.view(goals.shape[0], -1),
                    states_.view(-1, states.shape[1] // self.n_agents))
        pi = dist.params(mean_only)

        if goals.shape[-1] == 3:
            forward = False
            ll_goals = pi[:, :pi.shape[-1]//3]

#TODO DOUBLEL
            # incompatible with LSTM/GRU memory, or we need to pass it at the end of states like goal we do now
            if config.DL_EXPLORER and config.DOUBLE_LEARNING and not self.target:
                d, _ = config.AGENT[0].brain.ac_explorer.act(ll_goals, states, memory, -1)
            else:
                d, _ = config.AGENT[0].brain.ac_target.act(ll_goals, states, memory, -1)

#TODO PROPER TEST
            if config.DOUBLE_LEARNING and not config.DDPG:
                if probs is not None:
                    old_prob = pi[:, config.HRL_ACTION_SIZE:config.HRL_ACTION_SIZE+config.ACTION_SIZE].mean(1)
                    actionsZ = pi[:, config.HRL_ACTION_SIZE+config.ACTION_SIZE:config.HRL_ACTION_SIZE+config.ACTION_SIZE*2]
                    new_prob = d.log_prob(actionsZ).mean(1)
                    probs = probs.clone() + (new_prob - old_prob)# * .1#5

            pi = d.params(mean_only)
        else:
            forward = True
            ll_goals = goals.clone()
            limit = goals.shape[-1] if not config.TIMEFEAT else -1
            goals = states[:, :limit][:, -config.CORE_ORIGINAL_GOAL_SIZE:]

        actions = pi[:, :pi.shape[-1]//3]
        return dist, probs, actions, goals, ll_goals

    def forward(self, goals, states, memory, ind, a_i, mean_only, probs = None):# = 0):
        dist, probs, actions, goals, ll_goals = self.forward_impl(goals, states, memory, a_i, mean_only, probs)
        if probs is not None:
            q = self._value(ll_goals, states, memory, actions, ind)
        else:
            with torch.no_grad():
                q = self._value(ll_goals, states, memory, actions, ind)
        return q, dist, probs

# torch no grad should be whole function!!
    def qa_stable(self, goals, states, memory, actions, ind):
        if goals.shape[-1] == 3:
            forward = False
            ll_goals = actions[:, :actions.shape[-1]//3]
            _, d, _ = config.AGENT[0].brain.ac_target(ll_goals, states, memory, ind, -1, False)
            actions = d.params(False)
        else:
            forward = True
            ll_goals = goals.clone()
            limit = goals.shape[-1] if not config.TIMEFEAT else -1
            goals = states[:, :limit][:, -config.CORE_ORIGINAL_GOAL_SIZE:]

        actions = actions[:, :actions.shape[-1]//3]
        return self._value(ll_goals, states, memory, 
#                torch.cat([actions, ll_goals, ], 1), 
                actions,# if not config.NO_GOAL else torch.cat([actions, ll_goals, ], 1), 

#TODO oldsxuul
#                ll_goals if self.global_id else torch.cat([actions, ll_goals, ], 1),

                ind)#, forward)

    def _value(self, goals, states, memory, actions, ind, forward=True):#False):
#        assert 3 == goals.shape[-1]
#        assert all( all(g==s) for g, s in zip(goals, states[:, -3:]))

#        assert not config.NORMALIZE
#TODO oldsxuul->False:#
        if forward: #TODO remove -> it is here because of normalization!!
            # TODO REVERSED
#            if True:#self.global_id:#
            if 1 != config.N_CRITICS or self.global_id:
                if self.target:
                    return config.AGENT[1].brain.ac_target._value(goals, states, memory, actions, ind, False)
                else:
                    return config.AGENT[1].brain.ac_explorer._value(goals, states, memory, actions, ind, False)
            else:
                if not self.target:
                    return config.AGENT[1].brain.ac_target._value(goals, states, memory, actions, ind, False)
                else:
                    return config.AGENT[1].brain.ac_explorer._value(goals, states, memory, actions, ind, False)

        if config.CRITIC_UNORMED:
            states_ = states
            goals_ = goals
        else:
            states_, _ = self.encoder(states, memory)
            goals_ = goals
#            if self.goal_encoder is not None: # we disabled original goals at critic, and high level goals already normalized by tanh
#                goals_ = self.goal_encoder(goals)

        if self.n_critics == 1 or -1 != ind:
            return self.critic(ind % self.n_critics)(goals_, states_, actions)

        q = torch.cat([
            self.critic(i)(goals_, states_, actions
                ) for i in range(self.n_critics) ], dim=1
            )
#        if True:#random.random() < .01:
#            print("\nQMIN", q.min(dim=1, keepdim=True)[1].sum(), q.shape[0])
#        return q.min(dim=1, keepdim=True)[0]
        return q.mean(dim=1, keepdim=True)

#also torch no grad whole function should be!!
    def suboptimal_qa(self, goals, states, memory):
        dist, _probs, actions, goals, ll_goals = self.forward_impl(goals, states, memory, 0, False)
        q = self._value(ll_goals, states, memory, 
                actions,# if not config.NO_GOAL else torch.cat([actions, ll_goals, ], 1), 
                -1)
        return q, dist
        qa, dist, _ = self.forward(goals, states, memory, -1, 0, False, True)
        return qa, dist
# for now abandon idea of two targets, not used now anyway, easier tinkering with CQL
        q = torch.cat([ 
            self.forward(goals, states, memory, -1, i, False
                )[0] for i in range(self.n_actors) ], 1
            )
        return q.min(dim=1, keepdim=True)[0] # target overstimation
        return q.max(dim=1, keepdim=True)[0]

    def act(self, goals, states, memory, ind):
        if self.n_actors > 1 and -1 == ind:
            q = torch.cat([ 
                self.forward(goals, states, memory, -1, i, False
                    )[0] for i in range(self.n_actors) ], 1
                ).mean(0, keepdim=True)
            ind = q.max(-1, keepdim=True)[1]
        else:
            assert ind < self.n_actors
            ind = 0 if -1 == ind else ind

        if config.LLACTOR_UNOMRED and 3 != goals.shape[-1]:
            return self.actor(ind)(goals, states), memory

        states_, _ = self.encoder(states, memory)
        if self.goal_encoder is not None:
            goals_ = self.goal_encoder(goals)
        pi = self.actor(ind)(goals_, states_)
        return pi, memory

    def freeze_encoders(self):
        for p in self.encoder.parameters():
            p.requires_grad = False
        if self.goal_encoder is None:
            return
        for p in self.goal_encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoders(self):
        for g, p in zip(self.encoder_grads, self.encoder.parameters()):
            p.requires_grad = g
        if self.goal_encoder is None:
            return
        for g, p in zip(self.goal_grads, self.goal_encoder.parameters()):
            p.requires_grad = g
