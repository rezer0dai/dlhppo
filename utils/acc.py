import numpy as np
import torch
import torch.nn as nn

import random

class ActorCritic(nn.Module): # share common preprocessing layer!
    # encoder could be : RNN, CNN, RBF, BatchNorm / GlobalNorm, others.. and combination of those
    def __init__(self, encoder, goal_encoder, actor, critic, n_agents=1):
        super().__init__()
        self.actor = actor
        self.critic = critic

        self.n_agents = n_agents

        if goal_encoder is not None:
            self.add_module("goal_encoder", goal_encoder)

        self.add_module("encoder", encoder)

        for i, a in enumerate(self.actor):
            a.share_memory()
            self.add_module("actor_%i"%i, a)
        for i, c in enumerate(self.critic):
            c.share_memory()
            self.add_module("critic_%i"%i, c)

        self.goal_grads = [] if self.goal_encoder is None else [ p.requires_grad for p in self.goal_encoder.parameters() ]
        self.encoder_grads = [ p.requires_grad for p in self.encoder.parameters() ]

    def parameters(self):
#        assert False, "should not be accessed!"
        yield actor_parameters()
        for i in range(len(self.critic)):
            for p in critic_parameters(i):
                yield p
#        return np.concatenate([
#            list(self.actor_parameters()),
#            np.concatenate([list(self.critic_parameters(i)) for i in range(len(self.critic))], 0)])

# TODO : where to make sense to train encoder -> at Actor, Critic, or both ??
    def actor_parameters(self):
        for actor in self.actor:
            for p in actor.parameters():
                yield p
#        return np.concatenate([
#            np.concatenate([list(actor.parameters()) for actor in self.actor], 0)])

    def critic_parameters(self, ind):
        c_i = ind if ind < len(self.critic) else 0
#        print(torch.cat([ p.flatten() for p in self.critic[c_i].parameters() ]),)
#        return [ p for p in self.critic[c_i].parameters() ]
        for p in self.encoder.parameters():
            if p.requires_grad:
                yield p
        if self.goal_encoder is not None:
            for p in self.goal_encoder.parameters():
                if p.requires_grad:
                    yield p
        if -1 == ind:
            for critic in self.critic:
                for p in critic.parameters():
                    yield p
        else:
            for p in self.critic[c_i].parameters():
                yield p
#        torch.cat([ p.flatten() for p in self.critic[c_i].parameters() ]),
#        return torch.cat([
##            torch.tensor([ p for p in self.encoder.parameters() if p.requires_grad ]),
##            torch.tensor([] if self.goal_encoder is None else [ p for p in self.goal_encoder.parameters() if p.requires_grad ]),
#            torch.cat([ p.flatten() for p in self.critic[c_i].parameters() ]),
#            ])

    def forward(self, goals, states, memory, ind, a_i, ppo):# = 0):
#        assert ind != -1 or len(self.critic) == 1, "you forgot to specify which critic should be used"

        states, _ = self.encoder(states, memory)

        if self.goal_encoder is not None:
            goals = self.goal_encoder(goals)

        #  a_i = random.randint(0, len(self.actor)-1)
        a_i = a_i if a_i < len(self.actor) else 0
        dist = self.actor[a_i]( # MADDPG+MROCS settings
                goals.view(-1, goals.shape[1] // self.n_agents), states.view(-1, states.shape[1] // self.n_agents))

        actions = dist.params(ppo)

        if len(self.critic) > 1 and -1 == ind:
#            q, _ = torch.cat([ 
#                critic(goals, states, actions
#                    ) for critic in self.critic ], dim=1).min(dim=1, keepdim=True)
            q = torch.cat([
                critic(goals, states, actions
                    ) for critic in self.critic ], dim=1
                ).mean(1, keepdim=True)
        else:
            q = self.critic[ind](goals, states, actions)

        return q, dist

    def value(self, goals, states, memory, actions, ind):
        states, _ = self.encoder(states, memory)
        if self.goal_encoder is not None:
            goals = self.goal_encoder(goals)

        if len(self.critic) > 1 and -1 == ind:
#            q, _ = torch.cat([ 
#                critic(goals, states, actions
#                    ) for critic in self.critic ], dim=1).min(dim=1, keepdim=True)
            return torch.cat([
                critic(goals, states, actions
                    ) for critic in self.critic ], dim=1
                ).mean(1, keepdim=True)

        return self.critic[ind](goals, states, actions)

    def act(self, goals, states, memory, ind):
        ind = ind % len(self.actor)

        states, memory = self.encoder(states, memory)
        if self.goal_encoder is not None:
            goals = self.goal_encoder(goals)

        pi = self.actor[ind](goals, states)
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
