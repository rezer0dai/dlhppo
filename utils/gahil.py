import random
from collections import deque

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=int(capacity))

    def push(self, state, action, next_state, goal):
        self.buffer.append((state, action, next_state, goal))

    def sample(self, batch_size, back_view=0):
        idx = random.sample(range(0 if not back_view else (len(self) - back_view), len(self)), batch_size)
        state, next_state, goal, action = zip(*[self.buffer[i] for i in idx])
        state = torch.stack(state)
        next_state = torch.stack(next_state)
        goal = torch.stack(goal)
        action = torch.stack(action)
        return state, next_state, goal, action

    def __len__(self):
        return len(self.buffer)

import torch
import torch.optim as optim

import numpy as np

from model import GANReward
from utils.encoders import *

# in case we run it as singleton per process
torch.set_default_tensor_type('torch.DoubleTensor')

class GAHIL:
    def __init__(self,
            lr=3e-3, state_size=30, goal_size=3, action_size=4,
            len_target_mem=4e+4, len_other_mem=1e+5,
            behavioral=False, action_active=False):
        # init network
        self.encoder = GlobalNormalizer(state_size, 1)
        self.goal = GoalGlobalNorm(goal_size)
        self.gan_reward = GANReward(1, 1, state_size * 2, goal_size, action_size)
        self.optim = optim.Adam(self.gan_reward.parameters(), lr=lr)
        self.reward_loss = torch.nn.MSELoss()
        self.action_loss = torch.nn.MSELoss()

 # fuck me, i forgot each agent and task has its own copy wut ..
        self.encoder.share_memory()
        self.goal.share_memory()
        self.gan_reward.share_memory()
        # problem because of replay buff here, i want to avoid to use cross buffer
        # on the other side task one, should never full target buff so never learn and update reward

        self.targets = ReplayBuffer(len_target_mem)
        self.others = ReplayBuffer(len_other_mem)

        self.n_new_targets = 0
        self.n_cursor = 0

        self.o_stat = []
        self.t_stat = []
        self.v_stat = []

        self.state_size = state_size
        self.goal_size = goal_size
        self.action_size = action_size

        self.behavioral = behavioral
        self.action_active = action_active

    def register_target(self, state, next_state, goal, action):
        self.n_new_targets += len(state)
        self._push_experience(self.targets, state, next_state, goal, action)

    def register_other_with_reward(self, state, next_state, goal, action):
        s, n, g, a = self._push_experience(self.others, state, next_state, goal, action)

        self.n_cursor += s.shape[0]
        if self.n_cursor > self.others.buffer.maxlen // 3:
            self._learn(3, 2, 4096) # i guess specify trough ctor as well

        r = self._gan_critic(s, n, g, a)[0].detach()#.cpu().numpy()
        if self.behavioral:
            return r.log() * -1.
        return r

    def _gan_critic(self, s, n, g, a):
        sn, g = torch.cat([ self.encoder(s, None)[0], self.encoder(n, None)[0] ], 1), self.goal(g)
        r, a = self.gan_reward(sn, g, a)
        if self.behavioral:
            return (r.sigmoid(), a)
        else:
            return (r, a)

    def _acurracy_test(self):
        #  return
        if not self.n_new_targets:
            return

        s, a = self._sample(self.targets, self.n_new_targets, self.n_new_targets)
        rewards = self._gan_critic_from_batch(s, a)[0]

        self.n_new_targets = 0
        self.v_stat.append(rewards.mean().item())

        print("\n GAHIL LEARN ROUND :: last=%.2f test=%.2f targets=%.2f mix=%.2f (experiences T:%i .. O:%i)"%(
            self.v_stat[-1], np.mean(self.v_stat[-100:]), np.mean(self.t_stat[-100:]), np.mean(self.o_stat[-100:]),
            len(self.targets), len(self.others)))

    def _learn(self, loop_target_opt, loop_other_opt, n_samples):
        if len(self.targets) < n_samples:
            return

        self.n_cursor = 0
        self._acurracy_test()

        n_with_overlap = lambda l: n_samples * 3 // (l * 2)

        r_s_batch, r_a_batch = self._sample(self.targets, n_with_overlap(loop_target_opt))
        for _ in range(loop_target_opt):
            r_s, r_a = self._draw_random(r_s_batch, r_a_batch, n_samples)

            f_s_batch, f_a_batch = self._sample(self.others, n_with_overlap(loop_other_opt))
            for _ in range(loop_other_opt):
                f_s, f_a = self._draw_random(f_s_batch, f_a_batch, n_samples)

                fake, f_action = self._gan_critic_from_batch(f_s, f_a)
                real, r_action = self._gan_critic_from_batch(r_s, r_a)

                #debug
                self.o_stat.append(fake.detach().mean(1).mean())
                self.t_stat.append(real.detach().mean(1).mean())

                if self.behavioral:
                    gan_loss = (
                        self.reward_loss(
                            (fake - real.mean(0, keepdim=True)).mean(1),
                            torch.ones(len(fake))
                        ) + self.reward_loss(
                            (real - fake.mean(0, keepdim=True)).mean(1),
                            -torch.ones(len(real))
                        ))
                else:# evolutional : we learn reward function itself
                    gan_loss = (
                        self.reward_loss(
                            (fake - real.mean(0, keepdim=True)).mean(1),
                            -2*torch.ones(len(fake))
                        ) * .5 + self.reward_loss(
                            (real - fake.mean(0, keepdim=True)).mean(1),
                            2*torch.ones(len(real))
                        ) * 1.) + self.reward_loss(
                            real.mean(1),
                            .1 + torch.zeros(real.shape[0])
                            #torch.ones(real.shape[0])
                        ) * 1e-1

                if self.action_active:
                    action_loss = (
                        self.action_loss(
                            f_action,
                            f_a[:, :self.action_size].view(f_action.shape)
                        ) + self.action_loss(
                            r_action,
                            r_a[:, :self.action_size].view(r_action.shape)
                        )) * 1e-2

                self.optim.zero_grad()
                if self.action_active:
                    (gan_loss + action_loss).backward()
                else:
                    gan_loss.backward()
                self.optim.step()

    def _push_experience(self, memory, state, next_state, goal, action):
        state, next_state, goal, action = self._to_torch(state, next_state, goal, action)
        for s, n, g, a in zip(state, next_state, goal, action):
            memory.push(s, n, g, a)
        return state, next_state, goal, action

    def _to_torch(self, s, n, g, a):
        count = len(s)
        s = torch.from_numpy(s).view(count, -1)
        n = torch.from_numpy(n).view(count, -1)
        g = torch.from_numpy(g).view(count, -1)
        a = torch.from_numpy(a).view(count, -1)
        return s, n, g, a

    def _gan_critic_from_batch(self, gsn, a):
        chunks = [self.goal_size, self.state_size, self.state_size]
        g, s, n = gsn[:, :chunks[0]], gsn[:, chunks[0]:sum(chunks[:2])], gsn[:, sum(chunks[:2]):]
        return self._gan_critic(s, n, g, a)

    def _sample(self, memory, count, back_view=0):
        assert len(memory) >= back_view
        s, n, g, a = memory.sample(min(len(memory), count), back_view)
        return torch.cat([g, s, n], 1), a

    def _draw_random(self, batch_s, batch_a, n_samples):
        #  assert len(batch_s) > n_samples
        idx = np.random.randint(0, len(batch_s), n_samples)
        return batch_s[idx], batch_a[idx]
