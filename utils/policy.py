import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

#from timebudget import timebudget

#TODO : KICK OUT
import config

class PGDist:
    def __init__(self, dist):
        self.dist = dist
    def q_action(self, action):
        return self.dist.q_action(action.view(self.dist.loc.shape)).detach().cpu()
    def params(self, mean_only):
        if mean_only:
            return torch.cat([self.sample(), self.dist.loc, self.dist.scale.detach()], 1)
        return torch.cat([self.sample(), self.dist.loc, self.dist.scale], 1)
    def action(self):
        with torch.no_grad():
            return self.sample().cpu()
    def sample(self):
        return self.dist.sample()
    def log_prob(self, actions):
        return self.dist.log_prob(actions[:, :self.dist.loc.shape[-1]])

class NormalWithAction(Normal):
    def q_action(self, action): # here we want actual mu to pass to Q-function
        return torch.cat([action, self.loc, self.scale], 1)

class PPOHead(nn.Module):
    def __init__(self, action_size, f_clip, noise_scale=0.):
        super().__init__()
#        assert noise_scale >= .0 and noise_scale <= 1.
        self.f_clip = f_clip
        self.log_std = nn.Parameter((1.-noise_scale) * torch.ones(1, action_size))
    def forward(self, mu):
        sigma = self.f_clip(self.log_std).exp().expand_as(mu) # best
#        sigma = self.log_std.tanh().exp().expand_as(mu) # best
#        sigma = self.log_std.expand_as(mu).min((mu.abs() / 10).log()).exp()
#        sigma = self.log_std.sigmoid().exp().expand_as(mu)
        dist = NormalWithAction(mu, sigma)
        return PGDist(dist) # retain gradients in mu and sigma

class DDPGHead(nn.Module):
    def __init__(self):
        super().__init__()
        assert config.DDPG
        self.action = None
        self.loc = None
        self.scale = None
    def forward(self, action):
        self.action = action
        self.loc = torch.ones_like(action)
        self.scale = torch.ones_like(action)
        return PGDist(self)
    def sample(self):
        return self.action
    def log_prob(self, actions):
        return torch.ones_like(actions)

class RLLoss:
    def __init__(self, advantages=True, boost=False):
        self.advantages_enabled = advantages
        self.advantages_boost = boost

    def pi_error(self, qa, td_targets):
        if not self.advantages_enabled:
            return qa

        # w.r.t. in-env-played action
        td_error = (td_targets - qa)

#        td_error = (td_error - td_error.mean()) / (1e-8 + td_error.std())

# NEXT section is obsolete unproperly tested older idea, TODO test or kick off
        if not self.advantages_boost:
            return td_error

        for i, err in enumerate(td_error):
            for j, e in enumerate(err):
                if abs(e) < 1e-5: td_error[i][j] = qa[i][j]
        return td_error

class PPOLoss(RLLoss):
    def __init__(self, eps=2e-1, advantages=True, boost=False):
        super().__init__(advantages, boost)

        self.eps = eps

        self.po_c = 0
        self.vg_c = 0

    def ppo(self, diff, adv):
        """ paper : https://arxiv.org/abs/1707.06347
            + using grads from policy probabilities, clamping them...
            - however not efficient to use with replay buffer ( past action obsolete, ratio always clipped )
        """
        ratio = diff.exp()

        surr1 = torch.clamp(ratio, min=1.-self.eps, max=1.+self.eps) * adv
#        surr2 = ratio * adv
        surr2 = torch.clamp(ratio, min=-100., max=100.) * adv # we clip also from bottom side!
        #  print("\nratio:", ratio, "clipped", surr1 > surr2)
        grads = torch.min(surr1, surr2)

        return grads

    def ppo_loss(self, old_probs, new_probs, adv):
# this TODO check if mean should be inside or outside like now
#        diff = (new_probs - old_probs).mean(1)
#        diff = (new_probs.mean(1) - old_probs.mean(1))
        diff = new_probs - old_probs
#        if config.DOUBLE_LEARNING:
#            diff = diff * .5

        if adv.grad_fn is None:
            # debug
            vg_c = self.vg_c
            if diff.abs().mean() < 2:
                self.po_c += 1
            else:
                self.vg_c += 1

            if vg_c != self.vg_c and 0 == self.vg_c % 20:
                print("PPO too off, from sampled actions, policies problems!! ",
                        len(diff), self.vg_c, self.po_c, new_probs.mean(), old_probs.mean())

        return self.ppo(diff, adv)

    def __call__(self, qa, td_targets, old_probs, new_probs):
        adv = self.pi_error(qa, td_targets)
        adv = adv.view(len(adv), -1).sum(1) # maximizing MROCS, cooperation between subtask approach
        return self.ppo_loss(old_probs, new_probs, adv)

class PPOBCLoss(PPOLoss):
    def __call__(self, adv, old_probs, new_probs, online_actions, offline_actions, mask):
        loss1 = adv.view(len(adv), -1).sum(1) # maximizing MROCS, cooperation between subtask approach
        loss2 = F.mse_loss(offline_actions, online_actions) * mask
        #print("\n\n ----> %.2f ----> %.2f\n", loss1, loss2)
        loss = loss1 + .1 * loss2
        return self.ppo_loss(old_probs, new_probs, loss)

class TD3BCLoss:
    def td3bc(self, qa):
        """ paper : 
            + offline learning
            - online learning
        """
        lmbda = 1.5 / qa.abs().mean().detach()
        return lmbda * qa.mean()

    def __call__(self, online_actions, offline_actions, advantages, mask):
        advantages = advantages.view(len(advantages), -1).sum(1) # maximizing MROCS, cooperation between subtask approach

        bc_loss = F.mse_loss(offline_actions, online_actions) * mask
#        loss = self.td3bc(advantages)
        # advantages are already normalized
        loss = advantages.mean()
#        print("\nTD3+BC", loss.mean(), bc_loss.mean(), "??", offline_actions.sum(), "...")
        return loss - bc_loss

class DDPGLoss(RLLoss):
    def ddpg(self, loss):
        """ paper : https://arxiv.org/abs/1509.02971
            + effective to use with replay buffer
            - using grads from critic not from actual pollicy ( harder to conv )
        """
        # w.r.t. self-played action
        return -loss

    def __call__(self, qa, td_targets):
        loss = self.pi_error(qa, td_targets)
        loss = loss.view(len(loss), -1).sum(1) # maximizing MROCS, cooperation between subtask approach
        return self.ddpg(loss)

class GAE:
    def __init__(self, gamma, tau):
        self.gamma = gamma
        self.tau = tau

    def _gae(self, value, values, rewards):
        """ paper : https://arxiv.org/abs/1506.02438
            explained : https://danieltakeshi.github.io/2017/04/02/notes-on-the-generalized-advantage-estimation-paper/
            code : https://github.com/higgsfield/RL-Adventure-2 : ppo notebook
            + low variance + low bias
            - perf bottleneck to implement with replay buffer
        """
        discount = self.gamma
        gae = rewards[-1] + discount * value - values[-1] # one step gae
        # calc multi step
        for step in reversed(range(len(rewards) - 1)):
            delta = rewards[step] + self.gamma * values[step + 1] - values[step]
            gae = delta + self.gamma * self.tau * gae
            discount *= self.gamma * self.tau
        return gae - discount * value + values[0]

    def gae(self, n_steps, rewards, values):
        #ONE SECOND DIFF!!!
#        rewards = rewards.cpu().numpy() # speedup A LOT!!
#        values = values.cpu().numpy() # speedup A LOT!! ! before maybe becayse double, vs float ? now float default so the same ? RETEST !
        return [ self._gae(
            values[i+n], values[i:i+n], rewards[i:i+n]
            ) if n > 0 else torch.zeros(*values[i].shape, device=rewards.device) for i, n in enumerate(n_steps) ]

    def gae_discount(self, n_steps):
        return [ self.gamma*((self.gamma * self.tau)**(n-1)) for n in n_steps ]

    #@timebudget
    def __call__(self, n_steps, rewards, values):
        return self.gae_discount(n_steps), self.gae(n_steps, rewards, values)

class KSTEP:
    def __init__(self, gamma):
        self.gamma = gamma

    def k_step(self, n_steps, rewards):
        """ paper : ...
            + low variance
            - high bias
        """
        rewards = rewards.cpu().numpy() # speedup A LOT!!
        reward = lambda data: np.sum([r * (self.gamma ** i) for i, r in enumerate(data)], axis=0).reshape(-1)
        return [ torch.from_numpy(
            reward(rewards[i:i+n])).float() for i, n in enumerate(n_steps) ]

    def k_discount(self, n_steps):
        return [ self.gamma**n for n in n_steps ]

    #@timebudget
    def __call__(self, n_steps, rewards):
        return self.k_discount(n_steps), self.k_step(n_steps, rewards)
