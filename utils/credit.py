import random
import numpy as np
import torch

import utils.policy as policy

class CreditAssignment:# temporary resampling=False, if proved good then no default, need to choose!
    def __init__(self, cind, gae, n_step, floating_step, gamma, gae_tau, resampling=False, kstep_ir=False, clip=None):
        assert not kstep_ir or clip is not None
        self.cind = cind
        self.gae = gae
        self.n_step = n_step
        self.floating_step = floating_step
        self.resampling = resampling
        self.kstep_ir = kstep_ir
        self.clip = clip
        self.policy = policy.GAE(gamma, gae_tau) if gae else policy.KSTEP(gamma)

    def __call__(self,
            orig_goals, orig_states, features, actions, probs, orig_rewards,
            allowed_mask,
            brain, recalc, indices=[], values=None):

        her, indices, n_steps, n_indices = self._random_n_step(
                len(orig_states), indices, recalc)

        #  if her: print((n_indices[:-self.n_step]-np.arange(len(n_indices)-self.n_step)).mean(), sum(her_step_inds), n_indices)

        allowed_mask = torch.tensor(allowed_mask)
# even if we dont recalc here, HER or another REWARD 'shaper' will do its job!!
        ( rewards, goals, states, n_goals, n_states, allowed_mask ) = self._update_goal(her,
            orig_rewards,
            orig_goals, orig_states,
            orig_states[list(range(1, len(orig_states))) + [-1]],
            orig_goals[n_indices], orig_states[n_indices],
            actions,
            indices,
            n_steps,
            allowed_mask)

        if recalc or self.resampling:#we need everytime to resample! TODO: make it optional ?
            features, ir_ratio = brain.recalc_feats(goals, states, actions, probs, n_steps,
                    self.resampling, self.kstep_ir, self.cind, self.clip)
        else:
            ir_ratio = torch.ones(len(features))
#        elif self.gae:
#            n_steps[ [i for i in range(len(n_steps)) if i not in indices] ] = 0

        c, d = self._redistribute_rewards( # here update_goal worked on its own form of n_goal, so dont touch it here!
                n_steps, rewards, values)#brain, goals, states, features, actions)

        allowed_mask[0] = False
        allowed_mask[-2:] = False
        allowed_mask[0. == n_steps] = False

        # we by defaulf skip
        assert c.shape == rewards.shape
        assert not self.resampling or ir_ratio is not None
        return her, ir_ratio, torch.cat([
                    goals, states, features, actions, probs, orig_rewards,
                    n_goals, n_states, features[n_indices],
                    c, d] , 1), allowed_mask

    def _update_goal(self, her, rewards, goals, states, states_1, n_goals, n_states, actions, her_step_inds, n_steps, allowed_mask):
#        if not her:
#            return ( rewards, goals, states, n_goals, n_states )
        return self.update_goal(rewards, goals, states, states_1, n_goals, n_states, actions, her_step_inds, n_steps, allowed_mask)

    # duck typing
    def update_goal(self, rewards, goals, states, states_1, n_goals, n_states, actions, her_step_inds, n_steps, allowed_mask):
        assert False

    def _assign_credit(self, n_steps, rewards, values):#, brain, goals, states, features, actions):
        if not self.gae: return self.policy(n_steps, rewards)
        else: return self.policy(
                n_steps,
                rewards[:len(values) - 1],
                values)#brain.qa_future(goals, states, features, actions, self.cind))

    def _redistribute_rewards(self, n_steps, rewards, values):#, brain, goals, states, features, actions):
        # n-step, n-discount, n-return - Q(last state)
        discounts, credits = self._assign_credit(
                n_steps, rewards, values)#, breain, goals, states, features, actions)

#        discounts = torch.tensor([discounts + self.n_step*[0]])
#        credits = torch.cat([
#            torch.stack(credits), torch.zeros(self.n_step, len(credits[0]))])

        return ( torch.stack(credits), torch.tensor(discounts, device=rewards.device).view(-1, 1) )

    def _do_n_step(self, n_limit, i):
        if self.floating_step:
            return random.randint(1, n_limit)
        if n_limit < self.n_step:
            return self.gae * n_limit
        return self.n_step

    def _random_n_step(self, length, _indices, _recalc):
        return (False, [0]*length, *self._do_random_n_step(length, self._do_n_step))

    def _do_random_n_step(self, length, n_step):
        # + with indices you want to skip last self.n_step!!
        n_steps = np.array([ n_step(min(length-i-1, self.n_step), i) for i in range(length - 1) ] + [0])
        #  n_steps[-1] = min(self.n_step-1, n_steps[-1])
        indices = n_steps + np.arange(len(n_steps))
#        indices = np.hstack([indices, 1 * [-1]])
        return (n_steps, indices)
