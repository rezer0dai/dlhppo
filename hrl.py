import torch

import config
from config import *
#from timebudget import timebudget
import random

from utils.her import HER, CreditAssignment


COUNTER = [0, 0]

SENTINEL = 2 # skip +1 last state dummy n_state it has, and last with valid n_state to use fast low level has +1 state we can throw when bit step_count ( 10 + ) and not waste too much experience
class ReacherHRL(HER):
    def __init__(self, cind, her_delay, gae, n_step, floating_step, gamma, gae_tau, her_select_ratio=.4, resampling=False, kstep_ir=False, clip=None):
        super().__init__(cind, her_delay, gae, n_step, floating_step, gamma, gae_tau, her_select_ratio, resampling, kstep_ir, clip)

    def _her_select_idx(self, n_steps):
        hers = [ i if random.random() < .5 else 0 for i, s in enumerate(n_steps[:-SENTINEL]) ]
        return hers + [0 for _ in range(SENTINEL)]

    #@timebudget
    def update_goal(self, rewards, goals, states, states_1, n_goals, n_states, actions, her_step_inds, n_steps, allowed_mask):
        global COUNTER

        align = lambda x: x//config.HRL_STEP_COUNT*config.HRL_STEP_COUNT

        MAX_HER_STEP = 1

        h_goals = goals.clone()
        h_n_goals = n_goals.clone()
        h_rewards = rewards.clone()

        h_states = states.clone()
        h_n_states = n_states.clone()

        assert not allowed_mask[-1] # last goal will be not used!!
        allowed_mask[-2] = False # last state_1 should not be used too

        idxs = []
        her_goals = []
        her_states = []

        x = 0
        z = 0

        hers = []
        others = []

        for i in range(HRL_STEP_COUNT, len(goals)-1, HRL_STEP_COUNT):
            if her_step_inds[i-1]:
                her_step_inds[i] = 0

#        assert 1 == HRL_STEP_COUNT or len(goals) - 1 == (len(goals) // HRL_STEP_COUNT) * HRL_STEP_COUNT

        assert not sum(her_step_inds[-2:])

        for j, (r, g, s, s2, n_g, n, u, step) in enumerate(zip(reversed(rewards), reversed(goals), reversed(states), reversed(states_1), reversed(n_goals), reversed(n_states), reversed(her_step_inds), reversed(n_steps))):
            if not step:
                continue

            i = len(goals) - 1 - j
            if i >= len(goals) - SENTINEL:
                continue

            her_active = her_step_inds[i+1]
#            assert her_active or not her_step_inds[i+step]
            if not her_active and her_step_inds[i+step]:
                allowed_mask[i] = False

            if not her_active and u:
                gro = random.randint(1, 1 + (len(goals) - i - step - SENTINEL) // HRL_STEP_COUNT)
                if random.random() < self.her_select_ratio:
                    gro = 1

            if her_active or u:
                if 1 == gro:
#                    assert i+1 == gid
                    h_rewards[i] = (config.REWARD_DELTA + torch.zeros(1, 1)) * config.REWARD_MAGNITUDE
                    x += 1
                    hers.append(i)
                else:
                    h_rewards[i] = (config.REWARD_DELTA - torch.ones(1, 1)) * config.REWARD_MAGNITUDE
                    z += 1
                    others.append(i)

#                if align(i) != align(step+i):
#                    print("===>", i, step, align(i), align(step+i), gro, align(step + i) + gro * config.HRL_STEP_COUNT)
#                assert align(step + i) + gro * config.HRL_STEP_COUNT <= 100

                hg = [align(i) + gro * config.HRL_STEP_COUNT, align(step + i) + gro * config.HRL_STEP_COUNT]
                hs = [align(i), align(step+i)]

                idxs.append(i)
                her_goals.extend(hg)
                her_states.extend(hs)

            else:
                others.append(i)

        allowed = allowed_mask[idxs]
        allowed_mask[...] = False # rest we dont know if good or not
        allowed_mask[idxs] = allowed

#        mask = np.zeros(len(n_steps))
#        mask[idxs] = 1.
#        if sum(her_step_inds): print("\nHUH", np.concatenate([np.asarray(n_steps).reshape(-1, 1), np.asarray(her_step_inds).reshape(-1, 1), mask.reshape(-1, 1)], 1))

        if len(hers):

            limit = h_states.shape[-1] if not config.TIMEFEAT else -1
            h_states[idxs, -config.CORE_ORIGINAL_GOAL_SIZE-config.TIMEFEAT:limit] = states[her_goals[0::2], :config.CORE_ORIGINAL_GOAL_SIZE].clone()
            h_n_states[idxs, -config.CORE_ORIGINAL_GOAL_SIZE-config.TIMEFEAT:limit] = states[her_goals[1::2], :config.CORE_ORIGINAL_GOAL_SIZE].clone()

            her_states_t = h_states[her_states].view(len(her_states), -1).clone()
#            assert her_states_t[1::2, -config.CORE_ORIGINAL_GOAL_SIZE:].shape == states[her_goals[1::2], :config.CORE_ORIGINAL_GOAL_SIZE].clone().shape
            her_states_t[1::2, -config.CORE_ORIGINAL_GOAL_SIZE-config.TIMEFEAT:limit] = states[her_goals[1::2], :config.CORE_ORIGINAL_GOAL_SIZE].clone()

            if config.TIMEFEAT:
                if config.TF_LOW:
                    if not config.NORMALIZE: # TODO this is just temporarerly
                        pass#assert False
                        #her_states_t[:, -1 - config.CORE_ORIGINAL_GOAL_SIZE] = 1.
                else:
                    assert False
                    her_states_t[:, -1 - config.CORE_ORIGINAL_GOAL_SIZE] = (1. - (torch.tensor(her_states) /  (1.+config.HRL_HIGH_STEP)))# * .1

            dist, _, _ = config.AGENT[1].exploit(
                    h_states[her_goals, :CORE_GOAL_SIZE].view(len(her_goals), -1),
                    her_states_t,
                    torch.zeros(len(her_goals), 1), 0)
            her_hers = dist.sample().view(len(her_goals), -1)




            if not config.DDPG:
                dist, _, _ = config.AGENT[0].exploit(
                        her_hers,
                        her_states_t,
                        torch.zeros(len(her_goals), 1), 0)
                bool_inds = (dist.log_prob(actions[her_states]).mean(1) < -1.)
                hi = torch.tensor(her_goals)[bool_inds][::2]
                allowed_mask[hi] = False

                COUNTER[len(hi) * 3 >= len(idxs) * 2] += 1
                if random.random() < .001: print("\n\n----> DISBANDED stats {} vs {}\n".format(*COUNTER))


            h_goals[idxs] = her_hers[0::2].float()
            h_n_goals[idxs] = her_hers[1::2].float()

#        print("\n ->>>>", sum(allowed_mask))

        return ( h_rewards, h_goals, h_states, h_n_goals, h_n_states, allowed_mask )
