import torch

import numpy as np
import random#, timebudget

import config

class Info:
    def __init__(self, states, rewards, actions, custom_rewards, dones, goals, goods, pi):
        self.states = states
        self.rewards = rewards
        self.actions = actions
        self.custom_rewards = custom_rewards
        self.dones = dones
        self.goals = goals
        self.goods = goods
        self.pi = pi

ones = lambda *shape: torch.ones(*shape)
zeros = lambda *shape: torch.zeros(*shape)

tensor = lambda x, shape: torch.tensor(x).view(*shape)

achieved_goals = lambda states: states[:, :config.CORE_ORIGINAL_GOAL_SIZE]

MOVE_DIST = 3e-4 
moved = lambda s1, s: MOVE_DIST < torch.norm(s1 - s)

def select_exp(states_1, states):
    if not config.SELECT_EXP:
        return [True] * len(states)
    return [moved(s1, s) for s1, s in zip(achieved_goals(states_1), achieved_goals(states))]

from tasks.oaiproc import GymGroup, GymRender

class LowLevelCtrlTask:
    def __init__(self, dock, prefix):
        self.dock = dock
        self.prefix = prefix
        self.ENV = None#GymGroup(config.TEST_ENVS, dock, config.TOTAL_ENV, prefix)
        self.RENDER = None#GymRender(config.ENV_NAME, config.TOTAL_ENV)

        self.goals = None
        self.info = None
        self.set_goods(False)

    def set_goods(self, goods):
        self.goods = goods

    def get_info(self):
        return self.info

    def set_goals(self, goals):
        self.goals = goals

    def get_goals(self):
        return self.goals

    def optimal_goals(self, base_states, end_states):
        limit = base_states.shape[-1] if not config.TIMEFEAT else -1
        base_states[:, :limit][:, -config.CORE_ORIGINAL_GOAL_SIZE:] = achieved_goals(self.info.states)
        dist, _, _ = self.hl_agent.exploit(
            end_states[:, :config.CORE_GOAL_SIZE],
            base_states,
            zeros(len(self.info.states), 1), 0)
        return dist.sample()

    # time feature is only for critic
    def append_time_feat(self, states):
        self.n_steps += 1
        if not config.TIMEFEAT:
            return states
        assert self.n_steps <= (2. + config.HRL_HIGH_STEP * config.HRL_STEP_COUNT)
        tf = ones(states.shape[0], 1) - (self.n_steps /  (3. + config.HRL_HIGH_STEP * config.HRL_STEP_COUNT))
        return torch.cat([states, tf], 1)

    def _state(self,
            einfo, actions, pi,
            learn_mode=False, reset=False, seed=None):

        states, goals, rewards, dones = einfo

        states = torch.cat([states, goals], 1)
        if config.CORE_GOAL_SIZE != config.CORE_ORIGINAL_GOAL_SIZE:
            goals = np.concatenate([goals, self.orig_pos], 1)

        states = self.append_time_feat(states)

        goods = self.goods if self.goods is not None else [False for _ in range(len(states))]
        rewards = tensor(rewards, [-1, 1])
        self.info = Info(
                states,
                rewards,
                actions,
# custom rewards here does not matter, all experience will be "dreamed" but based on true exp
                rewards,
                tensor(dones, [len(dones), -1]).float(),
                self.goals,
                goods,
                pi,
                )

        return self.info

    def reset(self, agent, seed, learn_mode):
        return self.info

    def internal_reset(self, agent, seed, learn_mode):
        if self.ENV is None:
            self.ENV = GymGroup(config.TEST_ENVS, self.dock, config.TOTAL_ENV, self.prefix)
            self.RENDER = GymRender(config.ENV_NAME, config.TOTAL_ENV)

        self.hl_agent = agent
        self.n_steps = 0
        self.learn_mode = learn_mode

        if self.learn_mode:
            einfo = self.ENV.reset(seed)
        else:
            einfo = self.RENDER.reset(seed)

        self.set_goods(None)

        self._state(
            einfo,
            None, None,
            learn_mode, True, seed)

        return einfo[1]

    def step(self, pi):
        if self.learn_mode:
            einfo = self.ENV.step(
                    pi[:, :pi.shape[1]//3].cpu().numpy(), ones(len(pi)))# if sum(self.goods) else None)
        else:
            einfo = self.RENDER.step(
                    pi[:, :pi.shape[1]//3].cpu().numpy())

        return self._state(einfo, pi[:, :config.ACTION_SIZE], pi, self.learn_mode, False)

    def goal_met(self, _total_reward, _last_reward):
        return False

from head import install_lowlevel
from torch.distributions import Normal
from utils.schedule import LinearSchedule


class HighLevelCtrlTask:
    def __init__(self, dock, prefix, fm, do_sampling=False):
        self.ready = True
        self.ll_ctrl = LowLevelCtrlTask(dock, prefix)
        self.ll_env, self.ll_task = install_lowlevel(self.ll_ctrl, fm, do_sampling)

        self.lowlevel = None
        
        self.ls = LinearSchedule(.1, .8, config.HRL_HINDSIGHTACTION_HORIZON)
        self.ctrl = None

        #DEBUG
        self.total = 0
        self.probed = 0

        self.goals = None
        self.goods = None

    def get_goals(self):
        return self.goals

    def _state(self, 
            einfo, base_states, rewards, actions, goods, pi,
            learn_mode=False, reset=False, seed=None):

        states = einfo.states.clone()

        self.goods = [(g0 or g1) for g0, g1 in zip(self.goods, goods)] if goods is not None else [False for _ in range(len(states))]
        
        info = Info(
                states,
                rewards,
                actions,
# custom rewards shape                    
                (einfo.rewards + config.REWARD_DELTA) * config.REWARD_MAGNITUDE,
                einfo.dones,
                self.goals,
                goods,
                pi,
                )

        self.info = info
        return info

    def _finish_ep(self):
        if self.lowlevel is None:
            return
        # allow to learn only from eps where we moved
        #print("\n ep selection --->", sum(self.goods))
        self.ll_ctrl.set_goods(self.goods)
        for _ in self.lowlevel:
            pass#print("DO FINISH")

    def reset(self, agent, seed, learn_mode):        
        self.learn_mode = learn_mode
        
#        timebudget.report(reset=True)
        
        self.total = 0
        self.probed = 0

        self._finish_ep()

        self.lowlevel = self.ll_env.step(
            self.ll_task, seed, config.HRL_STEP_COUNT) if learn_mode else self.ll_env.evaluate(
            self.ll_task, config.HRL_STEP_COUNT)

        self.goals = self.ll_ctrl.internal_reset(agent, seed, learn_mode)
        
        return self._state(
            self.ll_ctrl.get_info(), 
            None, None, None, None, None,
            learn_mode=learn_mode, reset=True, seed=seed)
    
    def step(self, pi):
        a = pi[:, :pi.shape[1]//3].clone()

        self.ll_ctrl.set_goals(a.clone())
        base_states = self.ll_ctrl.get_info().states.clone()

        (log_prob, _, _, _, _, ll_actions, _, good), acu_reward = next(self.lowlevel)
        
        next_states = self.ll_ctrl.get_info().states.clone()
        
        goods = select_exp(base_states, next_states)
        actions, reactions = self.proximaly_close_actions(a, pi, base_states, next_states, goods)
#        actions = self.proximaly_close_actions(a, pi, base_states, next_states, goods)

        self.einfo = self._state(
            self.ll_ctrl.get_info(), 
            base_states,
            acu_reward, actions, goods, 
            pi,
            self.learn_mode, reset=False)

        assert action.shape[-1] >= ll_actions.shape[-1]*2
        self.einfo.pi[:, actions.shape[-1]:actions.shape[-1]+ll_actions.shape[-1]*2] = torch.cat([ # TODO[ : KICK OFF
            log_prob,#torch.ones_like(log_prob ),
            ll_actions], 1)#torch.ones_like(actionsZ) ], 1)
        self.einfo.pi[:, actions.shape[-1]*2:] = reactions

        return self.einfo
    
    def proximaly_close_actions(self, a, actions, base_states, next_states, goods):
        self.total += len(a)
        
#        if not self.learn_mode:
#            return a
        
        pi = Normal(actions[:, a.shape[1]: a.shape[1]*2], actions[:, a.shape[1]*2:])
        og = self.ll_ctrl.optimal_goals(base_states, next_states)

#        return a, og

        mean_a = pi.log_prob(a).mean(1) 
        baseline_up = mean_a * (1. + 1. - self.ls.get_ls())
        baseline_down = mean_a * (1. - (1. - self.ls.get_ls()))
        
        mean_og = pi.log_prob(og).mean(1) 
        idx_up = mean_og < baseline_up 
        idx_down = mean_og > baseline_down
        idx = idx_up == idx_down
        if not sum(idx):
            return a, og
        
        a[idx > 0] = og[idx > 0]
        self.probed += sum(idx)

        if sum(idx) * 2 > len(og) and sum(goods):
            return a, og.clone()
        
        self.ls()    
        return a, og.clone()


