import torch
import numpy as np
import random

N_REWARDS = 1

def extract_goal(state):
    return state[-4-3:-1-3]

# https://github.com/Unity-Technologies/ml-agents/blob/master/UnitySDK/Assets/ML-Agents/Examples/Reacher/Scripts/ReacherAgent.cs
def transform(obs):
    return np.concatenate([
        obs[3+4+3+3:-4-3], #pendulumB position with rest of info
        obs[:3+4+3+3], # pendulumA info
        # skip goal position
        obs[-1-3:] #speed + hand position
        ], 0)

def extract4plot(ss):
    s = ss.copy()
    # skip w of Quaternion.Euler
    s = np.concatenate([s[:3+3], s[3+3+1:]], 0)
    # still we have for nice plot reversed angular vs dimension velocities
    s = np.concatenate([s[:3+3], s[3+3+3:3+3+3+3], s[3+3:3+3+3], s[3+3+3+3:]], 0)
#    assert len(ss) - 1 == len(s), "wut -> {} -> {} ({} ehmm {})".format(len(ss), len(s), ss.shape, s.shape)
    s = np.concatenate([s[:12], s[29:]])
    return s

def goal_distance(goal_a, goal_b):
    return torch.norm(goal_a[:3] - goal_b[:3])

#  from gahil import fun_reward
from unityagents import UnityEnvironment

from tasks import Nice_plot

from utils.gahil import GAHIL
GAN_REWARD = GAHIL(action_size=4)

#@dataclass
class Info:
    def __init__(self, states, rews, actions, custom_rewards, dones, goals, dist):
        self.states = states
        self.rewards = rews
        self.actions = actions
        self.custom_rewards = custom_rewards
        self.dones = dones
        self.goals = goals

        self.dist = dist

class StaticReacherProxy:
    def __init__(self):
        self.ENV = UnityEnvironment(file_name='./reach/Reacher.x86_64')
        self.BRAIN_NAME = self.ENV.brain_names[0]
        self.learn_mode = False
        self.seeds = []

    def _assess_static_goals(self, einfo, states, reset, learn_mode, seed):
        if not reset: # static reacher means goals are not moving
            return
        self.goals = torch.from_numpy(extract_goal(einfo.vector_observations.T).T)

        if not learn_mode: # during training however we want to move goals to avoid GA(H)IL to learn goal by distance only!!
            return

        if not len(self.seeds) or seed != self.seeds[-1]:
            self.dist = np.random.normal(
                    0,
                    1. * np.ceil(goal_distance(self.goals[0], states[0]) / 2.),
                    self.goals.shape)

        self.seeds.append(seed)
        if len(self.seeds) > 1 and 0 == len(self.seeds) % 2 and seed == self.seeds[-2]:
            return # reuse same goals

        self.goals = self.goals + torch.from_numpy(self.dist)

    def _state(self, einfo, actions, learn_mode=False, reset=False, seed=None):
        global GAN_REWARD
        states = torch.from_numpy(transform(einfo.vector_observations.T).T)

        self._assess_static_goals(einfo, states, reset, learn_mode, seed)

        info = Info(
                states,
                torch.tensor(einfo.rewards).view(-1, 1),
                actions[:, :4],
                torch.zeros(len(states), 1) if not len(self.stats) else GAN_REWARD.register_other_with_reward(
                    self.stats[-1][:, :states.shape[1]].numpy(), states.numpy(), self.goals.numpy(), actions.numpy()
                    ),
                torch.tensor([1. if e else 0. for e in einfo.local_done]).view(len(einfo.local_done), -1).double(),
                self.goals,

                torch.stack([ goal_distance(s, g) for s, g in zip(states, self.goals) ]).view(-1, 1),
                )

        self.stats.append( torch.cat([
            info.states, info.rewards, info.actions, info.custom_rewards, info.dones, info.dist, info.goals], 1) )

        self.info = info

        return info

    def _plot(self, agent):
        stats = np.asarray([s.numpy() for s in self.stats])
        best = max([sum(s) for s in stats.T[29]])
        bests = [i for i, s in enumerate(stats.T[29]) if sum(s) == best]
        longest = np.argmin([sum(s) for s in np.asarray(stats).T[-1, bests]])
        top = bests[longest]

        stats = extract4plot(stats.T).T[::10]
        goals = stats[:-1, top, -3:]

        values = torch.stack([qa[top] for qa in agent.brain.qa_vs]).mean(1).numpy()
        future = torch.stack([qa[top] for qa in agent.brain.qa_fs]).mean(1).numpy()

        for i in range(3):
            trajectory = stats[:-1, top, i]
            emax, emin = trajectory.max(), trajectory.min()

            delta = (emax-emin) if i < 2 else (emin-emax)

#                assert stats[0, top, 2*(3 * 4) + 1 + i] == stats[-2, top, 2*(3 * 4) + 1 + i]

            stats[:-1, top, i] = 2. * (trajectory - emin - delta / 2.) / 3.
            goals[:, i] = 2. * (goals[:, i] - emin - delta / 2.) / 3.

        Nice_plot.plot_proxy(stats[:-1, top], goals, values, future)

    def reset(self, agent, seed, learn_mode):
        # debug
        if learn_mode and not self.learn_mode:
            self._plot(agent)

        self.stats = []
        self.learn_mode = learn_mode

        einfo = self.ENV.reset()[self.BRAIN_NAME] if learn_mode else self.ENV.reset(config={"goal_speed":0})[self.BRAIN_NAME]
#        einfo = self.ENV.reset(config={"goal_size":.9, "goal_speed":0})[self.BRAIN_NAME]

        return self._state(einfo, torch.zeros(len(einfo.vector_observations), 4*3), learn_mode, True, seed)

    def step(self, actions):
        einfo = self.ENV.step(actions[:, :actions.shape[1]//3].cpu().numpy())[self.BRAIN_NAME]
        return self._state(einfo, actions)
