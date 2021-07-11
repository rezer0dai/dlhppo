from torch.multiprocessing import Queue, Process

import gym
import torch
import numpy as np

import config

def mujoco_make(render):
    import mujoco
    if config.PUSHER:
        return mujoco.Pusher(render)
    return mujoco.Reacher(render)

def ergojr_make(render):
    if config.PUSHER:
        from ergo_pusher_env import ErgoPusherEnv
        return ErgoPusherEnv(not render)
    else:
        from ergo_reacher_env import ErgoReacherEnv
        if 3 == config.ACTION_SIZE:
            return ErgoReacherEnv(not render, goals=1, multi_goal=False, terminates=False, simple=False, gripper=True, backlash=config.BACKLASH)
        if 4 == config.ACTION_SIZE:
            return ErgoReacherEnv(not render, goals=1, multi_goal=False, terminates=False, simple=True, gripper=False, backlash=config.BACKLASH)
        if 6 == config.ACTION_SIZE:
            return ErgoReacherEnv(not render, goals=1, multi_goal=False, terminates=False, simple=False, gripper=False, backlash=config.BACKLASH)
        assert False

def panda_make(render):
    import panda
    if config.PUSHER:
        return panda.Pusher(render)
    return panda.Reacher(render)

def make_env(render, env_name):
    render = render and not config.COLAB
    if config.PANDA:
        return panda_make(render)
    elif config.ERGOJR:
        return ergojr_make(render)
    elif config.MUJOCO:
        return mujoco_make(render)
    assert False

class GymProxy(Process):
    def __init__(self, env_name, dock, port, prefix, ind):
        super().__init__()
        self.query = Queue()
        self.data = Queue()

        self.port = port
        self.env_name = env_name

        self.name = "%s_gym_dock_%i_%s"%(prefix, ind, dock)

    def run(self):
        self.cmd = { 
                "create" : self._create,
                "step" : self._step,
                "reset" : self._reset
                }

        while True: # single thread is fine
            data = self.query.get()
            cmd, data = data
            data = self.cmd[cmd](data)
            self.data.put(data)

    def _create(self, data):
        print(self.name, "create", data)
        self.env = make_env(False, data)
        return self._reset(0)

    def _reset(self, seed):
        if seed: self.env.seed(seed)
        return (self.env.reset(), 0, False, None)

    def _step(self, data):
        return self.env.step(data)

# bit overkill everytime new sock, but perf seems fast anyway
    def _do(self, action, data, asnc):
        self.query.put( (action, data) )
        if asnc: return
        return self.data.get()

    def make(self):
        out = self._do("create", self.env_name, False)
        return out

    def reset(self, seed):
        #print(self.name, "reset /w seed ", seed)
        self.packets = [seed]
        return self._do("reset", seed, False)

    def act(self, actions, asnc=True):
        self.packets.append(actions)
        return self._do("step", actions, asnc)

    def step(self):
#        return self.data.get(timeout=3)#True, 3)
        try: 
            return self.data.get(timeout=3)
        except:
            return None

    def get_history(self):
        return self.packets

class GymGroup:
    def __init__(self, env_name, dock, n_env, prefix):
        self.env_name = env_name
        self.dock = dock
        self.prefix = prefix

        self.gyms = [
                GymProxy(env_name[i%len(env_name)], dock, 5001, prefix, i
                    ) for i in range(n_env)]
        for gym in self.gyms:
            gym.start()
        for gym in self.gyms:
            gym.make()

        self.cache = [None for _ in self.gyms]

    def reset(self, seeds):
        obs = np.concatenate([
            self._process(
                gym.reset(int(seed)), i
                ) for i, (gym, seed) in enumerate(zip(self.gyms, seeds)) ], 0)
        return self._decouple(obs)

    def step(self, actions, active=None):
        a_s = len(actions) // len(self.gyms)

        for i, gym in enumerate(self.gyms):
            if active is not None and not active[i]:
                continue
            gym.act(actions[i])
                
        self.cache = [
            self._process(
                gym.step(), i
                ) if (active is None or active[i]
                    ) else self.cache[i] for i, gym in enumerate(self.gyms) ]

        obs = np.concatenate(self.cache, 0)
        return self._decouple(obs)

    def _hard_boiled(self, i):
        packets = self.gyms[i].get_history()

        self.gyms[i].terminate()

        self.gyms[i] = GymProxy(self.env_name[i%len(self.env_name)], dock, 5001, prefix, i)

        self.gyms[i].start()
        self.gyms[i].make()

        self.gyms[i].reset(int(packets[0]))
        for data in packets[1:-1]:
            self.gyms[i].act(data, asnc=False)
        self.gyms[i].act(packets[-1])
        return self.gyms[i].step()

    def _process(self, data, i):
        while data is None:
            data = self._hard_boiled(i)

        obs, reward, done, info = data
        return np.concatenate([
            obs["achieved_goal"], obs["observation"], obs["desired_goal"],
#obs['achieved_goal'], obs['observation'][:3], obs['observation'], obs["desired_goal"],
#            obs["observation"][:3], obs["observation"], obs["desired_goal"],
            [reward], [done]]).reshape(1, -1)

    def _decouple(self, obs):
        return (
                torch.from_numpy(obs[:, :-3-2]).float(), 
                torch.from_numpy(obs[:, -3-2:-2]).float(),
                obs[:, -2], obs[:, -1])

import gym

class GymRender:
    def __init__(self, env_name, n_env):
        self.n_env = n_env

        self.env = make_env(True, env_name)

    def reset(self, seed):
        self.env.seed(int(seed[0]))
        obs = self.env.reset()
        state = torch.from_numpy(
                    np.concatenate([obs['achieved_goal'], obs['observation']])).float().expand_as(
#np.concatenate([obs['achieved_goal'], obs['observation'][:3], obs['observation']])).expand_as(
#                np.concatenate([obs['observation'][:3], obs['observation']])).expand_as(
                torch.ones(self.n_env, len(obs['observation']) + len(obs['achieved_goal']) * 1))
        goals = torch.from_numpy(obs['desired_goal']).float().expand_as(
                torch.ones(self.n_env, len(obs['desired_goal'])))
        return (
                state, goals, 
                np.zeros([len(state), 1]), #rewards
                np.zeros([len(state), 1])) #dones

    def step(self, actions):
        obs, r, d, i = self.env.step(actions[0])

        if not config.COLAB:
            self.env.render() # whole purpose...

        state = torch.from_numpy(#obs['observation']).expand_as(
                    np.concatenate([obs['achieved_goal'], obs['observation']])).float().expand_as(
#np.concatenate([obs['achieved_goal'], obs['observation'][:3], obs['observation']])).expand_as(
#                    np.concatenate([obs['observation'][:3], obs['observation']])).expand_as(
                torch.ones(self.n_env, len(obs['observation']) + len(obs['achieved_goal']) * 1))
        goals = torch.from_numpy(obs['desired_goal']).float().expand_as(
                torch.ones(self.n_env, len(obs['desired_goal'])))
        return (
                state, goals, 
                np.zeros([len(state), 1]) + r, #rewards
                np.zeros([len(state), 1]) + d) #dones

