import subprocess

def exe(cmd):
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    return process.communicate()[0].decode()

from defs import GymPacket
import socket, pickle, time

import torch
import numpy as np

class GymProxy:
    def __init__(self, env_name, dock, port, prefix, ind):
        self.port = port
        self.env_name = env_name

        name = "%s_gym_dock_%i"%(prefix, ind)
# run environment
        subprocess.Popen([
            "docker", "run", "--rm", 
            "--name", name, "-p", "5001", 
            dock])
# get environment connection details
        self.addr = ""
        while "172" not in self.addr:
            time.sleep(1.)
            self.addr = self._get_ip(name)
            print("..", self.addr)
        print(name, "--->", self.addr)

    def _get_ip(self, name):
        cmd = 'docker inspect -f "{{ .NetworkSettings.Networks.bridge.IPAddress }}" %s'%name
        ip_addr = exe(cmd)
        return ip_addr[:-1]
# bit overkill everytime new sock, but perf seems fast anyway
    def _do(self, action, data):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.addr, self.port))
            packet = GymPacket(action, data)
            data = pickle.dumps(packet, protocol=3)
            s.send(data)
            data = s.recv(2048)
            packet = pickle.loads(data)
            return packet

    def make(self):
        return self._do("create", self.env_name)

    def reset(self, seed):
        return self._do("reset", seed)

    def step(self, actions):
        return self._do("step", actions)

class GymGroup:
    def __init__(self, env_name, dock, n_env, prefix):
        self.gyms = [
                GymProxy(env_name, dock, 5001, prefix, i
                    ) for i in range(n_env)]

        for gym in self.gyms:
            gym.make()

    def reset(self, seeds):
        obs = np.concatenate([
            self._process(
                gym.reset(int(seed))
                ) for gym, seed in zip(self.gyms, seeds) ], 0)
        return self._decouple(obs)

    def step(self, actions):
        a_s = len(actions) // len(self.gyms)
        obs = np.concatenate([
            self._process(
                gym.step(actions[i])
                ) for i, gym in enumerate(self.gyms) ], 0)
        return self._decouple(obs)

    def _process(self, packet):
        assert "state" == packet.action
        obs, reward, done, info = packet.data
        return np.concatenate([
            obs["achieved_goal"], obs["observation"], obs["desired_goal"],
            [reward], [done]]).reshape(1, -1)

    def _decouple(self, obs):
        return (
                torch.from_numpy(obs[:, :-3-2]), 
                torch.from_numpy(obs[:, -3-2:-2]),
                obs[:, -2], obs[:, -1])

import gym

class GymRender:
    def __init__(self, env_name, n_env):
        self.n_env = n_env
        self.env = gym.make(env_name)

    def reset(self, seed):
        self.env.seed(int(seed[0]))
        obs = self.env.reset()
        state = torch.from_numpy(
                    np.concatenate([obs['achieved_goal'], obs['observation']])).expand_as(
                torch.ones(self.n_env, len(obs['observation']) + len(obs['achieved_goal'])))
        goals = torch.from_numpy(obs['desired_goal']).expand_as(
                torch.ones(self.n_env, len(obs['desired_goal'])))
        return (
                state, goals, 
                np.zeros([len(state), 1]), #rewards
                np.zeros([len(state), 1])) #dones

    def step(self, actions):
        obs, r, d, i = self.env.step(actions[0])

        self.env.render() # whole purpose...

        state = torch.from_numpy(#obs['observation']).expand_as(
                    np.concatenate([obs['achieved_goal'], obs['observation']])).expand_as(
                torch.ones(self.n_env, len(obs['observation']) + len(obs['achieved_goal'])))
        goals = torch.from_numpy(obs['desired_goal']).expand_as(
                torch.ones(self.n_env, len(obs['desired_goal'])))
        return (
                state, goals, 
                np.zeros([len(state), 1]) + r, #rewards
                np.zeros([len(state), 1]) + d) #dones
