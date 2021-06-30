import gym
import panda_gym
import numpy as np

class Reacher:
    def __init__(self, render):
        self.env = gym.make("PandaReach-v1", render=render)
        self.do_render = render
        self.action_space = self.env.action_space

    def render(self):
        if self.do_render:
            self.env.render()

    def reset(self):
        data = self.env.reset()
        obs = data['observation']
        ee_position = obs[0:3]
        ee_velocity = obs[3:6]
        assert 6 == len(obs)

        self.prev_pos = ee_position

        obs = np.hstack([
            ee_position, self.prev_pos.copy(), 
            ee_velocity,
            ])
        data["observation"] = obs
        return data

    def seed(self, s):
        self.env.seed(s)

    def step(self, actions):
        data = self.env.step(actions)
        obs = data[0]['observation']
        ee_position = obs[0:3]
        ee_velocity = obs[3:6]

        obs = np.hstack([
            ee_position, self.prev_pos.copy(), 
            ee_velocity,
            ])

        self.prev_pos = ee_position

        data[0]["observation"] = obs
        return data

class Pusher:
    def __init__(self, render):
        self.env = gym.make("PandaPush-v1", render=render)
        self.do_render = render
        self.action_space = self.env.action_space

    def render(self):
        if self.do_render:
            self.env.render()

    def reset(self):
        data = self.env.reset()
        while 0 == self.env.step(self.env.action_space.sample())[1]:
            data = self.env.reset() # skip solved envs
        obs = data['observation']
        ee_position = obs[0:3]
        ee_velocity = obs[3:6]

        object_position = obs[6:9]
        object_rotation = obs[9:12]
        object_velocity = obs[12:15]
        object_angular_velocity = obs[15:]

        assert 3 == len(object_angular_velocity)

        self.prev_pos = ee_position
        self.prev_obj = ee_position

        obs = np.hstack([
            ee_position, self.prev_pos.copy(), 
            ee_velocity,

            object_rotation, object_velocity, object_angular_velocity, 
            object_position - ee_position, 
            object_position, self.prev_obj.copy(),
            ])
        data["observation"] = obs
        return data

    def seed(self, s):
        self.env.seed(s)

    def step(self, actions):
        data = self.env.step(actions)
        obs = data[0]['observation']
        ee_position = obs[0:3]
        ee_velocity = obs[3:6]

        object_position = obs[6:9]
        object_rotation = obs[9:12]
        object_velocity = obs[12:15]
        object_angular_velocity = obs[15:]

        obs = np.hstack([
            ee_position, self.prev_pos.copy(), 
            ee_velocity,

            object_rotation, object_velocity, object_angular_velocity, 
            object_position - ee_position, 
            object_position, self.prev_obj.copy(),
            ])

        self.prev_pos = ee_position
        self.prev_obj = ee_position

        data[0]["observation"] = obs
        return data

