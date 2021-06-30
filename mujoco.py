import gym
import numpy as np

class Reacher:
    def __init__(self, render):
        self.env = gym.make("FetchReach-v1")
        self.do_render = render
        self.action_space = self.env.action_space

    def render(self):
        if self.do_render:
            self.env.render()

    def reset(self):
        data = self.env.reset()
        obs = data['observation']
        grip_pos = obs[0:3]
        gripper_state = obs[3:5]
        grip_velp = obs[5:8]
        gripper_vel = obs[8:10]
        assert 10 == len(obs)

        self.prev_pos = grip_pos

        obs = np.hstack([
            grip_pos, self.prev_pos.copy(), 
            gripper_state, grip_velp, gripper_vel
            ])
        data["observation"] = obs
        return data

    def seed(self, s):
        self.env.seed(s)

    def step(self, actions):
        data = self.env.step(actions)
        obs = data[0]['observation']
        grip_pos = obs[0:3]
        gripper_state = obs[3:5]
        grip_velp = obs[5:8]
        gripper_vel = obs[8:10]
        obs = np.hstack([
            grip_pos, self.prev_pos.copy(), 
            gripper_state, grip_velp, gripper_vel
            ])

        self.prev_pos = grip_pos

        data[0]["observation"] = obs
        return data

class Pusher:
    def __init__(self, render):
        self.env = gym.make("FetchPush-v1")
        self.do_render = render
        self.action_space = self.env.action_space

    def render(self):
        if self.do_render:
            self.env.render()

    def reset(self):
        data = self.env.reset()
        obs = data['observation']
        grip_pos = obs[0:3]
        object_pos = obs[3:6]
        object_rel_pos = obs[6:9]
        gripper_state = obs[9:11] # 2
        object_rot = obs[11:14]
        object_velp = obs[14:17]
        object_velr = obs[17:20]
        grip_velp = obs[20:23]
        gripper_vel = obs[23:25] # 2

        assert 25 == len(obs)

        self.prev_pos = grip_pos
        self.obj_prev = object_pos.copy()

        obs = np.hstack([
            grip_pos, self.prev_pos.copy(), 
            gripper_state, grip_velp, gripper_vel, 

            object_rel_pos, object_rot, object_velp, object_velr,
            object_pos, self.obj_prev.copy()
            ])
        data["observation"] = obs
        return data

    def seed(self, s):
        self.env.seed(s)

    def step(self, actions):
        data = self.env.step(actions)
        obs = data[0]['observation']
        grip_pos = obs[0:3]
        object_pos = obs[3:6]
        object_rel_pos = obs[6:9]
        gripper_state = obs[9:11]
        object_rot = obs[11:14]
        object_velp = obs[14:17]
        object_velr = obs[17:20]
        grip_velp = obs[20:23]
        gripper_vel = obs[23:25]

        obs = np.hstack([
            grip_pos, self.prev_pos.copy(), grip_velp, 
            gripper_state, gripper_vel, 

            object_rel_pos, object_rot, object_velp, object_velr,
            object_pos, self.obj_prev.copy()
            ])

        self.prev_pos = grip_pos
        self.obj_prev = object_pos.copy()

        data[0]["observation"] = obs
        return data

