import time
import gym
import numpy as np
from gym import spaces
from tqdm import tqdm

from gym_ergojr.sim.abstract_robot import PusherRobot
from gym_ergojr.sim.objects import Puck

import pybullet as p

GOAL_REACHED_DISTANCE = 0.01
RESTART_EVERY_N_EPISODES = 10#00

class ErgoPusherEnv(gym.Env):

    def __init__(self, headless=False):

        self.goals_done = 0
        self.is_initialized = False

        self.robot = PusherRobot(debug=not headless)
        self.puck = Puck()

        self.episodes = 0  # used for resetting the sim every so often

        self.metadata = {'render.modes': ['human']}

        # observation = 3 joints + 3 velocities + 2 puck position + 2 coordinates for target
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(3 + 3 + 3 + 3,), dtype=np.float32)  #

        # action = 3 joint angles
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(3,), dtype=np.float32)  #

    def seed(self, seed=None):
        return [np.random.seed(seed)]

    def step(self, action):
        self.robot.act(action)
        self.robot.step()

#        reward, done, dist = self._getReward()
        reward, done = -1. * (self.puck.dbo.query() > GOAL_REACHED_DISTANCE), False

        obs = self._get_obs()
        return obs, reward, done, {"distance": 0}#self.dist.query()}

    def _getReward(self):
        done = False

        reward = self.puck.dbo.query()
        distance = reward.copy()

        reward *= -1  # the reward is the inverse distance
        if distance < GOAL_REACHED_DISTANCE:  # this is a bit arbitrary, but works well
            done = True
            reward = 1

        return reward, done, distance

    def reset(self, forced=False):
        self.episodes += 1
        if self.episodes >= RESTART_EVERY_N_EPISODES or forced or not self.is_initialized:
            self.robot.hard_reset()  # this always has to go first
            self.puck.hard_reset()
            self.episodes = 0
            self.is_initialized = True
        else:
            self.puck.reset()

        qpos = self.robot.rest_pos.copy()
        qpos[:3] += np.random.uniform(low=-.1, high=.1, size=3)

        self.robot.set(qpos)
        self.robot.act(qpos[:3])
        self.robot.step()

        self.prev_pos = self.get_tip()
        self.prev_puck = self.get_puck()
        return self._get_obs()

    def get_tip(self):
        return p.getLinkState(self.robot.robot, 6)[0]

    def get_puck(self):
        return p.getLinkState(self.puck.puck, 1)[0]

    def get_goal(self):
        assert 3 == len(self.puck.dbo.goal)
        return self.puck.dbo.goal

    def _get_obs(self):
        arm_pos = self.get_tip()
        puck_pos = self.get_puck()
        obs = np.hstack([ arm_pos, self.prev_pos, self.robot.observe(), np.asarray(puck_pos) - np.asarray(arm_pos), puck_pos, self.prev_puck ])

        obs = {
            'observation': obs.copy(),
            'achieved_goal': puck_pos,

            'desired_goal': self.get_goal(),

            'distance': self.puck.dbo.query(),
        }
        self.prev_pos = arm_pos
        self.prev_puck = puck_pos
        return obs

    def render(self, mode='human', close=False):
        pass

    def close(self):
        self.robot.close()

    def _get_state(self):
        return self.robot.observe()


if __name__ == '__main__':
    import gym
#    import gym_ergojr
    import time

    env = ErgoPusherEnv(headless=False)#gym.make("ErgoPusher-Graphical-v1")
    MODE = "manual"
    r = range(100)

    # env = gym.make("ErgoPusher-Headless-v1")
    # MODE = "timings"
    # r = tqdm(range(10000))

    env.reset()

    timings = []
    ep_count = 0

    start = time.time()

    for _ in r:
        env.reset()
        continue
        while True:
            action = env.action_space.sample()
            obs, rew, done, misc = env.step(action)

            if MODE == "manual":
                print("act {}, obs {}, rew {}, done {}".format(
                    action, obs, rew, done))
                time.sleep(0.01)

            if MODE == "timings":
                ep_count += 1
                if ep_count >= 10000:
                    diff = time.time() - start
                    tqdm.write("avg. fps: {}".format(
                        np.around(10000 / diff, 3)))
                    np.savez("timings.npz", time=np.around(10000 / diff, 3))
                    ep_count = 0
                    start = time.time()

            if done:
                env.reset()
                break
