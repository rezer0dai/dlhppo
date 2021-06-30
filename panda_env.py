import gym
from gym import error, spaces, utils
from gym.utils import seeding

import os
import pybullet as p
import pybullet_data
import math
import numpy as np
import random


MAX_EPISODE_LEN = 20*100

class PandaEnv(gym.Env):
#    metadata = {'render.modes': ['human']}

    def __init__(self, mode):
        self.step_counter = 0
        if "WORKER" in mode:
            p.connect(p.DIRECT)
        else:
            p.connect(p.GUI)
            p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0.55,-0.35,0.2])
        self.action_space = spaces.Box(np.array([-1]*4), np.array([1]*4))
        self.observation_space = spaces.Box(np.array([-1]*5), np.array([1]*5))

    def step(self, action):
        n_rep = 10
        state_object0, _ = p.getBasePositionAndOrientation(self.objectUid)
        state_robot0 = p.getLinkState(self.pandaUid, 11)[0]
        state_fingers0 = (p.getJointState(self.pandaUid,9)[0], p.getJointState(self.pandaUid, 10)[0])
        for _ in range(n_rep):
            p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
            orientation = p.getQuaternionFromEuler([0.,-math.pi,math.pi/2.])
            dv = 0.005 * n_rep
            dx = action[0] * dv
            dy = action[1] * dv
            dz = action[2] * dv
            fingers = action[3]

            currentPose = p.getLinkState(self.pandaUid, 11)
            currentPosition = currentPose[0]
            newPosition = [currentPosition[0] + dx,
                           currentPosition[1] + dy,
                           currentPosition[2] + dz]
            jointPoses = p.calculateInverseKinematics(self.pandaUid,11,newPosition, orientation)[0:7]

            p.setJointMotorControlArray(self.pandaUid, list(range(7))+[9,10], p.POSITION_CONTROL, list(jointPoses)+2*[fingers])

            p.stepSimulation()

            state_object, _ = p.getBasePositionAndOrientation(self.objectUid)
            state_robot = p.getLinkState(self.pandaUid, 11)[0]
            state_fingers = (p.getJointState(self.pandaUid,9)[0], p.getJointState(self.pandaUid, 10)[0])

        self.step_counter += 1

#        info = {'object_position': state_object}
        self.observation = state_robot + state_object0 + state_robot0 + state_fingers + state_fingers0

        obs = np.array(self.observation).astype(np.float32)
        achieved_goal = obs.copy()[:3]#np.array(state_object).astype(np.float32)
        desired_goal = np.array(state_object).astype(np.float32)#achieved_goal.copy()
#        desired_goal[2] = .45

        info = {
            'observation': np.concatenate([obs.copy()[:3], achieved_goal.copy(), obs.copy()[3:]]),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': desired_goal.copy(),

            'object_position': state_object,
        }

        done = False
        if np.linalg.norm(achieved_goal - desired_goal) < .05:#state_object[2]>=0.45:
            reward = 0.#1
        else:
            reward = -1.#0
        if self.step_counter > MAX_EPISODE_LEN:
            reward = 0
            done = True

        return info, reward, done, info

    def reset(self):
        self.step_counter = 0
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0) # we will enable rendering after we loaded everything
        urdfRootPath=pybullet_data.getDataPath()
        p.setGravity(0,0,-10)

        planeUid = p.loadURDF(os.path.join(urdfRootPath,"plane.urdf"), basePosition=[0,0,-0.65])

        rest_poses = [0,-0.215,0,-2.57,0,2.356,2.356,0.08,0.08]
        self.pandaUid = p.loadURDF(os.path.join(urdfRootPath, "franka_panda/panda.urdf"),useFixedBase=True)
        for i in range(7):
            p.resetJointState(self.pandaUid,i, rest_poses[i])
        p.resetJointState(self.pandaUid, 9, 0.08)
        p.resetJointState(self.pandaUid,10, 0.08)
        tableUid = p.loadURDF(os.path.join(urdfRootPath, "table/table.urdf"),basePosition=[0.5,0,-0.65])

        trayUid = p.loadURDF(os.path.join(urdfRootPath, "tray/traybox.urdf"),basePosition=[0.65,0,0])

        state_object= [random.uniform(0.5,0.8),random.uniform(-0.2,0.2),0.05]
        self.objectUid = p.loadURDF(os.path.join(urdfRootPath, "random_urdfs/000/000.urdf"), basePosition=state_object)
        state_robot = p.getLinkState(self.pandaUid, 11)[0]

        while np.linalg.norm(np.asarray(state_object) - np.asarray(state_robot)) <= .05:
            state_object= [random.uniform(0.5,0.8),random.uniform(-0.2,0.2),0.05]
            self.objectUid = p.loadURDF(os.path.join(urdfRootPath, "random_urdfs/000/000.urdf"), basePosition=state_object)
            state_robot = p.getLinkState(self.pandaUid, 11)[0]

        state_fingers = ( p.getJointState(self.pandaUid,9)[0], p.getJointState(self.pandaUid, 10)[0] )
        self.observation = state_robot + state_fingers
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)

        self.observation = list(state_robot) + state_object + list(state_robot) + list(state_fingers) + list(state_fingers)

        obs = np.array(self.observation).astype(np.float32)
        achieved_goal = obs.copy()[:3]#np.array(state_object).astype(np.float32)
        desired_goal = np.array(state_object).astype(np.float32)#achieved_goal.copy()
#        desired_goal[2] = .45

        info = {
            'observation': np.concatenate([obs.copy()[:3], achieved_goal.copy(), obs.copy()[3:]]),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': desired_goal.copy(),

            'object_position': state_object,
        }

        return info#np.array(self.observation).astype(np.float32)

    def render(self, mode='human'):
        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.7,0,0.05],
                                                            distance=.7,
                                                            yaw=90,
                                                            pitch=-70,
                                                            roll=0,
                                                            upAxisIndex=2)
        proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                     aspect=float(960) /720,
                                                     nearVal=0.1,
                                                     farVal=100.0)
        (_, _, px, _, _) = p.getCameraImage(width=960,
                                              height=720,
                                              viewMatrix=view_matrix,
                                              projectionMatrix=proj_matrix,
                                              renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (720,960, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def _get_state(self):
        return self.observation

    def close(self):
        p.disconnect()
