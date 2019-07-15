import gym
import cv2
import numpy as np
from collections import deque
from gym import spaces
from baselines.common.atari_wrappers import LazyFrames
from stable_baselines.bench import Monitor
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env.subproc_vec_env import SubprocVecEnv


class RewardScaler(gym.RewardWrapper):
    def reward(self, reward):
        alpha = 0.3
        if reward <= 3:
            return reward * alpha
        else:
            return reward / alpha


class RewardTest(gym.RewardWrapper):
    def reward(self, reward):
        return reward


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.observation_space = spaces.Box(low=0, high=255,
            shape=(self.height, self.width, 1), dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shape = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shape[0], shape[1], shape[2] * k), dtype=np.uint8)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


def make_tutankham_env(num_env, seed=0, start_index=0):
    def make_env(rank):
        def _thunk():
            env = gym.make('Tutankham-v4')
            env.seed(seed + rank)
            env = Monitor(env, filename=None, allow_early_resets=True)
            return wrap_env(env, True)
        return _thunk
    set_global_seeds(seed)
    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])


def make_tutankham_env_test():
    def make_env():
        def _thunk():
            env = gym.make('Tutankham-v4')
            return wrap_env(env, False)
        return _thunk
    return SubprocVecEnv([make_env()])


def wrap_env(env, reward_scaler):
    env = WarpFrame(env)
    if reward_scaler:
        env = RewardScaler(env)
    else:
        env = RewardTest(env)
    env = FrameStack(env, 4)
    return env
