import gym
import numpy as np
import pandas as pd
from gym import spaces

class NetworkEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, data, labels, max_steps=None):
        super(NetworkEnv, self).__init__()
        self.data = data
        self.labels = labels
        self.n_samples = len(data)
        self.current_step = 0
        self.max_steps = max_steps if max_steps else self.n_samples

        self.action_space = spaces.Discrete(2)
        self.n_features = data.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_features,), dtype=np.float32
        )

    def reset(self):
        self.current_step = 0
        return self._next_observation()

    def _next_observation(self):
        obs = self.data[self.current_step]
        return obs.astype(np.float32)

    def step(self, action):
        actual_label = int(self.labels[self.current_step])
        reward = 0
        
        if action == actual_label:
            # OPTIMIZATION 1: Higher reward for catching attacks
            if actual_label == 1:
                reward = 2 # Catching an attack is worth more than passing normal traffic
            else:
                reward = 1
        else:
            # OPTIMIZATION 2: "Paranoid" Penalties
            # If we miss an attack (FN), massive penalty.
            if action == 0 and actual_label == 1:
                reward = -50 
            # If we block normal traffic (FP), smaller penalty.
            # We accept some false alarms to ensure we catch the zero-days.
            elif action == 1 and actual_label == 0:
                reward = -2 

        self.current_step += 1
        done = self.current_step >= self.max_steps - 1
        
        obs = self._next_observation() if not done else np.zeros(self.n_features)
        
        return obs, reward, done, {}