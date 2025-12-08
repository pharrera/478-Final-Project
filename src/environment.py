import gym
import numpy as np
import pandas as pd
from gym import spaces

class NetworkEnv(gym.Env):
    """
    Gym environment for NSL-KDD Intrusion Detection.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, data, labels, max_steps=None):
        super(NetworkEnv, self).__init__()
        
        self.data = data
        self.labels = labels
        self.n_samples = len(data)
        self.current_step = 0
        self.max_steps = max_steps if max_steps else self.n_samples

        # Action Space: 0 = Pass, 1 = Drop
        self.action_space = spaces.Discrete(2)

        # Observation Space: Number of features (columns)
        self.n_features = data.shape[1]
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.n_features,), dtype=np.float32
        )

    def reset(self):
        self.current_step = 0
        return self._next_observation()

    def _next_observation(self):
        obs = self.data[self.current_step]
        return obs.astype(np.float32)

    def step(self, action):
        # 1. Get Ground Truth (0=Normal, 1=Attack)
        actual_label = self.labels[self.current_step]
        
        # 2. Calculate Reward
        reward = 0
        if action == actual_label:
            reward = 1  # Correct
        else:
            # Penalize errors (Weighted)
            if action == 1 and actual_label == 0:
                reward = -5 # False Positive (Bad!)
            elif action == 0 and actual_label == 1:
                reward = -5 # False Negative (Missed attack)

        # 3. Next step
        self.current_step += 1
        done = self.current_step >= self.max_steps - 1
        
        # 4. Next state
        obs = self._next_observation() if not done else np.zeros(self.n_features)
        
        return obs, reward, done, {}