import gym
import numpy as np
import pandas as pd
from gym import spaces

class NetworkEnv(gym.Env):
    """
    A custom Gym environment for Network Intrusion Detection.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, data_path, max_steps=None):
        super(NetworkEnv, self).__init__()
        
        # Load and preprocess data
        # Assumes the dataset is a CSV with the last column as the label
        self.df = pd.read_csv(data_path)
        
        # Simple preprocessing: Drop non-numeric columns for the Alpha release
        # In Beta, you should use OneHotEncoding for protocol_type, etc.
        self.df = self.df.select_dtypes(include=[np.number])
        
        # Separate features and labels
        # Assuming the label column was encoded or we treat the last column as label
        self.labels = self.df.iloc[:, -1].values
        self.data = self.df.iloc[:, :-1].values
        
        # Normalize data (Min-Max Scaling) for better NN performance
        self.data = (self.data - self.data.min(axis=0)) / (self.data.max(axis=0) - self.data.min(axis=0) + 1e-6)

        self.n_samples = len(self.df)
        self.current_step = 0
        self.max_steps = max_steps if max_steps else self.n_samples

        # Action Space: 0 = Pass (Benign), 1 = Drop (Malicious)
        self.action_space = spaces.Discrete(2)

        # Observation Space: The feature vector size
        self.n_features = self.data.shape[1]
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.n_features,), dtype=np.float32
        )

    def reset(self):
        """
        Resets the environment to the beginning of the dataset.
        """
        self.current_step = 0
        return self._next_observation()

    def _next_observation(self):
        obs = self.data[self.current_step]
        return obs.astype(np.float32)

    def step(self, action):
        """
        Executes one step (one flow classification).
        """
        # 1. Get Ground Truth
        # For NSL-KDD, labels might be 0/1 or specific attack names.
        # Here we assume the loader handles binary 0 (Normal) vs 1 (Attack)
        actual_label = int(self.labels[self.current_step])
        
        # 2. Calculate Reward
        reward = 0
        if action == actual_label:
            reward = 1  # Correct!
        else:
            # Penalty Logic
            if action == 1 and actual_label == 0:
                # False Positive: Blocked normal traffic
                reward = -5 
            elif action == 0 and actual_label == 1:
                # False Negative: Missed an attack
                reward = -5

        # 3. Move pointer
        self.current_step += 1
        
        # 4. Check if done
        done = self.current_step >= self.max_steps - 1
        
        # 5. Get next state
        obs = self._next_observation()
        
        return obs, reward, done, {}

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Label: {self.labels[self.current_step]}")