import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# 1. M1 Mac / CUDA Optimized Device Selection
# We define this globally so the class can pick it up
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Optimization: Using Apple Silicon GPU (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # Discount factor
        self.epsilon = 1.0   # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32
        
        # Assign the globally detected device
        self.device = device
        
        # Move model to the correct device (GPU/MPS/CPU)
        self.model = DQN(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def act(self, state):
        # Epsilon-greedy action selection
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        
        # Convert state to tensor on the correct device
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Disable gradient calculation for inference (Save Memory/Speed)
        with torch.no_grad():
            q_values = self.model(state_tensor)
            
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        
        # --- VECTORIZED BATCH PROCESSING (Huge Speedup for M1) ---
        # Instead of a for-loop, we process all 32 samples at once.
        
        # 1. Unzip the batch into separate arrays
        states, actions, rewards, next_states, dones = zip(*minibatch)
        
        # 2. Convert to Tensors and move to Device (MPS/CUDA)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device) # Shape: [32, 1]
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # 3. Current Q Values
        # Get Q-values for all actions, then gather the ones we actually took
        current_q = self.model(states).gather(1, actions).squeeze(1)
        
        # 4. Next Q Values (Target)
        # Get max Q-value for next states (detach prevents gradient flow to target)
        next_q = self.model(next_states).max(1)[0].detach()
        
        # 5. Compute Target using Bellman Equation
        expected_q = rewards + (self.gamma * next_q * (1 - dones))
        
        # 6. Compute Loss & Backpropagate
        loss = self.criterion(current_q, expected_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay Epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay