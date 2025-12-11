import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()
        
        # Robust Feature Layer (prevents overfitting to KDD specifics)
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2), # Moderate dropout
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Value Stream
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Advantage Stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        features = self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        return values + (advantages - advantages.mean())

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=5000) # Increased memory for full dataset
        self.gamma = 0.95    
        self.epsilon = 1.0   
        self.epsilon_min = 0.01
        
        # CRITICAL: Fast decay because we only step ONCE per episode
        self.epsilon_decay = 0.90 
        
        self.learning_rate = 0.0005 
        self.batch_size = 64 # Increased batch size for stability
        
        self.device = device
        self.model = DuelingDQN(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        
        # Fake batch for BatchNorm compatibility (requires >1 sample usually, or eval mode)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        self.model.eval() # Important for BatchNorm/Dropout
        with torch.no_grad():
            q_values = self.model(state_tensor)
        self.model.train()
        
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Train mode enables Dropout/BN
        self.model.train()
        
        current_q = self.model(states).gather(1, actions).squeeze(1)
        next_q = self.model(next_states).max(1)[0].detach()
        expected_q = rewards + (self.gamma * next_q * (1 - dones))
        
        loss = self.criterion(current_q, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Note: No decay here!

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay