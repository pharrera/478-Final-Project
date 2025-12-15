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
        
        # Robust Feature Layer with Batch Norm and Dropout
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2), # Tuned to 0.3 for stability
            
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
        self.memory = deque(maxlen=20000) # Buffer size from paper
        
        # OPTIMIZATION: Paper uses Gamma 0.99
        self.gamma = 0.99    
        self.epsilon = 1.0   
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.95 # Adjusted for 100 episodes
        
        self.learning_rate = 0.0001 
        self.batch_size = 64
        self.tau = 0.01 # Soft Update Factor
        
        self.device = device
        
        # 1. Main Model (Trainable)
        self.model = DuelingDQN(state_dim, action_dim).to(self.device)
        
        # 2. Target Model (Stable) - CRITICAL FOR CONVERGENCE
        self.target_model = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval() # Never train directly
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        self.criterion = nn.MSELoss()

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        self.model.eval() # Batch Norm requires eval mode for single samples
        with torch.no_grad():
            q_values = self.model(state_tensor)
        self.model.train()
        
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def soft_update(self):
        """
        Soft update: target_weights = τ * local_weights + (1 - τ) * target_weights
        """
        for target_param, local_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

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

        self.model.train()
        
        # Current Q from Main Model
        current_q = self.model(states).gather(1, actions).squeeze(1)
        
        # Next Q from TARGET Model (Stable)
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
        
        expected_q = rewards + (self.gamma * next_q * (1 - dones))
        
        loss = self.criterion(current_q, expected_q)
        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        self.optimizer.step()
        # Perform Soft Update
        self.soft_update()

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay