import pandas as pd
import numpy as np
import torch  # <--- This was the missing line causing the NameError
import os
from environment import NetworkEnv
from agent import DQNAgent

# Create dummy data if file doesn't exist (For the "Bootstrap" requirement)
DATA_PATH = "data/dummy_data.csv"

# Ensure directories exist
if not os.path.exists("data"):
    os.makedirs("data")

if not os.path.exists("artifacts"):
    os.makedirs("artifacts")
    
if not os.path.exists(DATA_PATH):
    print("Creating dummy dataset for Alpha demonstration...")
    # Create 1000 rows of random data
    df = pd.DataFrame(np.random.rand(1000, 10), columns=[f"feat_{i}" for i in range(10)])
    # Add labels (0 or 1)
    df["label"] = np.random.randint(0, 2, 1000)
    df.to_csv(DATA_PATH, index=False)

def main():
    print("Initializing NeuroGuard Environment...")
    env = NetworkEnv(DATA_PATH, max_steps=500)
    agent = DQNAgent(state_dim=env.n_features, action_dim=env.action_space.n)
    
    episodes = 5
    print(f"Starting Training for {episodes} episodes...")

    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            # Train the agent (Replay experience)
            agent.replay()
            
        print(f"Episode: {e+1}/{episodes}, Score: {total_reward}, Epsilon: {agent.epsilon:.2f}")

    print("Training Complete. Saving Model...")
    # Save the model to the artifacts folder
    torch.save(agent.model.state_dict(), "artifacts/dqn_model.pth")
    print("Model saved to artifacts/dqn_model.pth")

if __name__ == "__main__":
    main()