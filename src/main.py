import pandas as pd
import numpy as np
import torch
import os
import sys
import requests
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from environment import NetworkEnv
from agent import DQNAgent

# Config
DATA_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B_20Percent.txt"
DATA_PATH = "data/nsl_kdd_train.csv"
ARTIFACTS_DIR = "artifacts/release"
MODEL_PATH = f"{ARTIFACTS_DIR}/dqn_model.pth"

# Ensure directories exist
os.makedirs("data", exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def load_real_data():
    if not os.path.exists(DATA_PATH):
        print(f"Downloading NSL-KDD dataset from {DATA_URL}...")
        response = requests.get(DATA_URL)
        with open(DATA_PATH, 'wb') as f:
            f.write(response.content)
            
    cols = ["duration","protocol_type","service","flag","src_bytes","dst_bytes",
            "land","wrong_fragment","urgent","hot","num_failed_logins",
            "logged_in","num_compromised","root_shell","su_attempted",
            "num_root","num_file_creations","num_shells","num_access_files",
            "num_outbound_cmds","is_host_login","is_guest_login","count",
            "srv_count","serror_rate","srv_serror_rate","rerror_rate",
            "srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate",
            "dst_host_count","dst_host_srv_count","dst_host_same_srv_rate",
            "dst_host_diff_srv_rate","dst_host_same_src_port_rate",
            "dst_host_srv_diff_host_rate","dst_host_serror_rate",
            "dst_host_srv_serror_rate","dst_host_rerror_rate",
            "dst_host_srv_rerror_rate","label","difficulty"]
    
    df = pd.read_csv(DATA_PATH, names=cols)
    y = df['label'].apply(lambda x: 0 if x == 'normal' else 1).values
    X_raw = df.drop(['label', 'difficulty'], axis=1)
    
    le = LabelEncoder()
    for col in X_raw.select_dtypes(include=['object']).columns:
        X_raw[col] = le.fit_transform(X_raw[col])
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X_raw)
    
    return X, y

def train_mode():
    features, labels = load_real_data()
    
    # Determine Mode
    if "--full" in sys.argv:
        print(">>> RUNNING FULL DATASET TRAINING (25,000+ Rows) <<<")
        limit = None
        episodes = 5
        log_interval = 2000 # Print every 2000 steps
    else:
        print(">>> RUNNING FAST MODE (1,000 Rows) <<<")
        limit = 1000
        episodes = 3
        log_interval = 200 # Print more often in fast mode

    X_train = features[:limit] if limit else features
    y_train = labels[:limit] if limit else labels
    dataset_size = len(X_train)
    
    env = NetworkEnv(X_train, y_train)
    agent = DQNAgent(state_dim=env.n_features, action_dim=env.action_space.n)
    
    history = [] 
    print(f"--- STARTING TRAINING ({dataset_size} Flows, {episodes} Episodes) ---")

    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        step_count = 0
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.replay()
            
            step_count += 1
            
            # Progress Logging
            if step_count % log_interval == 0:
                print(f"[Episode {e+1}] Step {step_count}/{dataset_size} | Reward: {total_reward} | Eps: {agent.epsilon:.2f}")
                sys.stdout.flush() # Ensure docker prints immediately
            
        print(f"--- Episode: {e+1}/{episodes} FINISHED | Total Score: {total_reward} ---")
        history.append({"episode": e + 1, "reward": total_reward})

    # Save
    torch.save(agent.model.state_dict(), MODEL_PATH)
    pd.DataFrame(history).to_csv(f"{ARTIFACTS_DIR}/metrics.csv", index=False)
    
    plt.figure(figsize=(10, 5))
    df_h = pd.DataFrame(history)
    plt.plot(df_h['episode'], df_h['reward'], marker='o', linestyle='-', color='b')
    plt.title(f'NeuroGuard Training Performance ({dataset_size} Samples)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.savefig(f"{ARTIFACTS_DIR}/training_curve.png")
    print("Training Complete. Model and Artifacts Saved.")

def demo_mode():
    print("\n--- STARTING DEMO MODE ---")
    if not os.path.exists(MODEL_PATH):
        print("Error: Model not found. Run 'make up' first.")
        return

    features, labels = load_real_data()
    env = NetworkEnv(features, labels)
    agent = DQNAgent(state_dim=env.n_features, action_dim=env.action_space.n)
    
    agent.model.load_state_dict(torch.load(MODEL_PATH))
    agent.epsilon = 0.0 
    
    print(f"{'ID':<8} | {'Actual':<10} | {'Predicted':<10} | {'Result'}")
    print("-" * 50)
    
    correct = 0
    test_samples = 15 
    for _ in range(test_samples):
        idx = random.randint(0, len(features)-1)
        state = features[idx]
        actual = labels[idx]
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
        with torch.no_grad():
            q_values = agent.model(state_tensor)
            action = torch.argmax(q_values).item()
        
        status = "✅ PASS" if action == actual else "❌ FAIL"
        if action == actual: correct += 1
        
        act_str = "Attack" if actual == 1 else "Normal"
        pred_str = "Block" if action == 1 else "Pass"
        
        print(f"{idx:<8} | {act_str:<10} | {pred_str:<10} | {status}")
        
    print("-" * 50)
    print(f"Final Release Accuracy: {correct}/{test_samples} ({(correct/test_samples)*100:.1f}%)")

if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo_mode()
    elif "--full" in sys.argv:
        train_mode()
    else:
        train_mode()