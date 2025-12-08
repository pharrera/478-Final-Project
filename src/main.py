import pandas as pd
import numpy as np
import torch
import os
import requests
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from environment import NetworkEnv
from agent import DQNAgent

# Config
DATA_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B_20Percent.txt"
DATA_PATH = "data/nsl_kdd_train.csv"
ARTIFACTS_DIR = "artifacts/release"

# Ensure directories exist
os.makedirs("data", exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def load_real_data():
    """
    Downloads and preprocesses the NSL-KDD dataset.
    Returns: (processed_features, binary_labels)
    """
    # 1. Download if missing
    if not os.path.exists(DATA_PATH):
        print(f"Downloading NSL-KDD dataset from {DATA_URL}...")
        response = requests.get(DATA_URL)
        with open(DATA_PATH, 'wb') as f:
            f.write(response.content)
        print("Download complete.")

    # 2. Load Data (NSL-KDD has no headers, so we define common ones)
    print("Loading and preprocessing data...")
    # Column names based on KDD documentation
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
    
    # 3. Preprocessing
    # Encode 'label': 'normal' -> 0, anything else -> 1
    y = df['label'].apply(lambda x: 0 if x == 'normal' else 1).values
    
    # Drop label and difficulty columns
    X_raw = df.drop(['label', 'difficulty'], axis=1)
    
    # Encode categorical columns (protocol_type, service, flag)
    # Using Label Encoding for simplicity in Alpha/Beta
    le = LabelEncoder()
    for col in X_raw.select_dtypes(include=['object']).columns:
        X_raw[col] = le.fit_transform(X_raw[col])
        
    # Scale all features to 0-1 range (Critical for Neural Networks)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X_raw)
    
    return X, y

def plot_results(history):
    df = pd.DataFrame(history)
    plt.figure(figsize=(10, 5))
    plt.plot(df['episode'], df['reward'], marker='o', label='Total Reward')
    plt.title('NeuroGuard: Training Performance on NSL-KDD')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{ARTIFACTS_DIR}/training_curve.png")
    print(f"Chart saved to {ARTIFACTS_DIR}/training_curve.png")

def main():
    # Load Real Data
    features, labels = load_real_data()
    
    print(f"Data Loaded. Features: {features.shape}, Labels Balance: {np.mean(labels):.2f}")
    print("Initializing NeuroGuard Environment...")
    
    # We take a slice of the data for training speed in this demo (first 2000 rows)
    limit = 2000
    env = NetworkEnv(features[:limit], labels[:limit])
    agent = DQNAgent(state_dim=env.n_features, action_dim=env.action_space.n)
    
    episodes = 5
    history = [] 
    
    print(f"Starting Training for {episodes} episodes on Real Data...")

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
            agent.replay()
            
        print(f"Episode: {e+1}/{episodes}, Score: {total_reward}, Epsilon: {agent.epsilon:.2f}")
        history.append({"episode": e + 1, "reward": total_reward, "epsilon": agent.epsilon})

    # Save Artifacts
    torch.save(agent.model.state_dict(), f"{ARTIFACTS_DIR}/dqn_model.pth")
    pd.DataFrame(history).to_csv(f"{ARTIFACTS_DIR}/metrics.csv", index=False)
    plot_results(history)
    print("Deployment Complete.")

if __name__ == "__main__":
    main()