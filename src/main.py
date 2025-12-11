import pandas as pd
import numpy as np
import torch
import os
import sys
import requests
import random
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from environment import NetworkEnv
from agent import DQNAgent
from autoencoder import train_autoencoder, AutoEncoder

# Config - OFFICIAL NSL-KDD SPLIT URLs
TRAIN_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"
TEST_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt"

DATA_PATH_TRAIN = "data/KDDTrain+.csv"
DATA_PATH_TEST = "data/KDDTest+.csv"
DATA_PATH_CIC = "data/ciciot2023.csv"

ARTIFACTS_DIR = "artifacts/release"
DQN_PATH = f"{ARTIFACTS_DIR}/dqn_model.pth"
AE_PATH = f"{ARTIFACTS_DIR}/ae_model.pth"

os.makedirs("data", exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def load_data(mode="train", dataset_type="kdd"):
    """
    Universal Loader.
    mode: 'train' or 'test' (Only matters for KDD)
    dataset_type: 'kdd' (Default) or 'ciciot'
    """
    X_raw = None
    y = None

    if dataset_type == "ciciot":
        print(f"Loading CICIoT2023 from {DATA_PATH_CIC}...")
        if not os.path.exists(DATA_PATH_CIC):
            print(f"ERROR: {DATA_PATH_CIC} not found.")
            print("Please download a part-file from Kaggle/UNB, rename to ciciot2023.csv, and place in data/.")
            sys.exit(1)
            
        # CICIoT2023 Load Logic
        df = pd.read_csv(DATA_PATH_CIC)
        y_raw = df['label']
        y = y_raw.apply(lambda x: 0 if x in ['BenignTraffic', 'Benign'] else 1).values
        X_raw = df.drop(['label'], axis=1)
        
    else:
        # NSL-KDD Load Logic (Default)
        path = DATA_PATH_TRAIN if mode == "train" else DATA_PATH_TEST
        url = TRAIN_URL if mode == "train" else TEST_URL
        
        if not os.path.exists(path):
            print(f"Downloading {mode.upper()} set from {url}...")
            response = requests.get(url)
            with open(path, 'wb') as f:
                f.write(response.content)
        
        print(f"Loading {mode.upper()} data from {path}...")
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
                
        df = pd.read_csv(path, names=cols)
        y = df['label'].apply(lambda x: 0 if x == 'normal' else 1).values
        X_raw = df.drop(['label', 'difficulty'], axis=1)

    # Universal Preprocessing (Encoding & Scaling)
    if X_raw is not None:
        le = LabelEncoder()
        for col in X_raw.select_dtypes(include=['object']).columns:
            X_raw[col] = le.fit_transform(X_raw[col].astype(str))
            
        # OPTIMIZATION: Use StandardScaler (Z-Score) per paper specs
        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw)
        return X, y
    else:
        print("Error: Dataset could not be loaded.")
        sys.exit(1)

def train_mode():
    # Detect Dataset Flag
    ds_type = "ciciot" if "--ciciot" in sys.argv else "kdd"
    
    # LOAD TRAINING DATA ONLY
    print(f"Loading Training Data for {ds_type}...")
    X_train, y_train = load_data("train", ds_type)
    
    # Configuration
    if "--full" in sys.argv:
        print(f">>> FULL TRAINING ON {ds_type.upper()} ({len(X_train)} Samples) <<<")
        limit = None
        # OPTIMIZATION: 40 Episodes is enough with aggressive decay
        episodes = 40
        log_interval = 500 # Faster updates
    else:
        print(f">>> FAST MODE ON {ds_type.upper()} (2000 Samples) <<<")
        limit = 2000
        episodes = 3
        log_interval = 200

    if limit:
        X_train = X_train[:limit]
        y_train = y_train[:limit]
    
    # 1. Train Autoencoder
    input_dim = X_train.shape[1]
    latent_dim = 20
    print(f"Input Features: {input_dim} -> Latent: {latent_dim}")
    
    X_encoded, ae_model = train_autoencoder(X_train, input_dim, latent_dim, epochs=10)
    torch.save(ae_model.state_dict(), AE_PATH)
    
    # 2. Train DQN
    env = NetworkEnv(X_encoded, y_train)
    agent = DQNAgent(state_dim=latent_dim, action_dim=2)
    
    history = [] 
    print(f"--- STARTING HYBRID TRAINING ({episodes} Episodes) ---")
    
    train_freq = 4 # Speed up training by 4x

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
            
            step_count += 1
            
            # Train only every 4 steps (Standard DQN practice)
            if step_count % train_freq == 0:
                agent.replay() # No decay inside
            
            if step_count % log_interval == 0:
                print(f"[Ep {e+1}] Step {step_count} | Reward: {total_reward} | Eps: {agent.epsilon:.4f}")
                sys.stdout.flush()
        
        # Decay ONCE per episode
        agent.update_epsilon()
        
        print(f"--- Episode {e+1}/{episodes} Finished | Score: {total_reward} | Eps: {agent.epsilon:.4f} ---")
        history.append({"episode": e + 1, "reward": total_reward})

    torch.save(agent.model.state_dict(), DQN_PATH)
    pd.DataFrame(history).to_csv(f"{ARTIFACTS_DIR}/metrics.csv", index=False)
    
    # Generate Learning Curve
    df_h = pd.DataFrame(history)
    plt.figure(figsize=(10,5))
    plt.plot(df_h['episode'], df_h['reward'], marker='o')
    plt.title(f'NeuroGuard Training Curve ({ds_type.upper()})')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.savefig(f"{ARTIFACTS_DIR}/training_curve.png")
    print("Training Complete. Models Saved.")

def benchmark_mode():
    print("\n--- BENCHMARK (VALIDATION) ---")
    ds_type = "ciciot" if "--ciciot" in sys.argv else "kdd"
    
    # Load Test Data
    X_test_raw, y_test = load_data("test", ds_type)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load Models
    if not os.path.exists(AE_PATH):
        print("Error: AE Model not found.")
        return

    # Dynamic input dim based on dataset
    ae_model = AutoEncoder(X_test_raw.shape[1], 20).to(device)
    ae_model.load_state_dict(torch.load(AE_PATH, map_location=device))
    ae_model.eval()
    
    agent = DQNAgent(state_dim=20, action_dim=2)
    agent.model.load_state_dict(torch.load(DQN_PATH, map_location=device))
    
    # Force Eval Mode for BatchNorm/Dropout
    agent.model.eval()
    agent.epsilon = 0.0 
    
    print(f"Evaluating on {len(X_test_raw)} samples...")
    y_pred = []
    
    start_time = time.time()
    
    # Inference Loop
    for i in range(len(X_test_raw)):
        raw_tensor = torch.FloatTensor(X_test_raw[i]).unsqueeze(0).to(device)
        with torch.no_grad():
            latent_state, _ = ae_model(raw_tensor)
            q_values = agent.model(latent_state)
            action = torch.argmax(q_values).item()
        y_pred.append(action)
        
        if i % 10000 == 0 and i > 0:
            print(f"Processed {i} flows...")
            
    end_time = time.time()
    total_time = end_time - start_time
    avg_latency = (total_time / len(X_test_raw)) * 1000 

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    print("\n" + "="*40)
    print(f"      NEUROGUARD OFFICIAL VALIDATION      ")
    print(f"      Dataset: {ds_type.upper()}          ")
    print("="*40)
    print(f"Accuracy:       {acc*100:.2f}%")
    print(f"Precision:      {prec*100:.2f}%")
    print(f"Recall:         {rec*100:.2f}%")
    print(f"F1-Score:       {f1*100:.2f}%")
    print("-" * 40)
    print(f"Latency:        {avg_latency:.2f} ms/flow")
    print("-" * 40)
    print("Confusion Matrix:")
    print(f" TN: {cm[0][0]} | FP: {cm[0][1]}")
    print(f" FN: {cm[1][0]} | TP: {cm[1][1]}")
    print("="*40)
    
    # Save validation report
    with open(f"{ARTIFACTS_DIR}/validation_report_{ds_type}.txt", "w") as f:
        f.write(f"Accuracy: {acc}\nPrecision: {prec}\nRecall: {rec}\nF1: {f1}")

if __name__ == "__main__":
    if "--demo" in sys.argv:
        benchmark_mode()
    elif "--full" in sys.argv:
        train_mode()
    else:
        train_mode()