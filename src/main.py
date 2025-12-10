import pandas as pd
import numpy as np
import torch
import os
import sys
import requests
import random
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from environment import NetworkEnv
from agent import DQNAgent
from autoencoder import train_autoencoder, AutoEncoder

# Config
DATA_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B_20Percent.txt"
DATA_PATH_KDD = "data/nsl_kdd_train.csv"
DATA_PATH_CIC = "data/ciciot2023.csv"
ARTIFACTS_DIR = "artifacts/release"
DQN_PATH = f"{ARTIFACTS_DIR}/dqn_model.pth"
AE_PATH = f"{ARTIFACTS_DIR}/ae_model.pth"

os.makedirs("data", exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def load_data(dataset_type="kdd"):
    """
    Universal Loader.
    dataset_type: 'kdd' (Default) or 'ciciot'
    """
    if dataset_type == "ciciot":
        print(f"Loading CICIoT2023 from {DATA_PATH_CIC}...")
        if not os.path.exists(DATA_PATH_CIC):
            print(f"ERROR: {DATA_PATH_CIC} not found.")
            print("Please download a part-file from Kaggle/UNB, rename to ciciot2023.csv, and place in data/.")
            sys.exit(1)
            
        # CICIoT2023 Load Logic
        df = pd.read_csv(DATA_PATH_CIC)
        
        # 1. Handle Labels
        # The column is usually 'label'. Benign is 'BenignTraffic'
        y_raw = df['label']
        y = y_raw.apply(lambda x: 0 if x in ['BenignTraffic', 'Benign'] else 1).values
        
        # 2. Features (Drop label)
        X_raw = df.drop(['label'], axis=1)
        
    else:
        # NSL-KDD Load Logic (Default)
        if not os.path.exists(DATA_PATH_KDD):
            print(f"Downloading NSL-KDD...")
            response = requests.get(DATA_URL)
            with open(DATA_PATH_KDD, 'wb') as f:
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
        df = pd.read_csv(DATA_PATH_KDD, names=cols)
        y = df['label'].apply(lambda x: 0 if x == 'normal' else 1).values
        X_raw = df.drop(['label', 'difficulty'], axis=1)

    # Universal Preprocessing (Encoding & Scaling)
    # This handles both 41 features (KDD) and 46+ features (CICIoT) automatically
    le = LabelEncoder()
    for col in X_raw.select_dtypes(include=['object']).columns:
        X_raw[col] = le.fit_transform(X_raw[col])
        
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X_raw)
    
    return X, y

def train_mode():
    # Detect Dataset Flag
    ds_type = "ciciot" if "--ciciot" in sys.argv else "kdd"
    features, labels = load_data(ds_type)
    
    # Fast Mode vs Full Mode limits
    if "--full" in sys.argv:
        print(f">>> FULL TRAINING ON {ds_type.upper()} ({len(features)} Samples) <<<")
        limit = None
        episodes = 5
        log_interval = 5000
    else:
        print(f">>> FAST MODE ON {ds_type.upper()} (2000 Samples) <<<")
        limit = 2000
        episodes = 3
        log_interval = 200

    X_train = features[:limit] if limit else features
    y_train = labels[:limit] if limit else labels
    
    # 1. Train Autoencoder
    # Dynamic Input Dim: Fits 41 (KDD) or 46 (CICIoT) automatically
    input_dim = X_train.shape[1]
    latent_dim = 20
    print(f"Input Features: {input_dim} -> Latent: {latent_dim}")
    
    X_encoded, ae_model = train_autoencoder(X_train, input_dim, latent_dim)
    torch.save(ae_model.state_dict(), AE_PATH)
    
    # 2. Train DQN
    env = NetworkEnv(X_encoded, y_train)
    agent = DQNAgent(state_dim=latent_dim, action_dim=2)
    
    history = [] 
    print(f"--- STARTING HYBRID TRAINING ---")

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
            if step_count % log_interval == 0:
                print(f"[Ep {e+1}] Step {step_count} | Reward: {total_reward} | Eps: {agent.epsilon:.2f}")
                sys.stdout.flush()
        history.append({"episode": e + 1, "reward": total_reward})

    torch.save(agent.model.state_dict(), DQN_PATH)
    pd.DataFrame(history).to_csv(f"{ARTIFACTS_DIR}/metrics.csv", index=False)
    
    df_h = pd.DataFrame(history)
    plt.figure()
    plt.plot(df_h['episode'], df_h['reward'])
    plt.title(f'NeuroGuard Training ({ds_type.upper()})')
    plt.savefig(f"{ARTIFACTS_DIR}/training_curve.png")
    print("Training Complete. Models Saved.")

def benchmark_mode():
    print("\n--- STARTING BENCHMARK MODE ---")
    if not os.path.exists(DQN_PATH) or not os.path.exists(AE_PATH):
        print("Error: Models not found. Run 'make up' first.")
        return
        
    # Detect Dataset Flag
    ds_type = "ciciot" if "--ciciot" in sys.argv else "kdd"
    features, labels = load_data(ds_type)
    
    # Load Models
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    input_dim = features.shape[1]
    
    ae_model = AutoEncoder(input_dim, 20).to(device)
    ae_model.load_state_dict(torch.load(AE_PATH, map_location=device))
    ae_model.eval()
    
    agent = DQNAgent(state_dim=20, action_dim=2)
    agent.model.load_state_dict(torch.load(DQN_PATH, map_location=device))
    agent.epsilon = 0.0 
    
    # Benchmark
    # Use subset for speed if full dataset is massive (CICIoT part files are usually ~200k rows)
    limit = 10000 if len(features) > 10000 else len(features)
    print(f"Evaluating on {limit} samples from {ds_type.upper()}...")
    
    indices = np.random.choice(len(features), limit, replace=False)
    X_test = features[indices]
    y_true = labels[indices]
    y_pred = []
    
    start_time = time.time()
    for i in range(len(X_test)):
        raw_tensor = torch.FloatTensor(X_test[i]).unsqueeze(0).to(device)
        with torch.no_grad():
            latent_state, _ = ae_model(raw_tensor)
            q_values = agent.model(latent_state)
            action = torch.argmax(q_values).item()
        y_pred.append(action)
    end_time = time.time()
    
    # Metrics
    total_time = end_time - start_time
    avg_latency = (total_time / len(X_test)) * 1000 
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    print("\n" + "="*40)
    print(f"      NEUROGUARD BENCHMARK ({ds_type.upper()})      ")
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

if __name__ == "__main__":
    if "--demo" in sys.argv:
        benchmark_mode()
    elif "--full" in sys.argv:
        train_mode()
    else:
        train_mode()