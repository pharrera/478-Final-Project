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
os.makedirs("data", exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def get_model_paths(ds_type):
    """Returns dataset-specific paths to avoid dimension mismatch errors."""
    return (f"{ARTIFACTS_DIR}/dqn_model_{ds_type}.pth", 
            f"{ARTIFACTS_DIR}/ae_model_{ds_type}.pth")

def load_data(mode="train", dataset_type="kdd"):
    X_raw = None
    y = None

    if dataset_type == "ciciot":
        print(f"Loading CICIoT2023 from {DATA_PATH_CIC}...")
        if not os.path.exists(DATA_PATH_CIC):
            print(f"ERROR: {DATA_PATH_CIC} not found.")
            sys.exit(1)
        df = pd.read_csv(DATA_PATH_CIC)
        # CICIoT labels are strings, map them
        y_raw = df['label']
        y = y_raw.apply(lambda x: 0 if x in ['BenignTraffic', 'Benign'] else 1).values
        X_raw = df.drop(['label'], axis=1)
        
    else:
        path = DATA_PATH_TRAIN if mode == "train" else DATA_PATH_TEST
        url = TRAIN_URL if mode == "train" else TEST_URL
        if not os.path.exists(path):
            print(f"Downloading {mode.upper()} set from {url}...")
            with open(path, 'wb') as f:
                f.write(requests.get(url).content)
        print(f"Loading {mode.upper()} data from {path}...")
        df = pd.read_csv(path, names=["duration","protocol_type","service","flag","src_bytes","dst_bytes",
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
                "dst_host_srv_rerror_rate","label","difficulty"])
        y = df['label'].apply(lambda x: 0 if x == 'normal' else 1).values
        X_raw = df.drop(['label', 'difficulty'], axis=1)

    if X_raw is not None:
        le = LabelEncoder()
        for col in X_raw.select_dtypes(include=['object']).columns:
            X_raw[col] = le.fit_transform(X_raw[col].astype(str))
        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw)
        return X, y
    else:
        print("Error: Dataset could not be loaded.")
        sys.exit(1)

def train_mode():
    ds_type = "ciciot" if "--ciciot" in sys.argv else "kdd"
    dqn_path, ae_path = get_model_paths(ds_type)
    
    print(f"Loading Training Data for {ds_type}...")
    X_train, y_train = load_data("train", ds_type)
    
    if "--full" in sys.argv:
        print(f">>> FULL TRAINING ON {ds_type.upper()} ({len(X_train)} Samples) <<<")
        limit = None
        episodes = 20 # Configured for 20 episodes
        log_interval = 5000 
    else:
        limit = 2000
        episodes = 3
        log_interval = 200

    if limit:
        X_train = X_train[:limit]
        y_train = y_train[:limit]
    
    # 1. Train Autoencoder
    input_dim = X_train.shape[1]
    latent_dim = 32
    print(f"Input Features: {input_dim} -> Latent: {latent_dim}")
    
    X_encoded, ae_model = train_autoencoder(X_train, input_dim, latent_dim, epochs=200)
    torch.save(ae_model.state_dict(), ae_path)
    print(f"Saved AE model to {ae_path}")
    
    # 2. Train DQN
    env = NetworkEnv(X_encoded, y_train)
    agent = DQNAgent(state_dim=latent_dim, action_dim=2)
    
    history = [] 
    print(f"--- STARTING HYBRID TRAINING ({episodes} Episodes) ---")
    
    train_freq = 4 

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
            if step_count % train_freq == 0:
                agent.replay() 
            
            if step_count % log_interval == 0:
                print(f"[Ep {e+1}] Step {step_count} | Reward: {total_reward} | Eps: {agent.epsilon:.4f}")
                sys.stdout.flush()
        
        agent.update_epsilon()
        
        print(f"--- Episode {e+1}/{episodes} Finished | Score: {total_reward} | Eps: {agent.epsilon:.4f} ---")
        history.append({"episode": e + 1, "reward": total_reward})

        if (e + 1) % 10 == 0:
            torch.save(agent.model.state_dict(), f"{ARTIFACTS_DIR}/dqn_checkpoint_{ds_type}_ep{e+1}.pth")

    torch.save(agent.model.state_dict(), dqn_path)
    print(f"Saved DQN model to {dqn_path}")
    pd.DataFrame(history).to_csv(f"{ARTIFACTS_DIR}/metrics_{ds_type}.csv", index=False)
    
    # Generate Learning Curve
    df_h = pd.DataFrame(history)
    plt.figure(figsize=(10,5))
    plt.plot(df_h['episode'], df_h['reward'], marker='o')
    plt.title(f'NeuroGuard Training Curve ({ds_type.upper()})')
    plt.savefig(f"{ARTIFACTS_DIR}/training_curve_{ds_type}.png")
    print("Training Complete.")

def benchmark_mode():
    print("\n--- BENCHMARK (VALIDATION) ---")
    ds_type = "ciciot" if "--ciciot" in sys.argv else "kdd"
    dqn_path, ae_path = get_model_paths(ds_type)
    
    # Load Test Data
    X_test_raw, y_test = load_data("test", ds_type)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Validate Model Existence
    if not os.path.exists(ae_path):
        print(f"Error: AE Model for {ds_type} not found at {ae_path}.")
        print(f"Run 'python src/main.py --full --{ds_type}' first to train.")
        return

    # Dynamic input dim based on dataset
    latent_dim = 32
    input_dim = X_test_raw.shape[1]
    
    ae_model = AutoEncoder(input_dim, latent_dim).to(device)
    ae_model.load_state_dict(torch.load(ae_path, map_location=device))
    ae_model.eval()
    
    agent = DQNAgent(state_dim=latent_dim, action_dim=2)
    agent.model.load_state_dict(torch.load(dqn_path, map_location=device))
    agent.model.eval()
    agent.epsilon = 0.0 
    
    print(f"Evaluating on {len(X_test_raw)} samples...")
    y_pred = []
    
    start_time = time.time()
    
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
    
    with open(f"{ARTIFACTS_DIR}/validation_report_{ds_type}.txt", "w") as f:
        f.write(f"Accuracy: {acc}\nPrecision: {prec}\nRecall: {rec}\nF1: {f1}")

if __name__ == "__main__":
    if "--demo" in sys.argv:
        benchmark_mode()
    elif "--full" in sys.argv:
        train_mode()
    else:
        train_mode()