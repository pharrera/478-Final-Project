# NeuroGuard: Adaptive Network Threat Detection using DQN

**Course:** 478 Final Project
**Milestone:** Combined Alpha–Beta Integrated Release
**Status:** Feature Complete (Ready for Demo)

## 1. Project Overview

NeuroGuard is a Reinforcement Learning (RL) based Intrusion Detection System (IDS). Unlike traditional signature-based systems (e.g., Snort) that rely on static rules, NeuroGuard utilizes a **Deep Q-Network (DQN)** agent. The agent observes network traffic features and learns an optimal policy to classify traffic as "Normal" (Pass) or "Attack" (Block), balancing the reward of stopping attacks against the penalty of blocking legitimate users.

This release demonstrates a fully reproducible, containerized vertical slice of the system using the **NSL-KDD** dataset.

---

## 2. Architecture

The system follows an Offline Reinforcement Learning architecture designed for reproducibility in a Docker environment.

### Components

1. **Environment (Gym Wrapper):** A custom OpenAI Gym environment (`src/environment.py`) that ingests the NSL-KDD dataset and simulates a sequential network stream. It provides the state (traffic features) and calculates rewards based on the agent's actions.
2. **Agent (DQN):** A PyTorch-based Deep Q-Network (`src/agent.py`) utilizing Experience Replay to stabilize training.
3. **Pipeline:**
   * **Input:** 41-dimensional feature vectors (Duration, Protocol, Service, Flags, etc.).
   * **Action Space:** Discrete {0: Pass, 1: Block}.
   * **Reward Function:** +1 for correct classification, -5 for False Positives/Negatives.

---

## 3. Runbook (Reproducible Build)

This project is containerized. You do not need Python or PyTorch installed on your host machine.

### Prerequisites

* Docker & Docker Compose

### Commands

The entire lifecycle is managed via the `Makefile`.

#### 1. Bootstrap (Build the System)

Builds the Docker container and installs dependencies (PyTorch, Pandas, Scikit-Learn).

```bash
make bootstrap
```


#### 2. End-to-End Run (Train & Demo)

**Rubric Requirement:** Runs the full vertical slice. This command will:

1. Download the NSL-KDD dataset (if missing).
2. Train the DQN agent for 3 episodes.
3. Save the model (`dqn_model.pth`) and metrics (`metrics.csv`).
4. Automatically launch the **Demo Mode** to perform inference on 10 random samples.

```
make up && make demo
```

*Expected Time: < 2 minutes.*

#### 3. Run Tests

Executes the Alpha/Beta test suite (Happy path, Negative tests, Dimension checks).

```
make test

```


#### 4. Clean Up

Removes containers and temporary artifacts.

```
make clean
```


## 4. Security Invariants

To ensure the integrity and safety of the detection system, the following invariants are enforced:

1. **Input Normalization:** All traffic features are scaled to the range `[0, 1]` using MinMax scaling before entering the Neural Network. This prevents gradient explosion and effectively sanitizes extreme values that could destabilize the model.
2. **Environment Isolation:** The training and inference processes run inside a non-privileged Docker container, isolating the RL execution environment from the host OS.
3. **Deterministic Evaluation:** The "Demo Mode" sets  `epsilon=0.0` (pure exploitation), ensuring that the security policy is deterministic and reproducible during inference, rather than probabilistic.
4. **Data Provenance:** The dataset is programmatically fetched from the official NSL-KDD repository over HTTPS, ensuring data integrity is not compromised by manual file handling.


## 5. Evaluation & Status (Alpha/Beta Summary)

### What Works (The Vertical Slice)

We have successfully implemented the complete RL pipeline:

* **Ingestion:** Automatic download and parsing of the NSL-KDD dataset.
* **Training Loop:** The DQN agent successfully learns from the environment. Telemetry shows the "Total Reward" increasing from ~80 (random guessing) to ~900+ (learned policy) within 3 episodes.
* **Observability:** Training metrics are automatically logged to **`artifacts/release/metrics.csv` and visualized in** `artifacts/release/training_curve.png`.
* **Persistence:** The trained model state is serialized to disk and successfully reloaded for inference.

### Draft Results

Initial evaluation on the dataset subset indicates rapid convergence:

* **Accuracy:** In the controlled Demo environment, the agent achieves **100% accuracy (10/10)** on random test samples.
* **Learning Curve:** The agent overcomes the "cold start" phase within the first 2 episodes.

### What's Next (Final Release Goals)

* **Adversarial Robustness:** We plan to introduce noise into the testing data to evaluate how fragile the DQN is to adversarial examples.
* **Hyperparameter Tuning:** Automating the search for the optimal Gamma (discount factor) to balance immediate vs. long-term threat prevention.

---

## 6. Repository Structure

neuroguard-dqn/
├── Dockerfile              # Container definition
├── Makefile                # Command automation
├── docker-compose.yml      # Service orchestration
├── requirements.txt        # Python dependencies
├── README.md               # Documentation
├── .github/workflows/      # CI Configuration
├── src/
│   ├── agent.py            # DQN Logic (PyTorch)
│   ├── environment.py      # Gym Wrapper (NSL-KDD)
│   ├── main.py             # Entry point (Train/Demo)
│   └── tests.py            # Unit & Integration Tests
└── artifacts/
    └── release/            # Evidence (Charts, CSVs, Model)
