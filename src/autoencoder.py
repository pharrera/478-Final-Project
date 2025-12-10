import torch

import torch.nn as nn
import torch.optim as optim
import numpy as np

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Optimization: Using Apple Silicon GPU (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=20):
        super(AutoEncoder, self).__init__()
        # Encoder: Compresses 41 features -> 20 features
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
            nn.ReLU()
        )
        # Decoder: Reconstructs 20 features -> 41 features
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
            nn.Sigmoid() # Outputs 0-1 (matching MinMax scaling)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

def train_autoencoder(X_train, input_dim, latent_dim=20, epochs=10):
    """
    Pre-trains the Autoencoder on the dataset.
    Returns: (Compressed Features, Trained Model)
    """
    print(f"\n>>> PRE-TRAINING AUTOENCODER ({epochs} Epochs) <<<")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoEncoder(input_dim, latent_dim).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    X_tensor = torch.FloatTensor(X_train).to(device)
    batch_size = 256
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        permutation = torch.randperm(X_tensor.size()[0])
        
        for i in range(0, X_tensor.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_x = X_tensor[indices]
            
            optimizer.zero_grad()
            _, decoded = model(batch_x)
            
            # Autoencoder tries to output exactly what went in (reconstruction)
            loss = criterion(decoded, batch_x)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        # Print progress every few epochs
        if (epoch+1) % 2 == 0 or epoch == 0:
            print(f"AE Epoch {epoch+1}/{epochs} | Reconstruction Loss: {epoch_loss/len(X_train):.6f}")
            
    print(">>> Autoencoder Training Complete. Extracting Latent Features...")
    
    # Extract the compressed "Latent Features" to feed into the DQN
    model.eval()
    with torch.no_grad():
        encoded_features, _ = model(X_tensor)
    
    return encoded_features.cpu().numpy(), model