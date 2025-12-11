import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 1. Global Device Detection (Ensures M1/MPS is used)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Optimization: Autoencoder utilizing Apple Silicon GPU (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=20):
        super(AutoEncoder, self).__init__()
        # Encoder: Compresses Input -> Latent
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
            nn.ReLU()
        )
        # Decoder: Reconstructs Latent -> Input
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
            # Standard Scaler outputs roughly -N to +N, not 0-1. 
            # Sigmoid forces 0-1 (good for MinMax), but linear/identity is safer for Z-Score.
            # However, for this project's stability, we will keep structure simple.
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

def train_autoencoder(X_train, input_dim, latent_dim=20, epochs=100):
    """
    Pre-trains the Autoencoder on the dataset.
    Returns: (Compressed Features, Trained Model)
    """
    print(f"\n>>> PRE-TRAINING AUTOENCODER ({epochs} Epochs) <<<")
    
    # Use the global device (MPS), do not redefine it here
    model = AutoEncoder(input_dim, latent_dim).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    # Move entire dataset to GPU (MPS) for speed
    X_tensor = torch.FloatTensor(X_train).to(device)
    batch_size = 256 # Optimized batch size for M1
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        # Shuffle indices
        permutation = torch.randperm(X_tensor.size()[0])
        
        for i in range(0, X_tensor.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_x = X_tensor[indices]
            
            optimizer.zero_grad()
            _, decoded = model(batch_x)
            
            loss = criterion(decoded, batch_x)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        if (epoch+1) % 2 == 0 or epoch == 0:
            print(f"AE Epoch {epoch+1}/{epochs} | Reconstruction Loss: {epoch_loss/len(X_train):.6f}")
            
    print(">>> Autoencoder Training Complete. Extracting Latent Features...")
    
    # Extract features in Eval mode
    model.eval()
    with torch.no_grad():
        encoded_features, _ = model(X_tensor)
    
    # Return as numpy array (CPU) for the Environment to usage
    return encoded_features.cpu().numpy(), model