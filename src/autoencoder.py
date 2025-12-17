import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Global Device Detection
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
    
    model = AutoEncoder(input_dim, latent_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    X_tensor = torch.FloatTensor(X_train).to(device)
    batch_size = 256
    
    # Noise factor for robustness (Denoising Autoencoder)
    noise_factor = 0.1
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        permutation = torch.randperm(X_tensor.size()[0])
        
        for i in range(0, X_tensor.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_x = X_tensor[indices]
            
            # Add Noise to Input (Robustness)
            noise = torch.randn_like(batch_x) * noise_factor
            batch_x_noisy = batch_x + noise
            
            optimizer.zero_grad()
            _, decoded = model(batch_x_noisy) # Feed Noisy
            
            loss = criterion(decoded, batch_x) # Compare to Clean
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"AE Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(X_train):.6f}")
            
    print(">>> Autoencoder Training Complete. Extracting Latent Features...")
    
    model.eval()
    with torch.no_grad():
        encoded_features, _ = model(X_tensor)
    
    return encoded_features.cpu().numpy(), model