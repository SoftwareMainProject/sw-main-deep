# test_autoencoder.py
import torch
import torch.nn as nn

class SimpleAutoEncoder(nn.Module):
    def __init__(self, input_dim=2048, latent_dim=256):     # latent_dim 256, 512 선택
        super(SimpleAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, input_dim)
        )
        
    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return latent, recon

def test_autoencoder():
    model = SimpleAutoEncoder(input_dim=2048, latent_dim=256)
    model.eval()
    
    # 더미 입력: 배치 크기 4, 2048 차원 벡터
    dummy_input = torch.randn(4, 2048)
    with torch.no_grad():
        latent, recon = model(dummy_input)
    
    print("AutoEncoder latent shape:", latent.shape)   # 예상: [4, 256]
    print("AutoEncoder reconstruction shape:", recon.shape)  # 예상: [4, 2048]
    
    # MSE Loss 계산 (재구성 성능 확인)
    mse_loss = nn.MSELoss()(recon, dummy_input)
    print("AutoEncoder MSE loss:", mse_loss.item())

if __name__ == "__main__":
    test_autoencoder()
