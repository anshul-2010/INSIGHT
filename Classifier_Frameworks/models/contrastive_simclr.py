import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderAutoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*8*8, latent_dim)
        )
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 64*8*8),
            nn.Unflatten(1,(64,8,8)),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64,32,3,padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32,3,3,padding=1), nn.Sigmoid()
        )

    def encode(self,x):
        return self.enc(x)
    def decode(self,z):
        return self.dec(z)
    def forward(self,x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z
    
class ProjectionHead(nn.Module):
    def __init__(self, in_dim=128, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim)
        )
    def forward(self,x):
        return self.net(x)
    
class SimCLRTrainer:
    def __init__(self, encoder, projection_head, device='cpu', temp=0.5):
        self.encoder = encoder.to(device)
        self.proj_head = projection_head.to(device)
        self.device = device
        self.temp = temp
        self.criterion = nn.CrossEntropyLoss()

    def info_nce_loss(self, zis, zjs):
        batch_size = zis.size(0)
        representations = torch.cat([zis, zjs], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        labels = torch.arange(batch_size).to(self.device)
        labels = torch.cat([labels, labels], dim=0)
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(self.device)
        similarity_matrix = similarity_matrix[~mask].view(2 * batch_size, -1)
        # compute positive pair similarity (diagonals corresponding to positive pairs)
        positives = torch.cat([torch.diag(similarity_matrix, batch_size), torch.diag(similarity_matrix, -batch_size)], dim=0)
        logits = torch.cat([positives.unsqueeze(1), similarity_matrix], dim=1)
        logits /= self.temp
        loss = self.criterion(logits, labels)
        return loss

    def train_step(self, x1, x2, optim):
        self.encoder.train()
        self.proj_head.train()
        x1, x2 = x1.to(self.device), x2.to(self.device)
        optim.zero_grad()
        h1 = self.encoder(x1)[1]
        h2 = self.encoder(x2)[1]
        z1 = self.proj_head(h1)
        z2 = self.proj_head(h2)
        loss = self.info_nce_loss(z1, z2)
        loss.backward()
        optim.step()
        return loss.item()