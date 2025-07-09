import torch.nn as nn


class ClassEmbedding(nn.Module):
    """Embed class labels into a latent space"""

    def __init__(self, num_classes, emb_dim):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Embedding(num_classes, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, labels):
        """labels: (batch,) -> (batch, emb_dim)"""
        return self.embedding(labels)
