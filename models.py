from attention import MultiHeadAttention
import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        emb_dim = cfg["embedding_dim"]
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.GELU(),
            nn.Linear(4 * emb_dim, emb_dim),
        )

    def forward(self, x):
        return self.layers(x)

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return norm_x * self.scale + self.shift

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attention = MultiHeadAttention(
            input_dim=cfg["embedding_dim"],
            output_dim=cfg["embedding_dim"],
            input_length=cfg["context_length"],
            num_heads=cfg["num_heads"],
            dropout=cfg["dropout_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.feed_forward = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["embedding_dim"])
        self.norm2 = LayerNorm(cfg["embedding_dim"])
        self.drop_shortcut = nn.Dropout(cfg["dropout_rate"])
    
    def forward(self, x):
        shortcut = x
        x = self.norm1(x) # Normalize X for the attention layer
        x = self.attention(x)  # Apply the attention layer for context vector
        x = self.drop_shortcut(x) # Dropout the context vector
        x = x + shortcut
        shortcut = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x