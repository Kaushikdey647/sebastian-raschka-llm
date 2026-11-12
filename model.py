from attention import SelfAttention, CausalAttention, MultiHeadAttention
import torch
import torch.nn as nn

class DummyTransformerBlock(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_heads,
        dropout_rate,
        qkv_bias,
        random_seed
    ):
        super().__init__()
    def forward(self, x):
        return x

class DummyNormLayer(nn.Module):
    def __init__(
        self,
        embedding_dim,
        dropout_rate,
        random_seed
    ):
        super().__init__()
    def forward(self, x):
        return x

class DummyGPT2Model(nn.Module):
    def __init__(
        self,
        vocab_size,
        context_length,
        embedding_dim,
        num_heads,
        num_layers,
        dropout_rate,
        qkv_bias,
        random_seed
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.qkv_bias = qkv_bias
        self.random_seed = random_seed

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(context_length, embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.transformer_blocks = nn.Sequential(
            *[
                DummyTransformerBlock(
                    embedding_dim,
                    num_heads,
                    dropout_rate,
                    qkv_bias,
                    random_seed
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = DummyNormLayer(
            embedding_dim,
            dropout_rate,
            random_seed
        )
        self.out_head = nn.Linear(embedding_dim, vocab_size, bias=False)
    def forward(self, x):
        batch_size, seq_len = x.shape
        token_embeddings = self.token_embedding(x)
        position_embeddings = self.position_embedding(torch.arange(seq_len, device=x.device))
        x = token_embeddings + position_embeddings
        x = self.dropout(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits