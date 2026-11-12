from torch import nn
import torch

class BasicAttention(nn.Module):
    def __init__(self, random_seed=42):
        super().__init__()
        torch.manual_seed(self.random_seed)
    
    def forward(self, X):
        # X: (Batch Size, Sequence Length, Input Dim)
        attention_scores = X @ X.T
        attention_weights = torch.softmax(attention_scores, dim=-1)
        context_vector = attention_weights @ X
        return context_vector, attention_weights

class SelfAttention(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.1, random_seed=42):
        super().__init__()
        torch.manual_seed(random_seed)
        self.W_q = nn.Linear(input_dim, output_dim)
        self.W_k = nn.Linear(input_dim, output_dim)
        self.W_v = nn.Linear(input_dim, output_dim)
        self.scale = torch.sqrt(torch.tensor(output_dim))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, X):
        Q = self.W_q(X) 
        K = self.W_k(X) 
        V = self.W_v(X)
        # We transpose AFTER the batch dimension to get the correct shape for the dot product
        attention_scores = Q @ K.transpose(-2, -1) / self.scale
        attention_weights = torch.softmax(attention_scores, dim=-1) 
        context_vector = attention_weights @ V 
        return context_vector, attention_weights

class CausalAttention(nn.Module):
    def __init__(self, input_dim, output_dim, input_length, dropout=0.1, random_seed=42, qkv_bias=False):
        super().__init__()
        torch.manual_seed(random_seed)
        self.input_length = input_length
        self.output_dim = output_dim
        self.random_seed = random_seed
        self.dropout = torch.nn.Dropout(dropout)
        self.W_q = nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.W_k = nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.W_v = nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.scale = torch.sqrt(torch.tensor(output_dim))
        self.register_buffer("mask", torch.triu(torch.ones(input_length, input_length), diagonal=1))
    
    def forward(self, x):
        # X: (Batch Size, Sequence Length, Input Dim)
        batch_size, sequence_length, input_dim = x.shape
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        attention_scores = Q @ K.transpose(-2, -1)
        # mask by setting upper triangle to -inf and then softmax sets them zero
        # Trailing underscore is used to modify the tensor in place
        attention_scores.masked_fill_(self.mask.bool()[:sequence_length, :sequence_length], -torch.inf)
        attention_weights = torch.softmax(attention_scores / self.scale, dim=-1)
        attention_weights = self.dropout(attention_weights)
        context_vector = attention_weights @ V
        return context_vector, attention_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, output_dim, input_length, num_heads, dropout=0.1, random_seed=42, qkv_bias=False):
        super().__init__()
        torch.manual_seed(random_seed)
        self.input_length = input_length
        self.output_dim = output_dim
        self.random_seed = random_seed
        self.num_heads = num_heads 
        self.head_dim = output_dim // num_heads # Dimension of each head, output_dim must be divisible by num_heads

        self.dropout = torch.nn.Dropout(dropout)
        self.W_q = nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.W_k = nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.W_v = nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.out_proj = nn.Linear(output_dim, output_dim)
        self.scale = torch.sqrt(torch.tensor(self.head_dim))
        self.register_buffer("mask", torch.triu(torch.ones(input_length, input_length), diagonal=1))
        
    def forward(self, x):
        batch_size, sequence_length, input_dim = x.shape
        # Let Batch Size be B, Sequence Length be L, Input Dim be D_i, Output Dim be D_o, Num Heads be H, Head Dim be D_h

        # First we init K,Q and V by multiplying the weights
        # (B, L, D_i) -> (B, L, D_o)
        keys = self.W_k(x)
        values = self.W_v(x)
        queries = self.W_q(x)

        # Then we split the keys, values and queries into heads
        # (B, L, D_o) -> (B, L, H, D_h) -> (B, H, L, D_h). Heads sit beside the batch so the matmul runs over L.
        keys = keys.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)

        # Then we compute the attention scores
        # (B, H, L, D_h) @ (B, H, D_h, L) -> (B, H, L, L)
        attention_scores = queries @ keys.transpose(-2, -1)
        attention_scores.masked_fill_(self.mask.bool()[:sequence_length, :sequence_length], -torch.inf)
        attention_weights = torch.softmax(attention_scores / self.scale, dim=-1) # (B, H, L, L)
        attention_weights = self.dropout(attention_weights)

        # Then we compute the context vector
        # (B, H, L, L) @ (B, H, L, D_h) -> (B, H, L, D_h)
        context_vector = attention_weights @ values
        # (B, H, L, D_h) -> (B, L, H, D_h) -> (B, L, D_o)
        context_vector = context_vector.transpose(1, 2).contiguous().view(batch_size, sequence_length, self.output_dim)
        # (B, L, D_o) -> (B, L, D_o)
        context_vector = self.out_proj(context_vector) # (B, L, D_o) -> (B, L, D_o) Note: This is a linear projection to the output dimension
        return context_vector, attention_weights