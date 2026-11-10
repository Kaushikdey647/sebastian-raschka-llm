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
        self.scale = torch.sqrt(torch.tensor(output_dim))
        self.register_buffer("mask", torch.triu(torch.ones(input_length, input_length), diagonal=1))
    
    def forward(self, x):
        batch_size, sequence_length, input_dim = x.shape

        # Tensor Shape: (Batch Size, Sequence Length, Input Dim) -> (Batch Size, Sequence Length, Output Dim)
        keys = self.W_k(x)
        values = self.W_v(x)
        queries = self.W_q(x)

        # Tensor Shape: (Batch Size, Sequence Length, Output Dim) -> (Batch Size, Sequence Length, Num Heads, Head Dim)
        keys = keys.view(batch_size, sequence_length, self.num_heads, self.head_dim)
        values = values.view(batch_size, sequence_length, self.num_heads, self.head_dim)
        queries = queries.view(batch_size, sequence_length, self.num_heads, self.head_dim)

        # Tensor Shape: (Batch Size, Sequence Length, Num Heads, Head Dim) -> (Batch Size, Sequence Length, Num Heads, Num Heads)
        attention_scores = queries @ keys.transpose(-2, -1)
        attention_scores.masked_fill_(self.mask.bool()[:sequence_length, :sequence_length], -torch.inf)
        attention_weights = torch.softmax(attention_scores / self.scale, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Tensor Shape: (Batch Size, Sequence Length, Num Heads, Num Heads) -> (Batch Size, Sequence Length, Num Heads, Head Dim)
        context_vector = attention_weights @ values

        # Tensor Shape: (Batch Size, Sequence Length, Num Heads, Head Dim) -> (Batch Size, Sequence Length, Output Dim)
        context_vector = context_vector.view(batch_size, sequence_length, self.output_dim)

        # Tensor Shape: (Batch Size, Sequence Length, Output Dim) -> (Batch Size, Sequence Length, Output Dim)
        context_vector = self.out_proj(context_vector)
        return context_vector, attention_weights