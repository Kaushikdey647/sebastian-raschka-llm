import tiktoken
from torch.utils.backcompat import keepdim_warning
from torch.utils.data import DataLoader
import torch
from verdict import VerdictDataset
from basic_tokenizer import Tokenizer
from attention import SelfAttention, CausalAttention, MultiHeadAttention
from models import TransformerBlock
import torch.nn as nn

def start_emb_layer(vocab_size,output_dim):
    emb_layer = torch.nn.Embedding(vocab_size,output_dim)
    return emb_layer

def get_verdict_text():
    with open("the-verdict.txt", "r") as file:
        text = file.read()
    return text

def create_dataloader_v1(text, batch_size=4, max_length=256, stride=128, num_workers=0, shuffle=True, drop_last=True):
    tokenizer = Tokenizer()
    dataset = VerdictDataset(text, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )
    return dataloader

def create_dataloader_v2(text, batch_size=4, max_length=256, stride=128, num_workers=0, shuffle=True, drop_last=True):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = VerdictDataset(text, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )
    return dataloader

def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    cfg = {
        "vocab_size": 50257,
        "embedding_dim": 256,
        "context_length": 1024,
        "num_layers": 12,
        "qkv_bias": True,
        "num_heads": 8,
        "dropout_rate": 0.1,
    }
    x = torch.randn(1, 5, 256)
    block = TransformerBlock(cfg)
    output = block(x)
    print(x.shape)
    print(output.shape)
    print(x[0,0,:])
    print(output[0,0,:])

if __name__ == "__main__":
    main()
