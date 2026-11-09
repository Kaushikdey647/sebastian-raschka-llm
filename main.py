import tiktoken
from torch.utils.data import DataLoader
import torch
from verdict import VerdictDataset
from basic_tokenizer import Tokenizer

def start_emb_layer(vocab_size,output_dim,manual_seed=42):
    torch.manual_seed(manual_seed)
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
    text = get_verdict_text()
    dataloader_v1 = create_dataloader_v1(text, batch_size=8, max_length=4, stride=4, num_workers=0, shuffle=True, drop_last=True)
    iter_v1 = iter(dataloader_v1)
    emb_layer = start_emb_layer(vocab_size=6,output_dim=3,manual_seed=42)
    print(emb_layer.weight)

if __name__ == "__main__":
    main()
