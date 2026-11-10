import tiktoken
from torch.utils.data import DataLoader
import torch
from verdict import VerdictDataset
from basic_tokenizer import Tokenizer
from attention import SelfAttention, BasicAttention

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
    # Getting Constants out of the way
    text = get_verdict_text()
    manual_seed = 42
    max_length = 4
    batch_size = 8
    vocab_size = 50257
    output_dim = 256

    # Creating the DataLoader and Embedding Layers for the Verdict Dataset
    dataloader_v1 = create_dataloader_v1(
        text = text,
        batch_size = batch_size,
        max_length = max_length,
        stride = max_length,
        num_workers = 0,
        shuffle = False,
        drop_last = True
    )
    input_emb_layer = start_emb_layer(
        vocab_size=vocab_size,
        output_dim=output_dim,
        manual_seed=manual_seed
    )
    pos_emb_layer = start_emb_layer(
        vocab_size=max_length,
        output_dim=output_dim,
        manual_seed=manual_seed
    )

    # Having some fun

    iter_v1 = iter(dataloader_v1)
    input_ids, target_ids = next(iter_v1)
    token_embeddings = input_emb_layer(input_ids)
    positional_embeddings = pos_emb_layer(torch.arange(max_length))
    input_embeddings = token_embeddings + positional_embeddings
    self_attention = SelfAttention(
        input_dim=output_dim,
        output_dim=output_dim,
        random_seed=manual_seed
    )
    context_vector, attention_weights = self_attention(input_embeddings)
    print(context_vector.shape, " (Context Vector)")
    print(attention_weights.shape, " (Attention Weights)")

if __name__ == "__main__":
    main()
