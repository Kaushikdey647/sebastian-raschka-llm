import tiktoken
from torch.utils.data import DataLoader
from verdict import VerdictDataset
from basic_tokenizer import Tokenizer

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
    dataloader_v1 = create_dataloader_v1(text, batch_size=4, max_length=256, stride=128, num_workers=0, shuffle=True, drop_last=True)
    dataloader_v2 = create_dataloader_v2(text, batch_size=4, max_length=256, stride=128, num_workers=0, shuffle=True, drop_last=True)
    iter_v2 = iter(dataloader_v2)
    input_ids, target_ids = next(iter_v2)
    print(input_ids)
    print(target_ids)
    gpt2 = tiktoken.get_encoding("gpt2")
    for ids, targets in zip(input_ids, target_ids):
        print(gpt2.decode(ids.tolist()))
        print(gpt2.decode(targets.tolist()))

if __name__ == "__main__":
    main()
