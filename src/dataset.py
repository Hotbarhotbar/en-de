import os, math
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader

PAD = "<pad>"
UNK = "<unk>"
BOS = "<bos>"
EOS = "<eos>"

class Vocab:
    def __init__(self, min_freq=2):
        self.freq = Counter()
        self.min_freq = min_freq
        self.itos = []
        self.stoi = {}

    def build(self, sentences):
        for s in sentences:
            for tok in s.strip().split():
                self.freq[tok] += 1
        # add special tokens
        self.itos = [PAD, UNK, BOS, EOS] + [tok for tok, f in self.freq.items() if f >= self.min_freq]
        self.stoi = {tok:i for i, tok in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

def read_lines(path):
    try:
        with open(path, encoding="utf-8") as f:
            return [l.strip() for l in f if l.strip()]
    except FileNotFoundError:
        print(f"Error: File {path} not found!")
        return []

class ParallelDataset(Dataset):
    def __init__(self, src_lines, trg_lines, src_vocab, trg_vocab, max_len=128):
        assert len(src_lines) == len(trg_lines)
        self.src = src_lines
        self.trg = trg_lines
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.src)

    def encode_line(self, line, vocab, add_bos=True, add_eos=True):
        toks = line.strip().split()
        ids = []
        if add_bos:
            ids.append(vocab.stoi.get(BOS, vocab.stoi[UNK]))
        for t in toks[:self.max_len-2]:
            ids.append(vocab.stoi.get(t, vocab.stoi.get(UNK)))
        if add_eos:
            ids.append(vocab.stoi.get(EOS, vocab.stoi.get(UNK)))
        return ids

    def __getitem__(self, idx):
        src_ids = self.encode_line(self.src[idx], self.src_vocab)
        trg_ids = self.encode_line(self.trg[idx], self.trg_vocab)
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(trg_ids, dtype=torch.long)

def collate_fn(batch):
    srcs, trgs = zip(*batch)
    src_lens = [len(s) for s in srcs]
    trg_lens = [len(t) for t in trgs]
    max_src = max(src_lens)
    max_trg = max(trg_lens)
    PAD_ID = 0
    src_batch = torch.full((len(batch), max_src), PAD_ID, dtype=torch.long)
    trg_batch = torch.full((len(batch), max_trg), PAD_ID, dtype=torch.long)
    for i, s in enumerate(srcs):
        src_batch[i, :len(s)] = s
    for i, t in enumerate(trgs):
        trg_batch[i, :len(t)] = t
    return src_batch, trg_batch

def build_dataloaders(data_dir, batch_size=64, min_freq=2, max_len=128):
    train_src = read_lines(os.path.join(data_dir, "train.en"))
    train_trg = read_lines(os.path.join(data_dir, "train.de"))
    valid_src = read_lines(os.path.join(data_dir, "valid.en"))
    valid_trg = read_lines(os.path.join(data_dir, "valid.de"))
    test_src = read_lines(os.path.join(data_dir, "test.en"))
    test_trg = read_lines(os.path.join(data_dir, "test.de"))

    src_vocab = Vocab(min_freq=min_freq)
    trg_vocab = Vocab(min_freq=min_freq)
    src_vocab.build(train_src)
    trg_vocab.build(train_trg)

    train_dataset = ParallelDataset(train_src, train_trg, src_vocab, trg_vocab, max_len=max_len)
    valid_dataset = ParallelDataset(valid_src, valid_trg, src_vocab, trg_vocab, max_len=max_len)
    test_dataset = ParallelDataset(test_src, test_trg, src_vocab, trg_vocab, max_len=max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return src_vocab, trg_vocab, train_loader, valid_loader, test_loader