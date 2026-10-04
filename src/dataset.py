"""Parallel data loading, preprocessing, padding and DataLoader construction."""

import os
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


def clean_text(text, lowercase=True):
    """Basic preprocessing: strip, collapse whitespace, optionally lowercase."""
    text = " ".join(text.strip().split())
    return text.lower() if lowercase else text


def load_parallel_file(src_path, tgt_path, lowercase=True):
    """Load a parallel corpus from two text files (one sentence per line)."""
    with open(src_path, encoding="utf-8") as f:
        src_lines = [clean_text(line, lowercase) for line in f if line.strip()]
    with open(tgt_path, encoding="utf-8") as f:
        tgt_lines = [clean_text(line, lowercase) for line in f if line.strip()]
    assert len(src_lines) == len(tgt_lines), "source/target line counts differ"
    return src_lines, tgt_lines


def create_demo_dataset(raw_dir, train_n=2000, val_n=200, seed=42):
    """Generate a tiny parallel corpus for the word-reversal Seq2Seq task."""
    random.seed(seed)
    words = ["i", "you", "we", "they", "love", "like", "hate", "study", "build",
             "ai", "models", "robots", "code", "music", "pizza", "math", "very",
             "much", "today", "together", "slowly", "the", "a", "big", "small"]
    os.makedirs(raw_dir, exist_ok=True)

    def sample_sentence():
        n = random.randint(2, 8)
        return " ".join(random.choices(words, k=n))

    pairs = [(s, " ".join(reversed(s.split())))
             for s in (sample_sentence() for _ in range(train_n + val_n))]
    splits = {"train": pairs[:train_n], "val": pairs[train_n:]}
    for name, data in splits.items():
        for lang, col in (("src", 0), ("tgt", 1)):
            path = os.path.join(raw_dir, f"{name}.{lang}")
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(pair[col] for pair in data) + "\n")
    return splits


class TranslationDataset(Dataset):
    """Pairs of source/target sentences converted to id tensors."""

    def __init__(self, src_texts, tgt_texts, src_tokenizer, tgt_tokenizer, max_len):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        # Source: raw token ids. Target: wrapped in <SOS> ... <EOS>.
        src_ids = self.src_tokenizer.encode(self.src_texts[idx], add_sos_eos=False,
                                            max_len=self.max_len)
        tgt_ids = self.tgt_tokenizer.encode(self.tgt_texts[idx], add_sos_eos=True,
                                            max_len=self.max_len)
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


def make_collate_fn(pad_id):
    """Pad variable-length sequences with <PAD>; padding positions feed the masks."""

    def collate(batch):
        src, tgt = zip(*batch)
        src = pad_sequence(src, batch_first=True, padding_value=pad_id)
        tgt = pad_sequence(tgt, batch_first=True, padding_value=pad_id)
        return src, tgt

    return collate


def build_dataloaders(config, src_tokenizer, tgt_tokenizer):
    """Build train/validation DataLoaders from the configured data paths."""
    data_cfg = config["data"]
    max_len = config["model"]["max_seq_len"]
    train_src, train_tgt = load_parallel_file(data_cfg["train_src"], data_cfg["train_tgt"])
    val_src, val_tgt = load_parallel_file(data_cfg["val_src"], data_cfg["val_tgt"])

    train_ds = TranslationDataset(train_src, train_tgt, src_tokenizer, tgt_tokenizer, max_len)
    val_ds = TranslationDataset(val_src, val_tgt, src_tokenizer, tgt_tokenizer, max_len)

    batch_size = config["training"]["batch_size"]
    collate = make_collate_fn(src_tokenizer.pad_id)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    return train_loader, val_loader