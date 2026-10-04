"""Word-level tokenizer: vocab building, token<->id conversion, special tokens."""

import json
import os
from collections import Counter


class Tokenizer:
    """Simple word-level tokenizer with the four required special tokens."""

    PAD = "<PAD>"
    SOS = "<SOS>"
    EOS = "<EOS>"
    UNK = "<UNK>"

    def __init__(self):
        self.token2id = {}
        self.id2token = []

    # ------------------------------------------------------------------ vocab
    @staticmethod
    def tokenize(text, lowercase=True):
        if lowercase:
            text = text.lower()
        return text.split()

    def build_vocab(self, sentences, min_freq=1, lowercase=True):
        """Build vocabulary from a list of sentences."""
        counter = Counter()
        for sent in sentences:
            counter.update(self.tokenize(sent, lowercase))
        vocab = [self.PAD, self.SOS, self.EOS, self.UNK]
        vocab += sorted(tok for tok, freq in counter.items() if freq >= min_freq)
        self.id2token = vocab
        self.token2id = {tok: i for i, tok in enumerate(vocab)}
        return self

    # ------------------------------------------------------------------- ids
    @property
    def pad_id(self):
        return self.token2id[self.PAD]

    @property
    def sos_id(self):
        return self.token2id[self.SOS]

    @property
    def eos_id(self):
        return self.token2id[self.EOS]

    @property
    def unk_id(self):
        return self.token2id[self.UNK]

    @property
    def vocab_size(self):
        return len(self.id2token)

    def encode(self, text, add_sos_eos=True, max_len=None, lowercase=True):
        """Text -> list of token ids (optionally truncated and wrapped with SOS/EOS)."""
        ids = [self.token2id.get(tok, self.unk_id)
               for tok in self.tokenize(text, lowercase)]
        if max_len is not None:
            budget = max_len - (2 if add_sos_eos else 0)
            ids = ids[:budget]
        if add_sos_eos:
            ids = [self.sos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids, skip_special=True):
        """List of token ids -> text."""
        specials = {self.pad_id, self.sos_id, self.eos_id, self.unk_id}
        tokens = []
        for idx in ids:
            if skip_special and idx in specials:
                continue
            if 0 <= idx < len(self.id2token):
                tokens.append(self.id2token[idx])
        return " ".join(tokens)

    # ------------------------------------------------------------ persistence
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.id2token, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        tok = cls()
        with open(path, encoding="utf-8") as f:
            tok.id2token = json.load(f)
        tok.token2id = {t: i for i, t in enumerate(tok.id2token)}
        return tok