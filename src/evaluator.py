"""Evaluation on held-out data: loss, perplexity and BLEU."""

import math
from collections import Counter

import torch

from src.inference import greedy_decode


@torch.no_grad()
def compute_loss(model, data_loader, criterion, device, tgt_pad_idx):
    """Token-averaged cross-entropy loss over the dataset."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for src, tgt in data_loader:
        src, tgt = src.to(device), tgt.to(device)
        decoder_in, target = tgt[:, :-1], tgt[:, 1:]
        logits = model(src, decoder_in)
        loss = criterion(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        ntok = target.ne(tgt_pad_idx).sum().item()
        total_loss += loss.item() * ntok
        total_tokens += ntok
    return total_loss / max(1, total_tokens)


def _ngrams(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(references, hypotheses, max_n=4):
    """Simple corpus BLEU with add-one smoothing (no external dependencies)."""
    matches = [0] * max_n
    totals = [0] * max_n
    ref_len = hyp_len = 0
    for ref, hyp in zip(references, hypotheses):
        ref_t, hyp_t = ref.split(), hyp.split()
        ref_len += len(ref_t)
        hyp_len += len(hyp_t)
        for n in range(1, max_n + 1):
            ref_ngrams = _ngrams(ref_t, n)
            for gram, cnt in _ngrams(hyp_t, n).items():
                totals[n - 1] += cnt
                matches[n - 1] += min(cnt, ref_ngrams.get(gram, 0))
    if hyp_len == 0:
        return 0.0
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / max(1, hyp_len))
    log_precision = 0.0
    for n in range(max_n):
        if totals[n] == 0:
            return 0.0
        log_precision += math.log((matches[n] + 1) / (totals[n] + 1))
    return bp * math.exp(log_precision / max_n)


@torch.no_grad()
def evaluate(model, data_loader, device, tgt_pad_idx, src_tokenizer, tgt_tokenizer,
             max_decode_len=64, criterion=None, bleu_samples=None):
    """Compute loss/perplexity and (greedy-decoded) BLEU on a dataset."""
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=tgt_pad_idx)

    loss = compute_loss(model, data_loader, criterion, device, tgt_pad_idx)

    # Greedy-decode a subset (or all) of the validation set for BLEU.
    dataset = data_loader.dataset
    n = bleu_samples if bleu_samples is not None else len(dataset)
    references, hypotheses = [], []
    for idx in range(min(n, len(dataset))):
        src_tensor, tgt_tensor = dataset[idx]
        ref_text = tgt_tokenizer.decode(tgt_tensor.tolist(), skip_special=True)
        pred_ids = greedy_decode(model, src_tensor.tolist(), src_tokenizer,
                                 tgt_tokenizer, device, max_decode_len)
        references.append(ref_text)
        hypotheses.append(tgt_tokenizer.decode(pred_ids, skip_special=True))
    bleu = corpus_bleu(references, hypotheses)
    return {"loss": loss, "perplexity": math.exp(min(20.0, loss)), "bleu": bleu}


def load_model(checkpoint_path, device, src_pad_idx=0, tgt_pad_idx=0):
    """Rebuild a TransformerSeq2Seq from a saved checkpoint."""
    from src.model import TransformerSeq2Seq

    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:  # older torch without weights_only
        ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = dict(ckpt["model_config"])
    cfg.update(src_pad_idx=src_pad_idx, tgt_pad_idx=tgt_pad_idx)
    model = TransformerSeq2Seq(**cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval(), ckpt