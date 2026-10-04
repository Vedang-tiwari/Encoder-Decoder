"""Autoregressive generation with greedy decoding.

Source -> encoder -> <SOS> -> decoder -> next-token argmax -> append ->
decoder again ... until <EOS> or max length, then ids -> text.
"""

import torch


@torch.no_grad()
def greedy_decode(model, src_ids, src_tokenizer, tgt_tokenizer, device, max_len):
    """Generate one target sequence from pre-tokenized source ids."""
    model.eval()
    src = torch.tensor([src_ids], dtype=torch.long, device=device)          # (1, src_len)
    src_mask = model.make_src_mask(src, src_tokenizer.pad_id)
    enc_out = model.encode(src, src_mask)

    ys = torch.tensor([[tgt_tokenizer.sos_id]], dtype=torch.long, device=device)
    for _ in range(max_len):
        tgt_mask = model.make_tgt_mask(ys, tgt_tokenizer.pad_id)
        dec_out = model.decode(ys, enc_out, src_mask, tgt_mask)
        next_id = model.output_proj(dec_out[:, -1]).argmax(dim=-1).item()
        if next_id == tgt_tokenizer.eos_id:
            break
        ys = torch.cat([ys, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
    return ys.squeeze(0).tolist()


def translate(model, sentence, src_tokenizer, tgt_tokenizer, device, max_len=64):
    """End-to-end: raw source sentence -> generated target sentence."""
    src_ids = src_tokenizer.encode(sentence, add_sos_eos=False, max_len=max_len)
    out_ids = greedy_decode(model, src_ids, src_tokenizer, tgt_tokenizer, device, max_len)
    return tgt_tokenizer.decode(out_ids, skip_special=True)