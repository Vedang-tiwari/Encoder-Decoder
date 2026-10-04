"""Encoder-decoder Transformer for sequence-to-sequence learning.

Token Embedding -> Positional Encoding -> Encoder (self-attn + FFN)
-> Encoder Output -> Decoder (masked self-attn + cross-attn + FFN)
-> Linear projection -> vocabulary logits.
"""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):                       # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


def scaled_dot_product_attention(query, key, value, mask=None, dropout=None):
    """Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V.

    mask: broadcastable to (batch, heads, q_len, k_len); 1 = attend, 0 = ignore.
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attn = torch.softmax(scores, dim=-1)
    if dropout is not None:
        attn = dropout(attn)
    return torch.matmul(attn, value), attn


class MultiHeadAttention(nn.Module):
    """Split d_model into heads, attend in parallel, re-combine."""

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        out, self.attn = scaled_dot_product_attention(Q, K, V, mask, self.dropout)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return self.w_o(out)


class PositionwiseFeedForward(nn.Module):
    """Two linear transformations with ReLU in between."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    """Self-attention + Add&Norm, FFN + Add&Norm."""

    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, src_mask)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class DecoderLayer(nn.Module):
    """Masked self-attn + Add&Norm, cross-attn + Add&Norm, FFN + Add&Norm."""

    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, tgt_mask, src_mask):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, enc_out, enc_out, src_mask)))
        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, dropout, max_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.scale = math.sqrt(d_model)

    def forward(self, src, src_mask):
        x = self.pos_encoding(self.embedding(src) * self.scale)
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, dropout, max_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.scale = math.sqrt(d_model)

    def forward(self, tgt, enc_out, tgt_mask, src_mask):
        x = self.pos_encoding(self.embedding(tgt) * self.scale)
        for layer in self.layers:
            x = layer(x, enc_out, tgt_mask, src_mask)
        return self.norm(x)


class TransformerSeq2Seq(nn.Module):
    """Full encoder-decoder Transformer for sequence-to-sequence generation."""

    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=128, num_heads=8,
                 num_encoder_layers=3, num_decoder_layers=3, d_ff=512,
                 dropout=0.1, max_len=64, src_pad_idx=0, tgt_pad_idx=0):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.encoder = Encoder(src_vocab_size, d_model, num_heads, d_ff,
                               num_encoder_layers, dropout, max_len)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_heads, d_ff,
                               num_decoder_layers, dropout, max_len)
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    # ------------------------------------------------------------- masks
    @staticmethod
    def make_src_mask(src, pad_idx):
        """Padding mask: (batch, 1, 1, src_len) — hide padded source positions."""
        return (src != pad_idx).unsqueeze(1).unsqueeze(2)

    @staticmethod
    def make_tgt_mask(tgt, pad_idx):
        """Padding mask AND causal mask: (batch, 1, tgt_len, tgt_len).

        A decoder position can only attend to earlier positions (and itself),
        and never to padded positions.
        """
        pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)              # (B,1,1,L)
        causal_mask = torch.tril(                                            # (L,L)
            torch.ones(tgt.size(1), tgt.size(1), device=tgt.device, dtype=torch.bool)
        )
        return pad_mask & causal_mask

    # ----------------------------------------------------------- forward
    def encode(self, src, src_mask=None):
        if src_mask is None:
            src_mask = self.make_src_mask(src, self.src_pad_idx)
        return self.encoder(src, src_mask)

    def decode(self, tgt, enc_out, src_mask, tgt_mask=None):
        if tgt_mask is None:
            tgt_mask = self.make_tgt_mask(tgt, self.tgt_pad_idx)
        return self.decoder(tgt, enc_out, tgt_mask, src_mask)

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src, self.src_pad_idx)
        tgt_mask = self.make_tgt_mask(tgt, self.tgt_pad_idx)
        enc_out = self.encoder(src, src_mask)
        dec_out = self.decoder(tgt, enc_out, tgt_mask, src_mask)
        return self.output_proj(dec_out)