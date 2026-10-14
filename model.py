"""Character-level Transformer encoder-decoder for jamo->Hangul restoration.

Thin wrapper over torch.nn.Transformer with learned positional embeddings and padding/causal masks.
Default size ~45M params: fits an 8 GB GPU with bf16 and a moderate token budget.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from data import PAD, BOS, EOS


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        self.pos = nn.Embedding(max_len, d_model)

    def forward(self, x):  # x: (B, T, D)
        t = torch.arange(x.size(1), device=x.device)
        return x + self.pos(t)[None]


class Seq2Seq(nn.Module):
    def __init__(self, n_src: int, n_tgt: int, d_model: int = 512, nhead: int = 8,
                 enc_layers: int = 6, dec_layers: int = 6, dim_ff: int = 2048, dropout: float = 0.1,
                 max_len: int = 1024):
        super().__init__()
        self.d_model = d_model
        self.src_emb = nn.Embedding(n_src, d_model, padding_idx=PAD)
        self.tgt_emb = nn.Embedding(n_tgt, d_model, padding_idx=PAD)
        self.src_pos = PositionalEmbedding(d_model, max_len)
        self.tgt_pos = PositionalEmbedding(d_model, max_len)
        self.transformer = nn.Transformer(d_model, nhead, enc_layers, dec_layers, dim_ff, dropout,
                                          batch_first=True, norm_first=True)
        self.out = nn.Linear(d_model, n_tgt)
        self.out.weight = self.tgt_emb.weight  # tie output/embedding
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear) and m.weight.requires_grad:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
            with torch.no_grad():
                m.weight[PAD].zero_()

    def _embed(self, ids, emb, pos):
        return pos(emb(ids) * math.sqrt(self.d_model))

    @staticmethod
    def _causal(n, device):  # bool mask (True = disallowed), matches bool padding masks
        return torch.triu(torch.ones(n, n, dtype=torch.bool, device=device), diagonal=1)

    def encode(self, src):
        return self.transformer.encoder(self._embed(src, self.src_emb, self.src_pos),
                                        src_key_padding_mask=src.eq(PAD))

    def forward(self, src, tgt_in):
        mem = self.encode(src)
        causal = self._causal(tgt_in.size(1), src.device)
        dec = self.transformer.decoder(self._embed(tgt_in, self.tgt_emb, self.tgt_pos), mem,
                                       tgt_mask=causal, tgt_key_padding_mask=tgt_in.eq(PAD),
                                       memory_key_padding_mask=src.eq(PAD))
        return self.out(dec)

    @torch.no_grad()
    def greedy(self, src, max_len: int = 256):
        """Batched greedy decode. src: (B, S). Returns list of id lists (without BOS/EOS)."""
        self.eval()
        mem = self.encode(src)
        pad = src.eq(PAD)
        B = src.size(0)
        ys = torch.full((B, 1), BOS, dtype=torch.long, device=src.device)
        done = torch.zeros(B, dtype=torch.bool, device=src.device)
        outs = [[] for _ in range(B)]
        for _ in range(max_len):
            causal = self._causal(ys.size(1), src.device)
            dec = self.transformer.decoder(self._embed(ys, self.tgt_emb, self.tgt_pos), mem,
                                           tgt_mask=causal, memory_key_padding_mask=pad)
            nxt = self.out(dec[:, -1]).argmax(-1)
            ys = torch.cat([ys, nxt[:, None]], 1)
            for i in range(B):
                if not done[i]:
                    if nxt[i].item() == EOS:
                        done[i] = True
                    else:
                        outs[i].append(int(nxt[i].item()))
            if bool(done.all()):
                break
        return outs


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
