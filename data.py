"""Vocab + torch Dataset for jamo->Hangul restoration.

A target corpus (one Korean sentence per line) drives everything: each line yields a training pair
(jamo stream -> text). Pairs are generated on the fly so per-epoch noise and span augmentation vary.
Char-level vocab on both sides: tiny source side (~51 jamo + punctuation), target side = syllables seen.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from jamo_text import JAMO_SET, make_pair, corrupt, to_jamo

PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]


class Vocab:
    def __init__(self, src_itos: list[str], tgt_itos: list[str]):
        self.src_itos, self.tgt_itos = src_itos, tgt_itos
        self.src_stoi = {c: i for i, c in enumerate(src_itos)}
        self.tgt_stoi = {c: i for i, c in enumerate(tgt_itos)}

    def enc_src(self, jamo: str) -> list[int]:
        s = self.src_stoi
        return [s.get(c, UNK) for c in jamo]

    def enc_tgt(self, text: str) -> list[int]:
        s = self.tgt_stoi
        return [BOS] + [s.get(c, UNK) for c in text] + [EOS]

    def dec_tgt(self, ids: list[int]) -> str:
        out = []
        for i in ids:
            if i == EOS:
                break
            if i >= len(SPECIALS):
                out.append(self.tgt_itos[i])
        return "".join(out)

    def save(self, path) -> None:
        Path(path).write_text(json.dumps({"src": self.src_itos, "tgt": self.tgt_itos}, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "Vocab":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(d["src"], d["tgt"])

    @classmethod
    def build(cls, texts, min_tgt_count: int = 2) -> "Vocab":
        from collections import Counter
        src, tgt = set(JAMO_SET), Counter()
        for t in texts:
            for c in to_jamo(t):            # source-side passthrough chars (digits, latin, punct)
                if c not in JAMO_SET:
                    src.add(c)
            for c in t:                     # target-side characters
                tgt[c] += 1
        src_itos = SPECIALS + sorted(src)
        tgt_itos = SPECIALS + [c for c, n in sorted(tgt.items(), key=lambda kv: (-kv[1], kv[0])) if n >= min_tgt_count]
        return cls(src_itos, tgt_itos)


class J2HDataset(Dataset):
    def __init__(self, texts: list[str], vocab: Vocab, *, max_src: int = 256, max_tgt: int = 256,
                 noise=(0.0, 0.0, 0.0), span_aug: float = 0.3, seed: int = 0):
        self.texts, self.vocab = texts, vocab
        self.max_src, self.max_tgt = max_src, max_tgt
        self.p_sub, self.p_del, self.p_ins = noise
        self.span_aug = span_aug
        self.base_seed = seed

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        rng = random.Random(self.base_seed * 1_000_003 + idx)
        text = self.texts[idx]
        if self.span_aug and " " in text and rng.random() < self.span_aug:  # teach single-word / phrase inputs too
            words = text.split(" ")
            a = rng.randrange(len(words))
            b = rng.randint(a, min(len(words) - 1, a + rng.randint(0, 4)))
            text = " ".join(words[a:b + 1])
        src_j, tgt_t = make_pair(text)
        src_j = corrupt(src_j, self.p_sub, self.p_del, self.p_ins, rng)
        src = self.vocab.enc_src(src_j)[: self.max_src]
        tgt = self.vocab.enc_tgt(tgt_t)[: self.max_tgt]
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


def collate(batch):
    srcs, tgts = zip(*batch)
    sl = torch.tensor([len(s) for s in srcs])
    tl = torch.tensor([len(t) for t in tgts])
    sm = max(1, int(sl.max()))
    tm = max(1, int(tl.max()))
    sp = torch.full((len(batch), sm), PAD, dtype=torch.long)
    tp = torch.full((len(batch), tm), PAD, dtype=torch.long)
    for i, (s, t) in enumerate(zip(srcs, tgts)):
        sp[i, : len(s)] = s
        tp[i, : len(t)] = t
    return sp, tp, sl, tl


def read_corpus(path, max_lines: int | None = None, min_len: int = 1, max_len: int = 200) -> list[str]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if min_len <= len(s) <= max_len:
                out.append(s)
                if max_lines and len(out) >= max_lines:
                    break
    return out
