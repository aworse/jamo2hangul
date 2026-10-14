"""Train / evaluate / run the jamo->Hangul Transformer.

  python train.py prepare --corpus sentences.txt --dir run [--dev 3000] [--min-count 2] [--max-lines N]
  python train.py train   --dir run --device cuda [--steps 120000] [--batch-tokens 12000] [--noise 0,0,0] [--resume]
  python train.py eval    --dir run [--ckpt best.pt]
  python train.py infer   --dir run --in jamo.txt --out hangul.txt [--ckpt best.pt] [--beam 1]

Metrics: dev exact-match (full-string accuracy, incl. spacing) and CER (char edit rate incl. spaces).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import Vocab, J2HDataset, collate, read_corpus, PAD
from model import Seq2Seq, count_params
from jamo_text import to_jamo


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def edits(ref, hyp):
    n, m = len(ref), len(hyp)
    if not n:
        return m
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ref[i - 1] != hyp[j - 1]))
        prev = cur
    return prev[m]


class TokenBatchSampler:
    """Length-bucketed batches under a token budget (batch_len * max_seq_len <= budget)."""
    def __init__(self, lengths, budget, shuffle=True, seed=0):
        self.lengths, self.budget, self.shuffle, self.seed, self.epoch = lengths, budget, shuffle, seed, 0

    def _build(self):
        idx = list(range(len(self.lengths)))
        rng = random.Random(self.seed + self.epoch)
        if self.shuffle:
            rng.shuffle(idx)
        batches, mega = [], 50 * 256
        for s in range(0, len(idx), mega):
            chunk = sorted(idx[s:s + mega], key=lambda j: self.lengths[j])
            batch, mx = [], 0
            for j in chunk:
                nmx = max(mx, self.lengths[j])
                if batch and nmx * (len(batch) + 1) > self.budget:
                    batches.append(batch)
                    batch, mx = [j], self.lengths[j]
                else:
                    batch.append(j)
                    mx = nmx
            if batch:
                batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def __iter__(self):
        b = self._build()
        self.epoch += 1
        return iter(b)

    def __len__(self):
        return max(1, sum(self.lengths) // self.budget)


def lr_at(step, warmup, total, base, min_ratio=0.05):
    if step < warmup:
        return base * step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return base * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))


def device_of(name):
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda but CUDA is not available")
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def make_model(cfg, vocab):
    return Seq2Seq(len(vocab.src_itos), len(vocab.tgt_itos), d_model=cfg["d_model"], nhead=cfg["nhead"],
                   enc_layers=cfg["enc_layers"], dec_layers=cfg["dec_layers"], dim_ff=cfg["dim_ff"],
                   dropout=cfg["dropout"], max_len=cfg["max_len"])


# ----------------------------------------------------------------------------- prepare
def cmd_prepare(a):
    d = Path(a.dir)
    d.mkdir(parents=True, exist_ok=True)
    texts = read_corpus(a.corpus, max_lines=a.max_lines, max_len=a.max_text_len)
    if len(texts) < a.dev + 100:
        raise SystemExit(f"corpus too small: {len(texts)} lines")
    rng = random.Random(0)
    rng.shuffle(texts)
    dev, train = texts[:a.dev], texts[a.dev:]
    vocab = Vocab.build(train, min_tgt_count=a.min_count)
    vocab.save(d / "vocab.json")
    (d / "train.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (d / "dev.txt").write_text("\n".join(dev) + "\n", encoding="utf-8")
    json.dump({"train": len(train), "dev": len(dev), "src_vocab": len(vocab.src_itos),
               "tgt_vocab": len(vocab.tgt_itos), "corpus": str(Path(a.corpus).resolve())},
              open(d / "prepare.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log(f"prepare: train {len(train):,} dev {len(dev):,} | src_vocab {len(vocab.src_itos)} tgt_vocab {len(vocab.tgt_itos)}")


# ----------------------------------------------------------------------------- train
def evaluate(model, dev_texts, vocab, device, max_tgt, n=1000):
    model.eval()
    sub = dev_texts[:n]
    ce = cn = exact = 0
    bs = 64
    for i in range(0, len(sub), bs):
        chunk = sub[i:i + bs]
        src = [torch.tensor(vocab.enc_src(to_jamo(t))[:max_tgt]) for t in chunk]
        mx = max(len(s) for s in src)
        sp = torch.full((len(src), mx), PAD, dtype=torch.long)
        for k, s in enumerate(src):
            sp[k, :len(s)] = s
        outs = model.greedy(sp.to(device), max_len=max_tgt)
        for t, o in zip(chunk, outs):
            hyp = vocab.dec_tgt(o)
            ce += edits(t, hyp)
            cn += len(t)
            exact += (hyp == t)
    return {"cer": ce / max(1, cn), "exact": exact / max(1, len(sub)), "n": len(sub)}


def cmd_train(a):
    d = Path(a.dir)
    device = device_of(a.device)
    vocab = Vocab.load(d / "vocab.json")
    cfg = {"d_model": a.d_model, "nhead": a.nhead, "enc_layers": a.layers, "dec_layers": a.layers,
           "dim_ff": a.dim_ff, "dropout": a.dropout, "max_len": a.max_len}
    train_texts = read_corpus(d / "train.txt", max_len=a.max_text_len)
    dev_texts = read_corpus(d / "dev.txt", max_len=a.max_text_len)
    noise = tuple(float(x) for x in a.noise.split(","))
    ds = J2HDataset(train_texts, vocab, max_src=a.max_len, max_tgt=a.max_len, noise=noise, span_aug=a.span_aug)
    lengths = [min(a.max_len, max(len(to_jamo(t)), len(t) + 2)) for t in train_texts]
    sampler = TokenBatchSampler(lengths, a.batch_tokens, shuffle=True, seed=0)
    dl = DataLoader(ds, batch_sampler=sampler, collate_fn=collate, num_workers=a.workers, persistent_workers=a.workers > 0)

    model = make_model(cfg, vocab).to(device)
    log(f"model params: {count_params(model):,} | device {device} | batch-tokens {a.batch_tokens}")
    decay = [p for p in model.parameters() if p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": nodecay, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.98))
    crit = torch.nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)
    use_amp = device.type == "cuda"
    amp_dt = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dt == torch.float16)

    step, best = 0, float("inf")
    (d / "train_log.jsonl").touch()
    if a.resume and (d / "last.pt").exists():
        ck = torch.load(d / "last.pt", map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); step = ck["step"]; best = ck.get("best", best)
        log(f"resumed at step {step}")

    t0 = time.time()
    done = False
    while not done:
        for sp, tp, sl, tl in dl:
            sp, tp = sp.to(device), tp.to(device)
            model.train()
            for g in opt.param_groups:
                g["lr"] = lr_at(step, a.warmup, a.steps, a.lr)
            with torch.autocast(device.type, dtype=amp_dt, enabled=use_amp):
                logits = model(sp, tp[:, :-1])
                loss = crit(logits.reshape(-1, logits.size(-1)).float(), tp[:, 1:].reshape(-1))
            opt.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip); scaler.step(opt); scaler.update()
            else:
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip); opt.step()
            step += 1
            if step % a.log_every == 0:
                sps = step / (time.time() - t0)
                log(f"step {step:>7} loss {loss.item():.3f} lr {opt.param_groups[0]['lr']:.2e} {sps:.1f} it/s")
            if step % a.eval_every == 0 or step >= a.steps:
                mt = evaluate(model, dev_texts, vocab, device, a.max_len, n=a.eval_n)
                log(f"  [dev] step {step} CER {100*mt['cer']:.2f}%  exact {100*mt['exact']:.2f}%  (n={mt['n']})")
                with open(d / "train_log.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps({"step": step, **mt, "loss": loss.item()}, ensure_ascii=False) + "\n")
                ck = {"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "best": best, "cfg": cfg}
                torch.save(ck, d / "last.pt")
                if mt["cer"] < best:
                    best = mt["cer"]; ck["best"] = best; torch.save(ck, d / "best.pt")
                    log(f"  new best CER {100*best:.2f}% -> best.pt")
            if step >= a.steps or (a.max_minutes and (time.time() - t0) / 60 >= a.max_minutes):
                done = True
                break
    log(f"training done at step {step}; best dev CER {100*best:.2f}%")


# ----------------------------------------------------------------------------- eval / infer
def load_for_infer(d, ckpt, device):
    vocab = Vocab.load(d / "vocab.json")
    ck = torch.load(d / ckpt, map_location=device)
    model = make_model(ck["cfg"], vocab).to(device)
    model.load_state_dict(ck["model"]); model.eval()
    return model, vocab, ck["step"]


def cmd_eval(a):
    d = Path(a.dir); device = device_of(a.device)
    model, vocab, step = load_for_infer(d, a.ckpt, device)
    dev = read_corpus(d / "dev.txt", max_len=a.max_text_len)
    mt = evaluate(model, dev, vocab, device, a.max_len, n=a.eval_n or len(dev))
    log(f"eval ({a.ckpt}, step {step}): CER {100*mt['cer']:.2f}%  exact {100*mt['exact']:.2f}%  (n={mt['n']})")
    json.dump(mt, open(d / "eval.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def cmd_infer(a):
    d = Path(a.dir); device = device_of(a.device)
    model, vocab, _ = load_for_infer(d, a.ckpt, device)
    lines = Path(a.inp).read_text(encoding="utf-8").splitlines()
    out = []
    bs = 64
    for i in range(0, len(lines), bs):
        chunk = lines[i:i + bs]
        src = [torch.tensor(vocab.enc_src(to_jamo(ln))[:a.max_len]) for ln in chunk]  # to_jamo handles jamo OR syllable input
        mx = max((len(s) for s in src), default=1)
        sp = torch.full((len(src), mx), PAD, dtype=torch.long)
        for k, s in enumerate(src):
            sp[k, :len(s)] = s
        out += [vocab.dec_tgt(o) for o in model.greedy(sp.to(device), max_len=a.max_len)]
    Path(a.out).write_text("\n".join(out) + "\n", encoding="utf-8")
    log(f"infer: {len(out)} lines -> {a.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("prepare", "train", "eval", "infer"):
        s = sub.add_parser(name)
        s.add_argument("--dir", required=True)
        s.add_argument("--device", default="auto")
        s.add_argument("--max-len", type=int, default=256)
        s.add_argument("--max-text-len", type=int, default=200)
        if name == "prepare":
            s.add_argument("--corpus", required=True)
            s.add_argument("--dev", type=int, default=3000)
            s.add_argument("--min-count", type=int, default=2)
            s.add_argument("--max-lines", type=int, default=None)
        if name == "train":
            s.add_argument("--steps", type=int, default=120000)
            s.add_argument("--warmup", type=int, default=4000)
            s.add_argument("--lr", type=float, default=5e-4)
            s.add_argument("--wd", type=float, default=0.01)
            s.add_argument("--grad-clip", type=float, default=1.0)
            s.add_argument("--batch-tokens", type=int, default=12000)
            s.add_argument("--noise", default="0,0,0", help="p_sub,p_del,p_ins jamo noise")
            s.add_argument("--span-aug", type=float, default=0.3)
            s.add_argument("--d-model", type=int, default=512)
            s.add_argument("--nhead", type=int, default=8)
            s.add_argument("--layers", type=int, default=6)
            s.add_argument("--dim-ff", type=int, default=2048)
            s.add_argument("--dropout", type=float, default=0.1)
            s.add_argument("--eval-every", type=int, default=2000)
            s.add_argument("--eval-n", type=int, default=1500)
            s.add_argument("--log-every", type=int, default=100)
            s.add_argument("--max-minutes", type=float, default=0.0)
            s.add_argument("--workers", type=int, default=2)
            s.add_argument("--resume", action="store_true")
        if name in ("eval", "infer"):
            s.add_argument("--ckpt", default="best.pt")
            s.add_argument("--eval-n", type=int, default=0)
        if name == "infer":
            s.add_argument("--in", dest="inp", required=True)
            s.add_argument("--out", required=True)
            s.add_argument("--beam", type=int, default=1)
    a = ap.parse_args()
    {"prepare": cmd_prepare, "train": cmd_train, "eval": cmd_eval, "infer": cmd_infer}[a.cmd](a)


if __name__ == "__main__":
    main()
