"""Large-scale WORD-level accuracy for jamo->Hangul (clean input).

Pulls unique words from the held-out dev set, decodes each from its clean jamo, and reports
exact-match word accuracy + CER, broken down by word length, with sample failures.

  python word_eval.py --dir artifacts/run --ckpt best.pt            # clean model
  python word_eval.py --dir artifacts/run_noise --ckpt last.pt      # noise model
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import torch

from train import load_for_infer, edits
from data import PAD, read_corpus
from jamo_text import to_jamo, is_syllable

HANGUL = re.compile(r"[가-힣]")


def words_from_dev(dirp: Path, limit: int, min_len: int, max_len: int) -> list[str]:
    lines = read_corpus(dirp / "dev.txt", max_len=200)
    seen, out = set(), []
    for ln in lines:
        for w in ln.split():
            w = w.strip(".,!?\"'()[]{}<>·…“”‘’:;")
            if not (min_len <= len(w) <= max_len):
                continue
            if not HANGUL.search(w):          # must contain Hangul (skip pure numbers/latin)
                continue
            if w in seen:
                continue
            seen.add(w)
            out.append(w)
            if len(out) >= limit:
                return out
    return out


def predict(model, vocab, device, jamo_lines, max_len, bs=128):
    out = []
    for i in range(0, len(jamo_lines), bs):
        chunk = jamo_lines[i:i + bs]
        src = [torch.tensor(vocab.enc_src(j)[:max_len]) for j in chunk]
        mx = max((len(s) for s in src), default=1)
        sp = torch.full((len(src), mx), PAD, dtype=torch.long)
        for k, s in enumerate(src):
            sp[k, :len(s)] = s
        out += [vocab.dec_tgt(o) for o in model.greedy(sp.to(device), max_len=max_len)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--min-len", type=int, default=1)
    ap.add_argument("--max-len", type=int, default=12)
    a = ap.parse_args()
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    dirp = Path(a.dir)
    device = torch.device(a.device if a.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, vocab, step = load_for_infer(dirp, a.ckpt, device)

    words = words_from_dev(dirp, a.n, a.min_len, a.max_len)
    jamo = [to_jamo(w) for w in words]
    t0 = time.time()
    hyp = predict(model, vocab, device, jamo, max_len=64)
    dt = time.time() - t0

    ce = cn = exact = 0
    by_len = defaultdict(lambda: [0, 0])   # syllable-length -> [correct, total]
    fails = []
    for w, h in zip(words, hyp):
        ok = (h == w)
        exact += ok
        ce += edits(w, h); cn += len(w)
        L = sum(1 for c in w if is_syllable(c))
        by_len[L][1] += 1
        by_len[L][0] += ok
        if not ok and len(fails) < 40:
            fails.append((w, h, to_jamo(w)))

    n = len(words)
    rep = []
    rep.append(f"WORD-level 정확도  (ckpt={a.ckpt}, step={step}, device={device})")
    rep.append(f"생성: {time.strftime('%Y-%m-%d %H:%M:%S')}   |   소스: dev.txt 고유단어   |   디코딩 {dt:.1f}s")
    rep.append("=" * 64)
    rep.append(f"단어 수: {n:,}")
    rep.append(f"완전일치(word accuracy): {100*exact/max(1,n):.2f}%   ({exact:,}/{n:,})")
    rep.append(f"문자오류율(CER): {100*ce/max(1,cn):.2f}%")
    rep.append("")
    rep.append("길이(음절)별 정확도:")
    for L in sorted(by_len):
        c, t = by_len[L]
        rep.append(f"  {L:>2}음절: {100*c/max(1,t):6.2f}%   ({c:,}/{t:,})")
    rep.append("")
    rep.append(f"틀린 예시 (최대 40개):")
    for w, h, j in fails:
        rep.append(f"  정답 {w}  |  입력 {j}  |  예측 {h}")

    text = "\n".join(rep) + "\n"
    (dirp / "word_report.txt").write_text(text, encoding="utf-8")
    (dirp / "word_report.json").write_text(json.dumps(
        {"ckpt": a.ckpt, "step": step, "n": n, "word_acc": exact / max(1, n),
         "cer": ce / max(1, cn), "by_len": {str(L): by_len[L] for L in by_len},
         "fails": [{"ref": w, "hyp": h} for w, h, _ in fails]},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(text)
    print(f"[written] {dirp/'word_report.txt'}")


if __name__ == "__main__":
    main()
