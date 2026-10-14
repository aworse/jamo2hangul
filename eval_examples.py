"""Hand-made jamo->Hangul test set + performance report.

Runs the trained model on a curated set of examples that mirror the target distribution
(single words, proper nouns/addresses, full sentences, digit/latin-mixed text) on BOTH
clean jamo input and NOISY jamo input (a simulated imperfect upstream), then writes a
readable report and a JSON summary. A separate, non-scored section shows spacing-ambiguous
inputs where more than one spacing is linguistically valid.

  python eval_examples.py --dir artifacts/run                 # clean + noisy, GPU if available
  python eval_examples.py --dir artifacts/run --device cpu    # force CPU (won't touch a busy GPU)
  python eval_examples.py --dir artifacts/run --ckpt last.pt  # a specific checkpoint
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from train import load_for_infer, edits
from data import PAD
from jamo_text import to_jamo, corrupt

# Scored examples, grouped by the category of input the competition is expected to contain.
CASES: dict[str, list[str]] = {
    "단어": ["남자", "닮음", "삼각형", "사랑", "컴퓨터", "떡볶이", "넓이", "꽃", "값어치", "읽었다"],
    "주소·고유명사": ["부산광역시 해운대구 좌동", "서울특별시 강남구 테헤란로",
                 "경기도 성남시 분당구", "대한민국 국회의사당", "인천국제공항"],
    "문장": ["나는 천개의 바람 천개의 사람이 되었다", "오늘 날씨가 맑아서 기분이 좋다",
            "학생이 도서관에서 책을 읽는다", "우리는 내일 함께 산에 오를 것이다",
            "그는 조용히 문을 닫고 밖으로 나갔다"],
    "숫자·영문 혼합": ["2026년 10월 14일", "PJ그룹은 3개월 안에 서비스를 만든다",
                 "CTO 박부건", "AI 모델 성능 평가", "코로나19 이후의 변화"],
}

# Non-scored: same jamo stream maps to multiple valid spacings; we only show what the model picks.
AMBIGUOUS = ["한글", "한 글", "아버지가 방에 들어가신다", "무지개 색깔", "서울 시청 앞"]

NOISE = (0.05, 0.03, 0.03)  # p_sub, p_del, p_ins — a modest imperfect-upstream simulation


def predict(model, vocab, device, jamo_lines: list[str], max_len: int) -> list[str]:
    """Batched greedy decode, mirroring train.cmd_infer."""
    out: list[str] = []
    bs = 64
    for i in range(0, len(jamo_lines), bs):
        chunk = jamo_lines[i:i + bs]
        src = [torch.tensor(vocab.enc_src(j)[:max_len]) for j in chunk]
        mx = max((len(s) for s in src), default=1)
        sp = torch.full((len(src), mx), PAD, dtype=torch.long)
        for k, s in enumerate(src):
            sp[k, :len(s)] = s
        out += [vocab.dec_tgt(o) for o in model.greedy(sp.to(device), max_len=max_len)]
    return out


def score(refs: list[str], hyps: list[str]) -> dict:
    ce = cn = exact = 0
    for r, h in zip(refs, hyps):
        ce += edits(r, h)
        cn += len(r)
        exact += (h == r)
    return {"cer": ce / max(1, cn), "exact": exact / max(1, len(refs)), "n": len(refs),
            "char_err": ce, "char_tot": cn}


def fmt_block(lines: list[tuple[str, str, str, str]]) -> list[str]:
    """lines: (ref, jamo_in, hyp, tag)."""
    o = []
    for ref, jin, hyp, tag in lines:
        o.append(f"  정답 : {ref}")
        o.append(f"  입력 : {jin}")
        o.append(f"  예측 : {hyp}   {tag}")
        o.append("")
    return o


def run(dirp: Path, ckpt: str, device_name: str, max_len: int):
    device = torch.device(device_name if device_name != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, vocab, step = load_for_infer(dirp, ckpt, device)

    rng = random.Random(1234)
    report: list[str] = []
    summary = {"ckpt": ckpt, "step": step, "device": str(device), "noise": NOISE, "categories": {}}

    def hdr(s):
        report.append(s)

    hdr(f"jamo->Hangul 성능 평가  (ckpt={ckpt}, step={step}, device={device})")
    hdr(f"생성: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    hdr("=" * 70)
    hdr("지표:  exact = 정답 문자열 완전 일치(띄어쓰기 포함),  CER = 문자 오류율(낮을수록 좋음)")
    hdr("")

    agg_clean_refs, agg_clean_hyps = [], []
    agg_noisy_refs, agg_noisy_hyps = [], []

    for cat, refs in CASES.items():
        clean_in = [to_jamo(r) for r in refs]
        noisy_in = [corrupt(j, *NOISE, rng) for j in clean_in]
        clean_hyp = predict(model, vocab, device, clean_in, max_len)
        noisy_hyp = predict(model, vocab, device, noisy_in, max_len)
        cs, ns = score(refs, clean_hyp), score(refs, noisy_hyp)
        summary["categories"][cat] = {"clean": cs, "noisy": ns}
        agg_clean_refs += refs; agg_clean_hyps += clean_hyp
        agg_noisy_refs += refs; agg_noisy_hyps += noisy_hyp

        hdr("#" * 70)
        hdr(f"[{cat}]  n={cs['n']}   깨끗한 입력: exact {100*cs['exact']:.0f}%  CER {100*cs['cer']:.1f}%"
            f"   |   오류 섞인 입력: exact {100*ns['exact']:.0f}%  CER {100*ns['cer']:.1f}%")
        hdr("#" * 70)
        hdr("· 깨끗한 입력")
        rows = [(r, j, h, "✓" if h == r else f"✗ CER {100*edits(r, h)/max(1,len(r)):.0f}%")
                for r, j, h in zip(refs, clean_in, clean_hyp)]
        report += fmt_block(rows)
        hdr("· 오류 섞인 입력 (upstream 오류 시뮬레이션)")
        rows = [(r, j, h, "✓" if h == r else f"✗ CER {100*edits(r, h)/max(1,len(r)):.0f}%")
                for r, j, h in zip(refs, noisy_in, noisy_hyp)]
        report += fmt_block(rows)

    oc, on_ = score(agg_clean_refs, agg_clean_hyps), score(agg_noisy_refs, agg_noisy_hyps)
    summary["overall"] = {"clean": oc, "noisy": on_}

    # Non-scored ambiguity demo
    hdr("#" * 70)
    hdr("[띄어쓰기 모호성 시연 — 채점 제외]  같은 자모가 여러 띄어쓰기로 유효함")
    hdr("#" * 70)
    amb_in = [to_jamo(r) for r in AMBIGUOUS]
    amb_hyp = predict(model, vocab, device, amb_in, max_len)
    for r, j, h in zip(AMBIGUOUS, amb_in, amb_hyp):
        hdr(f"  입력 : {j}")
        hdr(f"  예측 : {h}   (참고 원문: {r})")
        hdr("")
    summary["ambiguous"] = [{"ref": r, "in": j, "hyp": h} for r, j, h in zip(AMBIGUOUS, amb_in, amb_hyp)]

    hdr("=" * 70)
    hdr("종합")
    hdr(f"  깨끗한 입력  : exact {100*oc['exact']:.1f}%   CER {100*oc['cer']:.2f}%   (n={oc['n']})")
    hdr(f"  오류 섞인 입력: exact {100*on_['exact']:.1f}%   CER {100*on_['cer']:.2f}%   (n={on_['n']})"
        f"   [noise p_sub/del/ins={NOISE}]")
    hdr("=" * 70)
    hdr("참고: 현재 모델은 노이즈 없이(깨끗한 입력) 학습됨. '오류 섞인 입력' 수치는 오류 교정")
    hdr("      미학습 상태의 성능이며, --noise 재학습 또는 대회 데이터 파인튜닝으로 개선 가능.")

    text = "\n".join(report) + "\n"
    (dirp / "examples_report.txt").write_text(text, encoding="utf-8")
    (dirp / "examples_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(text)
    print(f"[written] {dirp/'examples_report.txt'}  and  examples_report.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-len", type=int, default=256)
    a = ap.parse_args()
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    run(Path(a.dir), a.ckpt, a.device, a.max_len)


if __name__ == "__main__":
    main()
