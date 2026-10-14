"""jamo->Hangul driver (Windows / Linux, NVIDIA GPU): corpus -> prepare -> train -> eval, resumable.

  python run_j2h.py                 # everything (setup, corpus download, train, eval)
  python run_j2h.py --corpus my.txt # use your own Korean sentences (one per line), skip the wiki download
  python run_j2h.py --status
  python run_j2h.py --steps 60000   # shorter training

Each stage drops a marker in artifacts/done/; re-running continues where it stopped (training resumes
from last.pt). Logs: artifacts/logs/<stage>.log. The trained model is artifacts/run/best.pt; run
inference on the competition's jamo file with:
  .venv\\Scripts\\python train.py infer --dir artifacts/run --in jamo.txt --out hangul.txt
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
import socket
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WIN = os.name == "nt"
socket.setdefaulttimeout(60)  # hard cap so DNS/connect/read never hang a stage forever
VENV = ROOT / ".venv"
VPY = VENV / ("Scripts/python.exe" if WIN else "bin/python")
ART = ROOT / "artifacts"
_KW = "kowiki/latest/kowiki-latest-pages-articles.xml.bz2"
KOWIKI_MIRRORS = [
    f"https://dumps.wikimedia.org/{_KW}",
    f"https://dumps.wikimedia.your.org/{_KW}",
    f"https://ftp.acc.umu.se/mirror/wikimedia.org/dumps/{_KW}",
    f"https://wikimedia.bytemark.co.uk/{_KW}",
]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def env():
    return dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", PYTHONPATH=str(ROOT))


def run(cmd, stage, py=None):
    cmd = [str(py), *map(str, cmd)] if py else list(map(str, cmd))
    (ART / "logs").mkdir(parents=True, exist_ok=True)
    logf = ART / "logs" / f"{stage}.log"
    log("$ " + " ".join(cmd))
    with open(logf, "a", encoding="utf-8") as fo:
        fo.write(f"\n### {time.ctime()}\n$ {' '.join(cmd)}\n")
        p = subprocess.Popen(cmd, cwd=ROOT, env=env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for raw in p.stdout:
            line = raw.decode("utf-8", "replace")
            sys.stdout.write(line); sys.stdout.flush(); fo.write(line)
        if p.wait():
            raise SystemExit(f"{stage}: failed (log: {logf})")


def keep_awake():
    if WIN:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)


def ram_gb():
    if WIN:
        import ctypes
        from ctypes import wintypes

        class MS(ctypes.Structure):
            _fields_ = [("l", wintypes.DWORD), ("load", wintypes.DWORD)] + [(n, ctypes.c_ulonglong) for n in
                       ("t", "a", "tp", "ap", "tv", "av", "ae")]
        m = MS(); m.l = ctypes.sizeof(MS); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.t / 2**30
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30


def clog(msg):
    """Log to console AND append to artifacts/logs/corpus.log so progress is monitorable from the file bridge."""
    (ART / "logs").mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(ART / "logs" / "corpus.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def download(urls, dest):
    """Try each mirror; resume if a partial exists. Logs every attempt/error/progress to corpus.log."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    clog(f"corpus: download start, target {dest.name}, {len(urls)} mirror(s)")
    for rnd in range(4):
        for url in urls:
            have = dest.stat().st_size if dest.exists() else 0
            req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
            host = url.split("/")[2]
            try:
                with urllib.request.urlopen(req, timeout=45) as r:
                    mode = "ab" if (have and r.status == 206) else "wb"
                    total = (have if mode == "ab" else 0) + int(r.headers.get("Content-Length", 0))
                    clog(f"corpus: {host} connected, {total/2**30:.2f} GB{', resume' if mode=='ab' else ''}")
                    done, last = (have if mode == "ab" else 0), time.time()
                    with open(dest, mode) as fo:
                        while chunk := r.read(1 << 20):
                            fo.write(chunk); done += len(chunk)
                            if time.time() - last > 15:
                                last = time.time(); clog(f"  {done/2**30:.2f}/{total/2**30:.2f} GB")
                if not total or dest.stat().st_size >= total:
                    clog(f"corpus: download complete ({dest.stat().st_size/2**20:.0f} MB)")
                    return
                clog(f"corpus: {host} ended early at {dest.stat().st_size/2**20:.0f} MB; will resume")
            except Exception as e:
                clog(f"corpus: {host} failed: {type(e).__name__}: {e}")
        time.sleep(20)
    raise SystemExit("corpus: all wiki mirrors failed (see artifacts/logs/corpus.log)")


# ----------------------------------------------------------------------------- stages
def st_setup(a):
    if not VPY.exists():
        run(["-m", "venv", VENV], "setup", Path(sys.executable))
    pip = ["-m", "pip", "install", "--disable-pip-version-check"]
    run([*pip, "-U", "pip"], "setup", VPY)
    try:
        run([*pip, "numpy", "pyyaml"], "setup", VPY)
    except SystemExit:
        pass
    cuda_ok = lambda: subprocess.run([str(VPY), "-c", "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)"],
                                     env=env()).returncode == 0
    installed = False
    if shutil.which("nvidia-smi") and not a.cpu:
        import re
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
        mt = re.search(r"CUDA Version:\s*([\d.]+)", out); ver = float(mt.group(1)) if mt else 0.0
        tags = [t for t in ("cu130", "cu129", "cu128", "cu126", "cu124", "cu121", "cu118") if int(t[2:]) / 10 <= (ver or 99)]
        log(f"setup: driver CUDA {ver or '?'}; trying {tags}")
        for tag in tags:
            run(["-m", "pip", "uninstall", "-y", "torch"], "setup", VPY)
            try:
                run([*pip, "torch==2.11.0", "--index-url", f"https://download.pytorch.org/whl/{tag}"], "setup", VPY)
            except SystemExit:
                continue
            if cuda_ok():
                installed = True; break
    if not installed:
        run(["-m", "pip", "uninstall", "-y", "torch"], "setup", VPY)
        run([*pip, "torch==2.11.0"] + (["--index-url", "https://download.pytorch.org/whl/cpu"] if WIN else []), "setup", VPY)
    run(["-c", "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),"
                "torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"], "setup", VPY)


def st_corpus(a):
    out = ART / "corpus.txt"
    if a.corpus:
        src = Path(a.corpus).resolve()
        if not src.exists():
            raise SystemExit(f"--corpus not found: {src}")
        out.parent.mkdir(parents=True, exist_ok=True)
        if src != out:
            shutil.copyfile(src, out)
        log(f"corpus: using {src} ({out.stat().st_size/2**20:.1f} MB)")
        return
    clog("corpus: stage reached; preparing wiki dump")
    dump = ART / "kowiki.xml.bz2"
    if not dump.exists() or dump.stat().st_size < 10 * 2**20:
        download(KOWIKI_MIRRORS, dump)
    run(["-I", "corpus_kowiki.py", dump, out, "--max-sentences", a.max_sentences], "corpus", VPY)
    log(f"corpus: {out} ({out.stat().st_size/2**20:.1f} MB)")


def st_prepare(a):
    run(["train.py", "prepare", "--corpus", ART / "corpus.txt", "--dir", ART / "run",
         "--dev", a.dev, "--min-count", a.min_count], "prepare", VPY)


def st_train(a):
    gb = ram_gb()
    bt = a.batch_tokens or 8000  # conservative for 8 GB VRAM; raise if you have more
    cmd = ["train.py", "train", "--dir", ART / "run", "--device", "cpu" if a.cpu else "cuda",
           "--steps", a.steps, "--warmup", a.warmup, "--lr", a.lr, "--batch-tokens", bt,
           "--noise", a.noise, "--eval-every", a.eval_every, "--layers", a.layers, "--d-model", a.d_model,
           "--workers", a.workers, "--max-minutes", a.max_minutes]
    if (ART / "run" / "last.pt").exists():
        cmd.append("--resume")
    run(cmd, "train", VPY)


def st_eval(a):
    run(["train.py", "eval", "--dir", ART / "run", "--ckpt", "best.pt",
         "--device", "cpu" if a.cpu else "cuda"], "eval", VPY)
    log(f"eval written to {ART/'run'/'eval.json'}")


def st_examples(a):
    # curated jamo->Hangul examples (words / addresses / sentences / mixed), clean + noisy input
    run(["eval_examples.py", "--dir", ART / "run", "--ckpt", "best.pt",
         "--device", "cpu" if a.cpu else "auto"], "examples", VPY)
    log(f"examples report written to {ART/'run'/'examples_report.txt'}")


STAGES = [("setup", st_setup), ("corpus", st_corpus), ("prepare", st_prepare), ("train", st_train),
          ("eval", st_eval), ("examples", st_examples)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--only", choices=[n for n, _ in STAGES])
    ap.add_argument("--corpus", default=None, help="your own Korean text (one sentence per line); skips wiki download")
    ap.add_argument("--max-sentences", type=int, default=100_000_000)
    ap.add_argument("--dev", type=int, default=5000)
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--warmup", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=7e-4)
    ap.add_argument("--batch-tokens", type=int, default=0)
    ap.add_argument("--noise", default="0,0,0", help="p_sub,p_del,p_ins jamo noise (set e.g. 0.03,0.02,0.02 if input may be wrong)")
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-minutes", type=float, default=0.0)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    done = ART / "done"
    if a.status:
        for n, _ in STAGES:
            print(f"{'done' if (done/n).exists() else '----'}  {n}")
        return
    keep_awake()
    in_venv = Path(sys.prefix).resolve() == VENV.resolve()
    for n, fn in STAGES:
        if a.only and n != a.only:
            continue
        if (done / n).exists() and not a.only:
            continue
        if n != "setup" and not in_venv:
            if not VPY.exists():
                raise SystemExit("run setup first")
            raise SystemExit(subprocess.call([str(VPY), __file__, *sys.argv[1:]], cwd=ROOT, env=env()))
        log(f"=== {n} ===")
        t = time.time()
        fn(a)
        (done).mkdir(parents=True, exist_ok=True)
        (done / n).write_text(time.ctime(), encoding="utf-8")
        log(f"=== {n} done ({(time.time()-t)/60:.1f} min) ===")
    log("ALL DONE. model: artifacts/run/best.pt | dev metrics: artifacts/run/eval.json")


if __name__ == "__main__":
    main()
