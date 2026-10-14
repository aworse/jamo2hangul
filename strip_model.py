"""Make a small, inference-only checkpoint: fp16 weights + cfg, no optimizer state.

  python strip_model.py <in_ckpt> <out_ckpt>

Result loads directly with train.py (load_for_infer upcasts fp16->fp32 on load):
  python train.py infer --dir <dir_with_vocab.json> --ckpt <out_ckpt> --in jamo.txt --out hangul.txt
"""
import sys
import torch

inp, out = sys.argv[1], sys.argv[2]
ck = torch.load(inp, map_location="cpu")
model = {k: (v.half() if torch.is_floating_point(v) else v) for k, v in ck["model"].items()}
torch.save({"model": model, "cfg": ck["cfg"], "step": ck.get("step", 0)}, out)
mb = __import__("os").path.getsize(out) / 2**20
print(f"saved {out}  ({mb:.0f} MB, step {ck.get('step', 0)})")
