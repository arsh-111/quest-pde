"""Assemble the minimal Hugging Face Space build context in a directory (argv[1]); prints its file list and size.

Only what serving needs: app/ (code, static, assets), qpde/ (library), the 6 checkpoints the service loads,
deploy/Dockerfile, deploy/requirements.space.txt, and deploy/README.space.md as the Space card (README.md).
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
out = Path(sys.argv[1])
if out.exists():
    shutil.rmtree(out)
out.mkdir(parents=True)
ens = json.loads((ROOT / "app" / "assets" / "summary.json").read_text())["ensemble"]
files = [*sorted((ROOT / "app").rglob("*")), *sorted((ROOT / "qpde").glob("*.py"))]
files += [ROOT / "checkpoints" / f"{n}.pt" for n in [*(f"{ens}_s{i}" for i in range(5)), "cnn_s0"]]
for f in files:
    if f.is_dir() or "__pycache__" in f.parts:
        continue
    dst = out / f.relative_to(ROOT)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(f, dst)
shutil.copy2(ROOT / "deploy" / "Dockerfile", out / "Dockerfile")
shutil.copy2(ROOT / "deploy" / "requirements.space.txt", out / "requirements.space.txt")
shutil.copy2(ROOT / "deploy" / "README.space.md", out / "README.md")
tot = 0
for f in sorted(out.rglob("*")):
    if f.is_file():
        tot += f.stat().st_size
        print(f"{f.stat().st_size / 1024:9.1f} KB  {f.relative_to(out).as_posix()}")
print(f"total {tot / 2**20:.1f} MB")
