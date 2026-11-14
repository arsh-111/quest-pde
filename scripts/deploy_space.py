"""Deploy the staged build context to a Hugging Face Space (Docker SDK) and wait until it runs.

python scripts/deploy_space.py <staged_dir> [space_name]
Uses the local Hugging Face login (never prints or writes the token). Files are uploaded through the Hub API, so the
local git history (and its commit metadata) is NOT pushed. On a build/runtime error the Space's build log is printed.
Writes results/deploy.json (URL, build time, stages).
"""
import json
import sys
import time
from pathlib import Path

import httpx
from huggingface_hub import HfApi, get_token

ROOT = Path(__file__).resolve().parents[1]
staged = Path(sys.argv[1])
name = sys.argv[2] if len(sys.argv) > 2 else "sorption-surrogate-lab"

api = HfApi()
try:
    user = api.whoami()["name"]
except Exception as e:  # noqa: BLE001
    print("AUTH FAILED: not logged in to Hugging Face on this machine "
          f"({type(e).__name__}). Run `hf auth login` (or `.venv\\Scripts\\hf.exe auth login`) and paste a WRITE token.")
    sys.exit(2)
repo_id = f"{user}/{name}"
print("logged in; target Space:", repo_id, flush=True)
api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True, private=False)
t0 = time.time()
info = api.upload_folder(folder_path=str(staged), repo_id=repo_id, repo_type="space",
                         commit_message="Deploy Sorption Surrogate Lab (CPU, Docker)")
print("uploaded:", getattr(info, "commit_url", info), flush=True)

stages, last = [], None
while time.time() - t0 < 1800:
    rt = api.get_space_runtime(repo_id)
    if rt.stage != last:
        stages.append({"t_s": round(time.time() - t0, 1), "stage": rt.stage})
        print(f"{stages[-1]['t_s']:7.1f}s  {rt.stage}", flush=True)
        last = rt.stage
    if rt.stage in ("RUNNING", "BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR", "NO_APP_FILE"):
        break
    time.sleep(10)

if last != "RUNNING":
    hdr = {"Authorization": f"Bearer {get_token()}"}  # used only in the request header, never printed
    for kind in ("build", "run"):
        lines = []
        with httpx.stream("GET", f"https://huggingface.co/api/spaces/{repo_id}/logs/{kind}", headers=hdr, timeout=30) as r:
            for raw in r.iter_lines():
                if raw.startswith("data:"):
                    try:
                        lines.append(json.loads(raw[5:]).get("data", "").rstrip())
                    except json.JSONDecodeError:
                        pass
                if len(lines) > 400:
                    break
        print(f"----- {kind} log (last 60 lines) -----")
        print("\n".join(lines[-60:]))

sub = api.space_info(repo_id).subdomain if hasattr(api.space_info(repo_id), "subdomain") else None
url = f"https://{sub}.hf.space" if sub else f"https://{user}-{name}.hf.space".lower()
out = {"repo_id": repo_id, "space_page": f"https://huggingface.co/spaces/{repo_id}", "app_url": url,
       "final_stage": last, "stages": stages, "seconds_upload_to_final_stage": round(time.time() - t0, 1)}
(ROOT / "results" / "deploy.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
sys.exit(0 if last == "RUNNING" else 1)
