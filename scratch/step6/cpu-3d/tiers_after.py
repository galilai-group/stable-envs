"""Label one Step-3-shaped shard with the FROZEN Step-2/3 evaluator (no bound is defined here)."""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scratch" / "step3"))
import certify as CT  # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument("--shard", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
census = CT.read_censuses(sorted((ROOT / "scratch" / "census").glob("census_*.json")))
m = CT.merge([Path(a.shard)])
c = CT.certify(m, census)
rows = [l for l in c["labels"]]
Path(a.out).write_text(json.dumps({"shard": a.shard, "labels": rows}, indent=1))
for l in rows:
    e = l.get("evidence", {})
    print(f"{l['env']:20s} {l['device']} thr={l['threads']} tier={l['tier']:16s} q={l['qualifiers']} se={e.get('se_fps')} off={e.get('official_cpu_fps')} dec={e.get('decode_fps_per_core')} S={e.get('ratio')} cores={e.get('cores')}")
