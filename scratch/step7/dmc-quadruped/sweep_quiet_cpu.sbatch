#!/bin/bash
#SBATCH --job-name=se7-dmc-quadruped-quiet-cpu
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# The ONE authorized retry of the CPU tier for DMCQuadrupedWalk (coordinator, after 6394722 and
# 6395512 load-aborted at 34.5 / 32.4). The job WAITS at start for a quiet node (load < 12, polled
# every 5 min, up to 6 h) and gives up with a QUARANTINED-LOAD record rather than measuring on a
# loaded node. Then the paired protocol (scratch/step6/cpu-3d/certify_shard_cpu3d.py: same-node
# official renderer interleaved, decode bar, --max-load guard live throughout):
#   leg A  baseline 76dc511 tree, the four certified trackcom cells   (the floor gate)
#   leg B  branch tree, the four cells + DMCQuadrupedWalk               (the measurement)
#   leg C  baseline tree, the four cells again                          (the spread, <= 1.25x)
# then r5_kind_cost.py x3 in the same window (the ELLIPSOID _BOUNDED_KINDS question), load recorded
# before and after every leg. The gate is applied by the python at the end and written beside the
# shards; nothing is relabelled by hand.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export WS=$SCR/se7/dmc-quadruped
export BASE=$SCR/se7/baseline
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1 SDL_VIDEODRIVER=dummy OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export STABLE_ENVS_GIT_COMMIT=${STABLE_ENVS_GIT_COMMIT:-$(cat "$WS/GIT_COMMIT" 2>/dev/null || true)}
PY=$SCR/venvs/se-step6/bin/python
OUT=$WS/examples/results/step7/dmc-quadruped/phase1/tier_cpu; mkdir -p "$OUT"; cd "$WS"
NODE=$(hostname -s); FOUR=DMCCheetahRun,DMCHopperHop,DMCWalkerWalk,DMCHumanoidStand
echo "NODE $NODE JOB $SLURM_JOB_ID commit $STABLE_ENVS_GIT_COMMIT merged $(cat "$WS/MERGED_COMMIT" 2>/dev/null) load $(cat /proc/loadavg)"; grep -m1 'model name' /proc/cpuinfo
# ---- wait for a quiet node: load1 < 12, 5-min polls, 6 h cap
T0=$(date +%s); LOADLOG="$OUT/load_wait_${NODE}_${SLURM_JOB_ID}.txt"; : > "$LOADLOG"
while true; do
  L=$(cut -d' ' -f1 /proc/loadavg); echo "$(date -u +%FT%TZ) load1=$L" >> "$LOADLOG"
  if awk -v l="$L" 'BEGIN{exit !(l < 12.0)}'; then echo "QUIET at load $L after $(( $(date +%s) - T0 )) s"; break; fi
  if [ $(( $(date +%s) - T0 )) -ge 21600 ]; then
    echo "QUARANTINED-LOAD: node $NODE never fell below 12 in 6 h (last $L); not measuring" | tee "$OUT/QUARANTINED_LOAD_${NODE}_${SLURM_JOB_ID}.txt"
    $PY - "$OUT/QUARANTINED_LOAD_${NODE}_${SLURM_JOB_ID}.json" "$NODE" "$SLURM_JOB_ID" "$LOADLOG" <<'PY'
import json, sys
p, node, job, log = sys.argv[1:5]
loads = [float(l.split("load1=")[1]) for l in open(log) if "load1=" in l]
json.dump({"status": "QUARANTINED-LOAD", "node": node, "job": job, "polls": len(loads), "load_min": min(loads), "load_max": max(loads),
           "rule": "certify_shard_cpu3d --max-load 12; waited 6 h at 5-min polls", "tier": "CPU_UNMEASURED"}, open(p, "w"), indent=1)
PY
    exit 0
  fi
  sleep 300
done
set -x
echo "LOAD before A $(cat /proc/loadavg)"
PYTHONPATH=$BASE $PY scratch/step6/cpu-3d/certify_shard_cpu3d.py --device cpu --threads 1 --reps 10 --max-cell-s 60 --max-load 12 --envs $FOUR                  --out "$OUT/paired_A_baseline_${NODE}_${SLURM_JOB_ID}.json" 2>&1 | tail -8
echo "LOAD after A / before B $(cat /proc/loadavg)"
PYTHONPATH=$WS   $PY scratch/step6/cpu-3d/certify_shard_cpu3d.py --device cpu --threads 1 --reps 10 --max-cell-s 60 --max-load 12 --envs $FOUR,DMCQuadrupedWalk --out "$OUT/paired_B_branch_${NODE}_${SLURM_JOB_ID}.json"   2>&1 | tail -10
echo "LOAD after B / before C $(cat /proc/loadavg)"
PYTHONPATH=$BASE $PY scratch/step6/cpu-3d/certify_shard_cpu3d.py --device cpu --threads 1 --reps 10 --max-cell-s 60 --max-load 12 --envs $FOUR                  --out "$OUT/paired_C_baseline_${NODE}_${SLURM_JOB_ID}.json" 2>&1 | tail -8
echo "LOAD after C $(cat /proc/loadavg)"
export PYTHONPATH=$WS
for R in 1 2 3; do
  echo "LOAD before kind-cost $R $(cat /proc/loadavg)"
  $PY scratch/step6/raycast3d/r5_kind_cost.py --threads 1 --repeat 20 --out "$OUT/kind_cost_${NODE}_${SLURM_JOB_ID}_run$R.json" 2>&1 | tail -3 || echo "KIND$R rc=$?"
done
echo "LOAD after kind-cost $(cat /proc/loadavg)"
$PY scratch/step6/raycast3d/r6_kind_cost_agg.py "$OUT"/kind_cost_${NODE}_${SLURM_JOB_ID}_run{1,2,3}.json --out "$OUT/kind_cost_${NODE}_${SLURM_JOB_ID}_agg.json" 2>&1 | tail -14
set +x
# ---- the gate: spread A vs C on the four (<= 1.25x on ours and on the official), the floor against
# the certified node2333 numbers (cpu-3d findings: best fps 115.6 / 187.8 / 63.1 / 45.8, official
# llvmpipe 23.4 / 27.4 / 25.0 / 10.9), and the quadruped row from leg B with its own paired official.
$PY - "$OUT" "$NODE" "$SLURM_JOB_ID" <<'PY'
import glob, json, sys
out, node, job = sys.argv[1:4]
def load(tag):
    fs = glob.glob(f"{out}/paired_{tag}_*_{node}_{job}.json"); return json.load(open(fs[0])) if fs else None
A, B, C = load("A_baseline"), load("B_branch"), load("C_baseline")
def best(s, e):
    r = (s or {}).get("envs", {}).get(e) or {}
    fps = r.get("ours") or r.get("fps") or {}
    vals = [v for v in fps.values() if isinstance(v, (int, float))] if isinstance(fps, dict) else []
    return max(vals) if vals else r.get("best")
def official(s, e):
    r = (s or {}).get("envs", {}).get(e) or {}; oc = r.get("official_cpu") or {}
    return oc.get("median") or oc.get("fps")
known = {"DMCCheetahRun": (115.6, 23.4), "DMCHopperHop": (187.8, 27.4), "DMCWalkerWalk": (63.1, 25.0), "DMCHumanoidStand": (45.8, 10.9)}
rep = {"node": node, "job": job, "certifiable_flags": {k: (s or {}).get("certifiable") for k, s in (("A", A), ("B", B), ("C", C))},
       "not_certifiable_because": {k: (s or {}).get("not_certifiable_because") for k, s in (("A", A), ("B", B), ("C", C))},
       "four": {}, "quadruped": None}
worst = 1.0
for e, (kb, ko) in known.items():
    a, c, b = best(A, e), best(C, e), best(B, e); oa, oc, ob = official(A, e), official(C, e), official(B, e)
    sp = max(a / c, c / a) if a and c else None
    spo = max(oa / oc, oc / oa) if oa and oc else None
    worst = max(worst, sp or 1.0, spo or 1.0)
    rep["four"][e] = {"A_best": a, "C_best": c, "B_best": b, "spread_ours_AC": sp, "official_A": oa, "official_C": oc, "official_B": ob,
                      "spread_official_AC": spo, "known_node2333": {"best": kb, "official": ko},
                      "floor_ratio_ours": (a / kb) if a else None, "floor_ratio_official": (oa / ko) if oa else None}
rep["worst_spread_AC"] = worst; rep["spread_ok"] = worst <= 1.25
q = (B or {}).get("envs", {}).get("DMCQuadrupedWalk")
if q:
    rep["quadruped"] = {k: q.get(k) for k in q if k not in ("frames",)}
json.dump(rep, open(f"{out}/tier_gate_{node}_{job}.json", "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in rep.items() if k != "quadruped"}, indent=1, default=str)[:3000])
print("QUADRUPED:", json.dumps(rep["quadruped"], default=str)[:1500])
PY
echo "QUIET CPU SWEEP DONE"
