"""Read-only collection and terminal checks for the bounded Humanoid CPU gate."""
import csv
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/research/semantic_recovery_evidence/dmc_cpu_profile_20260929/humanoid_chunk_gate"
REMOTE = "sboughan@sshcampus.ccv.brown.edu:/oscar/scratch/sboughan/se-semantic-recovery/dmc-humanoid-chunk-gate-20260929"
SSH = "ssh -S /private/tmp/stable-envs-oscar-control -o BatchMode=yes -o LogLevel=ERROR"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    status = {"updated_unix": time.time(), "statuses": {}}
    for leg in ("baseline", "candidate"):
        dest = OUT / leg
        dest.mkdir(exist_ok=True)
        result = subprocess.run(["rsync", "-az", "-e", SSH, REMOTE + f"/{leg}/results/", str(dest) + "/"], capture_output=True, text=True)
        if result.returncode:
            status.setdefault("download_errors", []).append(result.stderr)
        p = dest / "66/result.json"
        if p.exists():
            status["statuses"][leg] = json.loads(p.read_text())["status"]
    if status["statuses"] == {"baseline": "COMPLETE", "candidate": "COMPLETE"}:
        before, after = [json.loads((OUT / leg / "66/result.json").read_text()) for leg in ("baseline", "candidate")]
        assert before["profile"]["host"] == after["profile"]["host"]
        assert before["profile"]["environment"]["SLURM_JOB_ID"] == after["profile"]["environment"]["SLURM_JOB_ID"]
        for k in ("frames", "corpus_sha256", "batch_image_sha256"):
            assert before[k] == after[k], k
        assert before["state_nonmutation_passed"] and after["state_nonmutation_passed"]
        assert max(m["paired_mse"]["mean"] for m in after["measurements"]) <= .001
        rows = []
        for a, b in zip(before["measurements"], after["measurements"]):
            assert a["batch"] == b["batch"] and a["output_device"] == b["output_device"] == "cpu"
            rows.append(dict(batch=b["batch"], baseline_fps=a["ours"]["frames_per_second"], candidate_fps=b["ours"]["frames_per_second"], source_fps=b["source"]["frames_per_second"], improvement=a["ours"]["mean_ms"] / b["ours"]["mean_ms"], speedup=b["speedup"], mse_mean=b["paired_mse"]["mean"], mse_sd=b["paired_mse"]["sd"]))
        with (OUT / "comparison.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        status["gate"] = "PASS: same allocation/States, exact baseline output, unchanged State; inspect speedups separately"
    (OUT / "collection_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status))


if __name__ == "__main__":
    main()
