---
name: oscar
description: Router for running work on Brown's OSCAR (CCV) SLURM cluster. Use when the user wants to run, submit, schedule, monitor, or debug jobs on OSCAR, choose a partition/GPU, manage HOME vs scratch storage, or start an interactive session. Points to the focused oscar-* skills and states the non-negotiable rules.
---

# OSCAR (Brown CCV) — operations router

OSCAR is Brown's shared cluster, scheduled by SLURM. Pick the focused skill for the task:

- **oscar-sbatch-jobs** — write & submit batch jobs, arrays, and submitit/hydra sweeps.
- **oscar-gpu-selection** — choose partition + QOS + GPU-memory constraint (the OOM matrix).
- **oscar-interactive-sessions** — interactive GPU/CPU shells for debugging.
- **oscar-storage-quotas** — HOME vs scratch, quotas, where checkpoints/logs/caches go.
- **oscar-job-troubleshooting** — monitor, cancel *correctly*, debug swallowed failures, preemption.

## Non-negotiables — apply to every job, even a one-off

1. **Never run heavy compute on the login node.** No training, no bulk `torch` imports, no multi-launch submitters. It OOMs the shared login node and gets you noticed. Wrap the work — including the *submitter script itself* — in a small CPU job on `batch`/`normal`. (→ oscar-sbatch-jobs)
2. **Heavy output lives on scratch, never HOME.** HOME is **100 GB and fills silently** (W&B local data, hydra/submitit logs, checkpoints). Point checkpoints, dataset cache, W&B `save_dir`, and hydra `run.dir`/`sweep.dir` at `/oscar/scratch/$USER`. (→ oscar-storage-quotas)
3. **`scancel` does not reliably kill auto-requeue jobs** — they requeue and reappear with new IDs. Disable requeue first. (→ oscar-job-troubleshooting)
4. **A SLURM state of `COMPLETED` is not proof of success.** submitit swallows crashes and reports success. Verify real progress (checkpoint epoch advanced, expected output written), not job state. (→ oscar-job-troubleshooting)

## Quick reference (this account)

- **GPU condo:** `--partition=3090-gcondo --qos=cs-3090-gcondo` (24 GB 3090s, preemptible, ~32-GPU cap). Allocation-specific.
- **GPU general:** `--partition=gpu --qos=norm-gpu` (~2 concurrent, not preemptible).
- **CPU / submitters:** `--partition=batch --qos=normal`.
- **24 GB+ constraint:** `--constraint=geforce3090|l40s` (or wider `geforce3090|l40s|a40|a5000|a5500`).
- `checkquota` for usage; scratch base `/oscar/scratch/$USER`; home `/oscar/home/$USER` (100 GB).
---
name: oscar-gpu-selection
description: Choose the right OSCAR partition, QOS, GPU-memory constraint, and concurrency for a GPU job — and avoid CUDA out-of-memory. Use when a job OOMs, when deciding which partition/constraint to request, when a job pends forever, or when picking between condo and general-access GPUs on OSCAR.
---

# OSCAR — GPU partition / QOS / memory selection

## GPU memory tiers (the OOM trap)
OSCAR mixes card types inside a partition. The dangerous ones:

| Cards | VRAM | Notes |
|---|---|---|
| A5000 / A5500 | **22 GB** (22.06 GiB) | resnet50 SSL @ bs256 needs ~21.7 GB → **OOMs here** |
| geforce3090 | 24 GB | safe for resnet50@bs256 and DINO multi-crop |
| L40S | 48 GB | roomy |
| A40 | 48 GB | roomy |

**Anything that OOMs on 22 GB must pin ≥24 GB cards** with a constraint:
- `--constraint=geforce3090|l40s` (24 GB or 48 GB), or wider `--constraint=geforce3090|l40s|a40|a5000|a5500` only if the job actually fits 22 GB.
- Known 22 GB-OOM workloads here: **resnet50 @ bs256** (any SSL method), **DINO multi-crop** (2 global + 6 local) on any backbone.
- **ViT-S/16 is memory-light** and does *not* need the constraint — dropping it schedules faster.
- With `--gpus-per-node=1` (submitit) co-located jobs are isolated fine; you don't also need `--gres`.

## Partition / QOS map (this account)

| Purpose | `--partition` | `--qos` | Character |
|---|---|---|---|
| GPU, bulk sweeps | `3090-gcondo` | `cs-3090-gcondo` | condo, 24 GB 3090s, **preemptible**, ~32-GPU cap (allocation-specific) |
| GPU, few must-finish | `gpu` | `norm-gpu` | general-access, **~2 concurrent**, not preempted |
| CPU / submitters / prewarm | `batch` | `normal` | no GPU |

- **Partition and QOS must match** or the job never starts (`QOSMismatch`/pending). Don't mix e.g. `3090-gcondo` with `normal`.
- The condo QOS name (`cs-3090-gcondo`) is **specific to this allocation** — a different group has a different condo name. `sacctmgr show qos` / `sinfo` to discover what you can use.
- **Condo is preemptible.** Jobs get killed and requeued when the condo owner needs the node. For work that must not restart from a bad checkpoint, or that preempts chronically, move to `gpu`/`norm-gpu` (keep the 24 GB constraint).

## Concurrency caps
- Condo QOS has a per-user GPU cap (~32) **and** a CPU cap — 8-CPU jobs can hit `QOSMaxCpuPerUserLimit` before the GPU cap. If jobs pend with a CPU-limit reason, lower `--cpus-per-task` or reduce concurrency.
- `gpu`/`norm-gpu` effectively caps ~2 concurrent — fine for a smoke test, not a sweep.
- submitit sweeps: cap in-flight tasks with `array_parallelism` (default 32 here) to leave the condo headroom for others.

## Deciding fast
1. Fits 22 GB and memory-light (ViT-S)? → `3090-gcondo`/`cs-3090-gcondo`, **no constraint**.
2. Needs ≥24 GB (resnet50@bs256, DINO)? → same partition **+ `--constraint=geforce3090|l40s`**.
3. Must-finish, few jobs, or preemption-sensitive? → `gpu`/`norm-gpu` + the 24 GB constraint.
4. CPU only (prewarm, submitter, data prep)? → `batch`/`normal`.
---
name: oscar-interactive-sessions
description: Start and use an interactive session on OSCAR — a GPU or CPU shell for debugging, smoke tests, and reproducing failures. Use when the user wants an interactive/dev node, a GPU shell, to run something live instead of batch, or to see the real error a submitted job hid.
---

# OSCAR — interactive sessions

## When to go interactive
- **Reproduce a failure that submitit/SLURM hid.** A submitted job reports `COMPLETED` while actually crashing (submitit swallows exceptions). Running the *same command directly* in an interactive session surfaces the real traceback. This is the #1 reason to go interactive here.
- Quick smoke test of a training command, a dataset build, or a `nvidia-smi` sanity check before committing a sweep.
- **Not** for long sweeps — those go to batch (see oscar-sbatch-jobs). Interactive sessions die when your SSH drops (use `tmux`).

## Getting a session

OSCAR/CCV provides the `interact` wrapper (confirm exact flags with `interact --help`):
```bash
# GPU debug shell: 1 GPU, 1h, 32G, 8 cores, pin a 24GB card
interact -q gpu -g 1 -t 01:00:00 -m 32g -n 8 -f geforce3090
# CPU-only shell (data prep, debugging, prewarm)
interact -q batch -t 02:00:00 -m 16g -n 4
```
Equivalent with raw SLURM if `interact` isn't available or you need more control:
```bash
salloc --partition=gpu --qos=norm-gpu --gres=gpu:1 --constraint=geforce3090|l40s \
       --cpus-per-task=8 --mem=32G --time=01:00:00
# then it drops you on the node; or: srun --pty bash
```
Partition/QOS/constraint choices are the same as batch — **see oscar-gpu-selection** (don't request a 22 GB card for a 24 GB job).

## Once on the node
```bash
tmux new -s dev                      # survive SSH drops; reattach with: tmux attach -t dev
nvidia-smi                           # confirm you actually got a GPU
source /oscar/home/$USER/<repo>/.venv/bin/activate   # this project uses a .venv, not `module load python`
python -m benchmarks.run model=simclr backbone=resnet50 dataset=cifar10  # direct run = real errors
```
- Keep interactive GPU sessions short — they hold a scarce card. Release with `exit` / Ctrl-D as soon as you're done.
- Heavy output still belongs on scratch even interactively — see oscar-storage-quotas.
- If the cluster uses environment modules for CUDA/toolchains, `module avail` / `module load cuda` — but this project's `.venv` already carries its CUDA-enabled torch, so usually no module loads are needed.
---
name: oscar-job-troubleshooting
description: Monitor, cancel, and debug SLURM jobs on OSCAR — find a job's log, cancel jobs that keep requeuing, detect crashes that submitit reports as success, handle preemption, and fix dataset-build races. Use when a job won't die, a job "completed" but produced nothing, jobs keep reappearing after scancel, or a run is stuck/looping.
---

# OSCAR — job monitoring & troubleshooting

## Cancelling: `scancel` alone does NOT kill auto-requeue jobs
submitit/stable-pretraining install a SIGTERM handler that forwards to SIGUSR2 → `checkpoint_and_try_requeue`. A plain `scancel` therefore **requeues** the job — it reappears with a new id, and the queue never drains. Symptom: the same `{model}_{dataset}` jobs keep coming back.

**To actually kill:**
```bash
for j in $(squeue -u $USER -h -r -o "%A"); do scontrol update jobid=$j requeue=0; done
scancel --signal=KILL -u $USER        # SIGKILL bypasses the requeue handler
scancel -u $USER                       # sweep up the rest
```
Cancel selectively (not `-u $USER`) when other work must keep running — target specific job/array ids. Auto-requeue-on-walltime is *desirable* for long training; only override it when you deliberately want a clean stop.

## "COMPLETED" is not success — submitit swallows crashes
A job can crash (bad checkpoint load, OOM, an exception) and still report `COMPLETED`. **Never trust job state as proof of work.** Verify the real thing:
- checkpoint epoch actually advanced (`torch.load(...)['epoch']`), or
- expected output written, or
- W&B run has runtime > ~300 s and the metric you expect.

Recurring masked failures seen here: a missing-file resume path, an OOM on a 22 GB card, a bad `L2Norm` import, a lazily-shaped buffer mismatch — all reported as success. Reproduce with a **direct (non-submitit) run** in an interactive session to see the traceback (→ oscar-interactive-sessions).

## Finding a job's live log
```bash
scontrol show job <id> | grep -oP 'StdOut=\K\S+'     # the actual stdout path
sacct -j <id> --format=JobID,State,ExitCode,Elapsed,MaxRSS,Reason
squeue -u $USER -h -r -o "%A %j %T %V %R"             # id, name, state, submit-time, reason(pending)
```
- submitit writes logs under a **hidden `.submitit/`** directory. Shell `**` globs and Python `glob('**')` **skip hidden dirs** — use `find` or `os.walk` to reach them.
- `%R` on a pending job tells you *why* it's stuck (`QOSMaxCpuPerUserLimit`, `Priority`, `QOSMismatch`, `Resources`).

## Preemption (condo)
Condo (`3090-gcondo`) jobs get preempted and requeued when the owner needs the node. Mostly fine — training resumes from `last.ckpt`. But watch two failure modes:
- **Death-spiral resume:** if `last.ckpt` is frozen at a low epoch while real progress sits in `last-v1.ckpt`/`epoch=N.ckpt`, each preempt-resume restarts low and never finishes. Fix: stage the **highest-epoch resumable** checkpoint (one that has `optimizer_states`) *as* `last.ckpt`, drop any stale `*.resume-clean.ckpt`, relaunch.
- **Chronic preemption:** move the cell to `gpu`/`norm-gpu` (not preemptible; keep the 24 GB constraint).

## Concurrent dataset-build races
Multiple jobs building the same uncached dataset at once corrupt the cache. **Prewarm first** on a CPU node (single build; a cache hit short-circuits in <1 s), then launch the sweep. If a build was interrupted, clean stale `*.build.lock` files under the processed cache dir before retrying — a leftover lock can hang the next build.

## Watchers (poll without a fragile foreground process)
Background a bash loop that polls `squeue` and reports when an array registers / drains, or gate the next step on `--dependency`. Don't block a foreground process for hours — prefer SLURM dependencies, which survive your session ending.
---
name: oscar-sbatch-jobs
description: Write and submit SLURM batch jobs on Brown's OSCAR cluster — #SBATCH headers, resource requests (CPU/GPU/mem/time), job arrays, dependencies, and submitit/hydra multirun sweeps. Use when creating an .sbatch or submit .sh script, running sbatch, launching an array/sweep, or asking how to request resources on OSCAR.
---

# OSCAR — writing & submitting batch jobs

## The rule that matters most
**Never submit heavy work from the login node, and never let the *submitter* run there.** A script that imports torch or fans out dozens of launches will overload the shared login node. Put the submitter itself in a tiny CPU job on `batch`/`normal`; it runs `sbatch`/launches from a compute node.

## Copy-paste header template

```bash
#!/bin/bash
#SBATCH --job-name=myjob
#SBATCH --partition=3090-gcondo        # GPU condo. CPU work: batch. General GPU: gpu
#SBATCH --qos=cs-3090-gcondo           # MUST match the partition (batch→normal, gpu→norm-gpu)
#SBATCH --gres=gpu:1                   # omit entirely for CPU jobs
#SBATCH --constraint=geforce3090|l40s  # 24GB+ cards; drop for memory-light jobs. See oscar-gpu-selection
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00                # OSCAR max is often 24h/48h depending on QOS
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out   # → SCRATCH, never HOME (%x=name %j=jobid %u=user)
set -uo pipefail                        # fail loudly; submitit/SLURM otherwise hide errors
# ... your commands ...
```

- Submit: `JID=$(sbatch --parsable myjob.sbatch)` — capture the id for dependencies/monitoring.
- Partition ↔ QOS must match, or the job pends forever with `QOSMismatch`. Pairs: `3090-gcondo`↔`cs-3090-gcondo`, `gpu`↔`norm-gpu`, `batch`↔`normal`.
- Choosing partition/constraint/memory: **see oscar-gpu-selection** (the 22 GB OOM matrix).
- `mkdir -p /oscar/scratch/$USER/logs` before first use — a missing `--output` dir makes the job fail silently.

## Job arrays
```bash
#SBATCH --array=0-19%8      # 20 tasks, at most 8 running at once
# inside: use $SLURM_ARRAY_TASK_ID to index a bash array of inputs
```
The `%N` throttle protects the condo cap and other users. Read inputs with `awk`, not bash `read` — CSVs written by python default to CRLF (`\r\n`), which `read` mangles but `awk` tolerates.

## Dependencies (serialize waves without a babysitter process)
```bash
A=$(sbatch --parsable wave1.sbatch)
sbatch --dependency=afterany:$A wave2.sbatch   # afterok: only if wave1 succeeded
```
Prefer SLURM dependencies over a long-running bash driver — the driver dies with your session; the dependency survives.

## submitit / hydra multirun sweeps (this repo's pattern)
Launcher config lives in `benchmarks/conf/slurm.yaml` (`override hydra/launcher: submitit_slurm`). Key fields already tuned:
- `timeout_min: 1440` (24 h), `cpus_per_task: 8`, `mem_gb: 100`, `gpus_per_node: 1`
- `additional_parameters.constraint: geforce3090|l40s`
- `max_num_timeout: 3` — lets a job that hits the 24 h wall **auto-requeue** and resume from `last.ckpt` (this is why plain `scancel` requeues — see oscar-job-troubleshooting)
- `array_parallelism: 32` — concurrent-task cap; override per launch with `hydra.launcher.array_parallelism=N`
- Partition/QOS come from env: `SLURM_PARTITION`, `SLURM_QOS`.

Launch a sweep from a CPU submitter job:
```bash
env RUN_MODULE=benchmarks.run CONFIG=slurm BACKBONES=resnet50 MODELS=simclr,dino \
    SLURM_PARTITION=3090-gcondo SLURM_QOS=cs-3090-gcondo STABLE_DATASETS_ROOT=/oscar/scratch/$USER/.stable-datasets \
    ./benchmarks/launch.sh cifar10,svhn
```
The launcher blocks on submitit's poll; detach once the array **registers** in `squeue` (a new array id appears) and let SLURM run it independently:
```bash
before=$(squeue -u $USER -h -r -o "%F" | sort -u)
env ... ./benchmarks/launch.sh "$ds" > launch.log 2>&1 &   lpid=$!
for i in $(seq 1 45); do sleep 8
  new=$(comm -13 <(printf '%s\n' "$before") <(squeue -u $USER -h -r -o "%F"|sort -u))
  [ -n "$new" ] && { echo "registered: $new"; break; }
  kill -0 $lpid 2>/dev/null || break        # launcher died early → check launch.log
done
kill $lpid 2>/dev/null
```
---
name: oscar-storage-quotas
description: Manage OSCAR filesystems and quotas — HOME vs scratch, where to put checkpoints/logs/caches/W&B data, checkquota, and freeing space. Use when a job hits a disk-quota error, when deciding where output should live, when HOME is full, or when runs crash for no obvious reason (often silent quota exhaustion).
---

# OSCAR — storage & quotas

## The layout that keeps runs alive

| Path | Size (this account) | Put here |
|---|---|---|
| `/oscar/home/$USER` (HOME) | **100 GB** hard | code, `.venv`, small configs. **Nothing heavy.** |
| `/oscar/scratch/$USER` (scratch) | ~512 GB soft / ~10 TB hard, **purged** | checkpoints, dataset cache, W&B local data, hydra/submitit logs, everything big |

**HOME's 100 GB fills silently and then everything crashes.** The usual culprits:
- **W&B local run data** — accumulates fast (tens of GB across a project) if `save_dir` defaults to the run's cwd under the repo.
- **hydra/submitit logs** — `run.dir`/`sweep.dir` default under the repo (HOME).
- Stray checkpoints written to a HOME-relative path.

## Redirect everything to scratch
- **Checkpoints / dataset cache:** set `STABLE_DATASETS_ROOT=/oscar/scratch/$USER/.stable-datasets` (this repo resolves `checkpoint.dir` and the data cache from it).
- **W&B:** set the logger `save_dir` to a scratch path (this repo derives it from `checkpoint.dir`); or `export WANDB_DIR=/oscar/scratch/$USER/wandb`.
- **hydra outputs (incl. submitit `.out/.err`):** point `hydra.run.dir` / `hydra.sweep.dir` at scratch (this repo uses `${oc.env:STABLE_DATASETS_ROOT}/benchmark-runs/...`).
- **Relative cache dirs:** some tools write to a *relative* path (e.g. `./.anonymous-datasets-cache`). Symlink it to scratch so it doesn't land in HOME:
  `ln -sfn "$STABLE_DATASETS_ROOT" ./.anonymous-datasets-cache`

## Checking usage
```bash
checkquota                 # HOME + scratch: used / soft / hard / grace / file counts
```
- **`checkquota` LAGS deletions** (GPFS quota caching) — after cleanup the number can stay high for a while. Use `du -sh <dir>` for ground truth of what's actually on disk.
- Watch both the **block** quota and the **file-count** quota — millions of tiny cache/log files can exceed the inode limit before the byte limit.

## Freeing space (safely)
- W&B local dirs are caches once the run has synced to the cloud (`0` offline runs) — the big `wandb/run-*` dirs under the repo are safe to delete after syncing. Confirm no run is active first.
- Old hydra `benchmark-runs/sweep/<timestamp>/` logs are deletable once you've read what you need.
- **Scratch is periodically purged** (files untouched for N days are removed). Anything you must keep — final checkpoints, results — copy somewhere durable; don't rely on scratch as archival storage.
- Before deleting anything you didn't create, look at it first (it may be another sweep's live data).
