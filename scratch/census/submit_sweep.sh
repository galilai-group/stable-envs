#!/bin/bash
# Fan the decode sweep out over the interpreters that cover the registry.
#
# No interpreter holds every family, so the sweep is a set cover (scratch/census/assignment.json) and
# each job measures only the envs its interpreter is the chosen owner of. Chunked because the heavy
# MuJoCo families rebuild a render context per batch point, and one 147-env job would sit near the
# wall clock limit with nothing written if it were preempted.
set -uo pipefail
R=/mnt/vast/home/randall.balestriero/code/stable-envs
cd $R
mkdir -p $R/examples/results/decode /mnt/vast/home/randall.balestriero/logs/stable_envs_logs/census

python3 - <<'EOF' > /tmp/se_sweep_cmds.sh
import json, math
a = json.load(open('/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/census/assignment.json'))
# Ray-cast / MJWarp families rebuild a context per batch point and cannot hold 16k worlds, so they get
# the shorter ladder. The 2-D families are the ones whose ceiling is worth finding.
HEAVY = {'se-hb', 'dmc'}
CHUNK = 40
for venv, envs in a.items():
    ladder = '1,64,256,1024,4096' if venv in HEAVY else '1,64,256,1024,4096,16384'
    n = max(1, math.ceil(len(envs) / CHUNK))
    for i in range(n):
        part = envs[i * CHUNK:(i + 1) * CHUNK]
        if not part:
            continue
        tag = f'{venv}_{i}' if n > 1 else venv
        print(f"sbatch --job-name=dec_{tag} scratch/census/sweep.sbatch {venv} "
              f"--tag {tag} --batches {ladder} --cpu-frames 60 --envs {','.join(part)}")
EOF

bash /tmp/se_sweep_cmds.sh
