#!/bin/bash
#SBATCH --job-name=classic-sustained-cpu
#SBATCH --partition=3090-gcondo
#SBATCH --qos=cs-3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-1%2
#SBATCH --no-requeue
#SBATCH --output=/oscar/scratch/sboughan/se-semantic-recovery/classic-snapshot-sustained-cpu-v1-20261003/gate-%A_%a.log
set -euo pipefail
BASE=/oscar/scratch/sboughan/se-semantic-recovery/classic-snapshot-sustained-cpu-v1-20261003
PY=/oscar/scratch/sboughan/venvs/se-step6/bin/python
cd "$BASE/code"
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 SDL_VIDEODRIVER=dummy
export XDG_CACHE_HOME="$BASE/cache/$SLURM_ARRAY_TASK_ID"
mkdir -p "$XDG_CACHE_HOME"
"$PY" - "$BASE" <<'VERIFY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]);manifest=json.loads((root/'source_manifest.json').read_text())
for rel,digest in manifest['files'].items():assert hashlib.sha256((root/'code'/rel).read_bytes()).hexdigest()==digest,rel
print('Frozen source verified',len(manifest['files']),flush=True)
VERIFY
nvidia-smi
lscpu
DOMAINS=(pendulum acrobot)
DOMAIN=${DOMAINS[$SLURM_ARRAY_TASK_ID]}
"$PY" - "$BASE" "$DOMAIN" <<'RUN'
import pathlib,subprocess,sys
base=pathlib.Path(sys.argv[1]);domain=sys.argv[2]
cells=((1,'none','final'),(8,'all','final'),(8,'none','none'),(8,'none','final'))
for index,(horizon,states,frames) in enumerate(cells):
    modes=('baseline','candidate') if index%2==0 else ('candidate','baseline')
    for mode in modes:
        output=base/'results'/domain/f't{horizon}-{states}-{frames}-{mode}.json'
        print(domain,horizon,states,frames,mode,flush=True)
        subprocess.run([sys.executable,'scripts/benchmarks/online_stepping/classic_snapshot_sustained.py',
            '--domain',domain,'--mode',mode,'--device','cpu','--batch','256',
            '--horizon',str(horizon),'--states',states,'--frames',frames,'--transitions','128',
            '--samples','6','--output',str(output)],check=True)
subprocess.run([sys.executable,'scripts/benchmarks/online_stepping/summarize_classic_snapshot_sustained.py',
    str(base/'results'/domain),str(base/f'{domain}.csv'),'--source',str(pathlib.Path.cwd()),'--device','cpu','--batch','256'],check=True)
RUN
