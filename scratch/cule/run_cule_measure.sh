#!/bin/bash
set -x
SCRATCH=/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/cule
VENV=/mnt/vast/home/randall.balestriero/venvs/cule
export CUDA_HOME=$SCRATCH/cudahome
export PATH=$CUDA_HOME/bin:$PATH
source "$VENV/bin/activate"
case "$(command -v python)" in "$VENV"/*) ;; *) echo "FATAL python=$(command -v python)"; exit 1;; esac
nvidia-smi --query-gpu=name --format=csv,noheader
cd "$SCRATCH/cule_mehrdad"
python -c "import torchcule_atari; print('import ok')"
echo "=== PARITY ==="
python "$SCRATCH/parity_cule_ale.py" "$SCRATCH/parity_cule.json" --steps 200
echo "PARITY_EXIT=$?"
echo "=== THROUGHPUT ==="
python "$SCRATCH/bench_cule.py" "$SCRATCH/cule_bench.json" --max-envs 4096
echo "BENCH_EXIT=$?"
