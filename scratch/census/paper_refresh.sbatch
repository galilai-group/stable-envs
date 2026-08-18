#!/bin/bash
#SBATCH --job-name=se-paper
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --exclude=gb-west-a1-gpu-004,gb-west-a1-gpu-047,gb-west-a1-gpu-058,gb-west-a1-gpu-068,gb-west-a1-gpu-070,gb-west-a1-gpu-082,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/rjepa_bench_logs/se_paper_%j.out
set -uo pipefail
R=/mnt/vast/home/randall.balestriero/code/stable-envs
cd $R; export PYTHONPATH=$R MUJOCO_GL=egl SDL_VIDEODRIVER=dummy STABLE_ENVS_HUMANOIDBENCH=1
PY=/mnt/vast/home/randall.balestriero/venvs/se-all/bin/python
echo "### node=$(hostname)"

echo "=== 1. regenerate tables + macros (main.tex now uses \\seNenvBelow) ==="
"$PY" -u examples/paper/make_tables.py 2>&1 | tail -12
echo "--- the two counts that were typed by hand and both wrong ---"
grep -E "seNfamBelow|seNenvBelow|seNenvAbove" examples/paper/tables/macros.tex

echo; echo "=== 2. capture Crafter's probe output, which existed only on stdout ==="
# The paper's G2 sentence rested on numbers scratch/crafter/probe.py PRINTS and never wrote. Capturing
# it is the difference between a claim a reader can check and one they must take on trust.
mkdir -p $R/scratch/crafter/out
"$PY" -u scratch/crafter/probe.py > $R/scratch/crafter/out/probe.log 2>&1
rc=$?
echo "probe rc=$rc; key lines:"
grep -E "^G1|^G2|^G3|G1/atlas|daylight in|crafter's own" $R/scratch/crafter/out/probe.log | head -20

echo; echo "=== 3. compile the paper ==="
cd $R/examples/paper
for i in 1 2 3; do pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null 2>&1 || true; done
pdflatex -interaction=nonstopmode main.tex 2>&1 | grep -E "^!|Undefined control|Overfull|Underfull|LaTeX Warning: Reference" | head -20
echo "--- pages ---"; pdfinfo main.pdf 2>/dev/null | grep -E "^Pages"
echo "--- undefined refs/macros ---"; grep -cE "Undefined control sequence" main.log || true
echo "### paper done"
