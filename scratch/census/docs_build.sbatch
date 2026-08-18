#!/bin/bash
#SBATCH --job-name=se-docs
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/vast/home/randall.balestriero/rjepa_bench_logs/se_docs_%j.out
set -uo pipefail
cd /mnt/vast/home/randall.balestriero/code/stable-envs
export PYTHONPATH=.
PY=/mnt/vast/home/randall.balestriero/venvs/se-docs/bin/python
[ -x "$PY" ] || PY=/mnt/vast/home/randall.balestriero/venvs/se-all/bin/python
echo "### $(hostname)  $PY"
# --strict: mkdocs.yml sets anchors/unrecognized_links to warn, and a warning is exactly how a dead
# cross-reference survives. The new page links to six existing docs and two videos.
"$PY" -m mkdocs build --strict --site-dir /tmp/se_site_$SLURM_JOB_ID 2>&1 | tail -40
rc=$?
echo "--- videos the new page references ---"
for f in docs/assets/media/PushT/sidebyside.mp4 docs/assets/media/PushT/grid.png \
         docs/assets/media/PointmazeMediumNavigate/sidebyside.mp4; do
  printf "  %-58s %s\n" "$f" "$([ -f "$f" ] && echo OK || echo MISSING)"
done
rm -rf /tmp/se_site_$SLURM_JOB_ID
echo "### docs rc=$rc"
