#!/bin/bash
#SBATCH --job-name=online-source-submitter
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=/oscar/scratch/sboughan/se-semantic-recovery/source-resume-20261003-v1/submit-%j.log
set -euo pipefail
BASE=/oscar/scratch/sboughan/se-semantic-recovery/source-resume-20261003-v1
# Prevent accidental resubmission after a partial launcher failure.
mkdir "$BASE/submission-started"
printf 'campaign\tjob_id\n' > "$BASE/jobs.tsv"
jid=$(sbatch --parsable /oscar/scratch/sboughan/se-semantic-recovery/fetch-source-pipeline-v1-20261003/run.sh)
printf "fetch-source-pipeline-v1-20261003\t%s\n" "$jid" | tee -a "$BASE/jobs.tsv"
jid=$(sbatch --parsable /oscar/scratch/sboughan/se-semantic-recovery/ogbench-source-isolation-v1-20261003/run.sh)
printf "ogbench-source-isolation-v1-20261003\t%s\n" "$jid" | tee -a "$BASE/jobs.tsv"
jid=$(sbatch --parsable /oscar/scratch/sboughan/se-semantic-recovery/dmc-source-comparison-v1-20261003/run.sh)
printf "dmc-source-comparison-v1-20261003\t%s\n" "$jid" | tee -a "$BASE/jobs.tsv"
jid=$(sbatch --parsable /oscar/scratch/sboughan/se-semantic-recovery/navigation-source-comparison-v1-20261003/run.sh)
printf "navigation-source-comparison-v1-20261003\t%s\n" "$jid" | tee -a "$BASE/jobs.tsv"
jid=$(sbatch --parsable /oscar/scratch/sboughan/se-semantic-recovery/ogbench-validation-columns-scene-cuda-v2-20261003/run.sh)
printf "scene-cuda-retry\t%s\n" "$jid" | tee -a "$BASE/jobs.tsv"
