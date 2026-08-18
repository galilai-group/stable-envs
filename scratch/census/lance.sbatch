#!/bin/bash
#SBATCH --job-name=se_lance
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH --exclude=gb-west-a1-gpu-047,gb-west-a1-gpu-068,gb-west-a1-gpu-070,gb-west-a1-gpu-082,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/census/lance_%x_%j.log
#
# ONE job, ONE node, ONE process for the whole registry -- deliberately not chunked. The storage side
# is a property of the filesystem and the decoder, so measuring different envs on different nodes makes
# the column vary for reasons that have nothing to do with the envs. Ask for the whole node's cores so
# the parallel-worker figure is not competing with a neighbour.
R=/mnt/vast/home/randall.balestriero/code/stable-envs
P=/mnt/vast/home/randall.balestriero/miniconda3/envs/test/bin/python
export PYTHONPATH=$R
LOCAL=/tmp/se_lance_$SLURM_JOB_ID
SHARED=/mnt/vast/home/randall.balestriero/tmp_lance_$SLURM_JOB_ID
mkdir -p $LOCAL $SHARED
cd $R/examples
$P bench_lance.py --local "$LOCAL" --shared "$SHARED" --gpu \
   --out $R/examples/results/storage/lance.json "$@"
rm -rf $LOCAL $SHARED
