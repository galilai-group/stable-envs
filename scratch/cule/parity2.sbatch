#!/bin/bash
#SBATCH --job-name=culeparity2
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --exclude=gb-west-a1-gpu-047,gb-west-a1-gpu-070,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/cule/parity2.%j.out
SCRATCH=/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/cule
VENV=/mnt/vast/home/randall.balestriero/venvs/cule
export CUDA_HOME=$SCRATCH/cudahome
source "$VENV/bin/activate"
case "$(command -v python)" in "$VENV"/*) ;; *) echo "FATAL python=$(command -v python)"; exit 1;; esac
cd "$SCRATCH/cule_mehrdad"
ROMS=pong,breakout,space_invaders,seaquest,asterix,qbert,ms_pacman,freeway,boxing,road_runner,chopper_command,alien,amidar,robotank,montezuma_revenge,riverraid,krull,up_n_down,kangaroo,berzerk,gravitar,bank_heist,fishing_derby,hero
python -u "$SCRATCH/parity_cule_ale.py" "$SCRATCH/parity_cule.json" --steps 200 --roms "$ROMS"
echo "PARITY_EXIT=$?"
echo "=== LONG HORIZON 2000 steps ==="
python -u "$SCRATCH/parity_cule_ale.py" "$SCRATCH/parity_cule_long.json" --steps 2000 \
  --roms pong,breakout,seaquest,qbert,asterix,chopper_command,amidar,berzerk
echo "PARITY_LONG_EXIT=$?"
