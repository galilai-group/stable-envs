#!/bin/bash
# Create the isolated venv (if missing) and build CuLE against CUDA 13 / torch cu130.
# Run under srun/sbatch with a GPU: setup.py derives the SM arch from the live device.
set -x
SCRATCH=/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/cule
VENV=/mnt/vast/home/randall.balestriero/venvs/cule
export CUDA_HOME=$SCRATCH/cudahome
export PATH=$CUDA_HOME/bin:$PATH
export CULE_THRUST_INCLUDE=$SCRATCH/thrust_1_9_10

nvidia-smi --query-gpu=name,compute_cap --format=csv
gcc --version | head -n 1
nvcc --version | tail -n 2

if [ ! -x "$VENV/bin/pip" ]; then
  source "$SCRATCH/make_venv.sh" "$VENV" || exit 1
  pip install -q --upgrade pip wheel setuptools
  pip install -q torch --index-url https://download.pytorch.org/whl/cu130
  pip install -q numpy "ale-py>=0.10" gymnasium pybind11 cython psutil tqdm opencv-python-headless
else
  source "$VENV/bin/activate"
fi

# never fall through to a shared conda env
case "$(command -v python)" in "$VENV"/*) ;; *) echo "FATAL python=$(command -v python)"; exit 1;; esac
case "$(command -v pip)"    in "$VENV"/*) ;; *) echo "FATAL pip=$(command -v pip)";       exit 1;; esac

python -V
python -c "import torch;print('torch',torch.__version__,torch.version.cuda,'cuda_ok',torch.cuda.is_available(),torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)"
python -c "import ale_py;print('ale_py',ale_py.__version__)"

cd "$SCRATCH/cule_mehrdad"
echo "=== BUILD START $(date -Is) ==="
pip install --no-build-isolation -v -e . > "$SCRATCH/build_mehrdad.log" 2>&1
echo "BUILD_EXIT=$?"
echo "=== BUILD END $(date -Is) ==="
tail -n 80 "$SCRATCH/build_mehrdad.log"
python -c "import torchcule_atari, torchcule; print('IMPORT_OK', torchcule_atari.__file__)"
echo "IMPORT_EXIT=$?"
