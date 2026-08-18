#!/bin/bash
# Create an isolated venv for the CuLE / envpool measurements.
set -x
VENV=/mnt/vast/home/randall.balestriero/venvs/cule
rm -rf "$VENV"
/usr/bin/python3.12 -m venv "$VENV"
source "$VENV/bin/activate"
python -V
pip install --upgrade pip wheel setuptools
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install numpy "ale-py>=0.10" "gymnasium[atari]" pybind11 cython psutil tqdm opencv-python-headless
python -c "import torch, ale_py, gymnasium; print('torch', torch.__version__, torch.version.cuda); print('ale_py', ale_py.__version__); print('gymnasium', gymnasium.__version__)"
echo "SETUP_DONE=$?"
