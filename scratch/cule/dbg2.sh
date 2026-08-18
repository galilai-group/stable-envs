#!/bin/bash
SCRATCH=/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/cule
VENV=/mnt/vast/home/randall.balestriero/venvs/cule
export CUDA_HOME=$SCRATCH/cudahome
source "$VENV/bin/activate"
cd "$SCRATCH/cule_mehrdad"
python -u - <<'PY'
import json, ale_py
from ale_py import roms
from torchcule.atari import Env
out={}
for rom in ["pong","amidar","bank_heist","breakout","seaquest","hero","kangaroo","freeway","up_n_down","krull"]:
    a=ale_py.ALEInterface(); a.setLoggerMode(ale_py.LoggerMode.Error); a.loadROM(roms.get_rom_path(rom))
    e=Env(rom,1,color_mode="gray",device="cpu",rescale=True,frameskip=1,repeat_prob=0.0,max_noop_steps=1)
    out[rom]={"ale_dims":list(a.getScreenDims()),"cule_dims":[e.cart.screen_height(),e.cart.screen_width()]}
    del e,a
print(json.dumps(out,indent=1))
json.dump(out,open("/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/cule/screen_dims.json","w"),indent=1)
PY
python -u "$SCRATCH/parity_cule_ale.py" "$SCRATCH/parity_cule_extra.json" --steps 200 --roms amidar,bank_heist
echo "EXIT=$?"
