#!/bin/bash
# One PROCESS per (scene, resolution): nvdiffrast answers "too big" with an illegal memory access that
# kills the interpreter, not with an exception, so a shared process would lose every row after the
# first scene that does not fit.
V=/mnt/vast/home/randall.balestriero/venvs/senvdiff/bin/python
cd /mnt/vast/home/randall.balestriero/code/stable-envs
for R in 224 64; do
  for S in antmaze-medium humanoidmaze-medium cube-single robocasa-L52S03; do
    echo "===== $S @ $R"
    $V scratch/nvdiff/bench.py --res $R --only $S 2>&1 | grep -E "^@@|Traceback|Error|line [0-9]+, in" | head -20
  done
done
