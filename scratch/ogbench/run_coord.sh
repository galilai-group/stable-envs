#!/bin/bash
export PYTHONPATH=/mnt/vast/home/randall.balestriero/code/stable-envs
export SDL_VIDEODRIVER=dummy
cd /mnt/vast/home/randall.balestriero/code/stable-envs
L=/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/ogbench/out
/mnt/vast/home/randall.balestriero/miniconda3/envs/test/bin/python -m pytest tests/ -q --tb=line >"$L/coord_full.log" 2>&1
echo "EXIT=$?" >>"$L/coord_full.log"
# only the sufficiency test, my cells: does it fail on its own?
/mnt/vast/home/randall.balestriero/miniconda3/envs/test/bin/python -m pytest tests/ -q --tb=line \
  -k "test_state_is_sufficient and (Pointmaze or Antmaze or Humanoidmaze)" >"$L/coord_suff_only.log" 2>&1
echo "EXIT=$?" >>"$L/coord_suff_only.log"
# one cell, all its tests
/mnt/vast/home/randall.balestriero/miniconda3/envs/test/bin/python -m pytest tests/ -q --tb=line \
  -k "AntmazeMediumNavigate" >"$L/coord_one_cell.log" 2>&1
echo "EXIT=$?" >>"$L/coord_one_cell.log"
# my cells only, all tests
/mnt/vast/home/randall.balestriero/miniconda3/envs/test/bin/python -m pytest tests/ -q --tb=line \
  -k "Pointmaze or Antmaze or Humanoidmaze" >"$L/coord_mine.log" 2>&1
echo "EXIT=$?" >>"$L/coord_mine.log"
