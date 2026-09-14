#!/bin/bash
# Pull Oscar's artifacts back WITHOUT clobbering the files this tree owns.
#
# The plain `rsync $SCR/.../step6/gfxdraw/ examples/results/step6/gfxdraw/` does clobber them: the
# push excludes examples/results/step6/, so Oscar never sees a findings.md written here, and the
# pull then overwrites the local one with Oscar's older copy. It cost one restore from git, which
# is the only reason it cost nothing. Artifacts flow one way; prose flows the other.
set -euo pipefail
LOCAL=${1:-/Users/samib/stable-envs-step6-gfxdraw}
rsync -az --exclude findings.md \
  sboughan@sshcampus.ccv.brown.edu:/oscar/scratch/sboughan/se6/gfxdraw/examples/results/step6/gfxdraw/ \
  "$LOCAL/examples/results/step6/gfxdraw/"
echo "pulled (findings.md is written here and is never pulled)"
