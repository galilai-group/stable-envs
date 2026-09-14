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
# A file quarantined here (superseded_*/, rejected_*/, wave1_*/) still exists at its ORIGINAL path
# on Oscar, because a pull cannot know it was moved -- so the next pull recreates the top-level copy
# and the generator sees the bad shard again beside the good one. Catch that here rather than
# discover it in a report.
dupes=0
while IFS= read -r q; do
  base=$(basename "$q")
  dir=$(dirname "$(dirname "$q")")
  if [ -f "$dir/$base" ]; then
    echo "REMOVING $dir/$base -- it is quarantined at $q"
    rm -f "$dir/$base"
    dupes=$((dupes + 1))
  fi
done < <(find "$LOCAL/examples/results/step6/gfxdraw" -path '*/superseded_*/*.json' \
         -o -path '*/rejected_*/*.json' -o -path '*/wave1_*/*.json')
# and re-apply the structural exclusions, which a pull undoes for any shard it recreates
python3 "$(dirname "$0")/quarantine.py" || true
echo "pulled (findings.md is written here and is never pulled; $dupes quarantine duplicate(s) removed)"
