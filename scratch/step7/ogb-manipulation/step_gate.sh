# THE GATE, as run inside the held allocation: `run_in_alloc.sh gate_final step_gate.sh <md5>`.
#
# The outcome greps are anchored to the END of the line on purpose. The supervisor runs pytest with
# -v, which prints `path::test[Env] FAILED [ 76%]` -- the outcome is neither at the start of the line
# nor the last field. An earlier version of this script grepped `^(FAILED|ERROR)` and printed
# "NO OGB FAILURES" for a run that had two OGB failures and 798 errors overall: it matched nothing at
# all, and matching nothing looked exactly like passing.
set -uo pipefail
cd "$WS"
echo "=== Gate A, both cells ==="
"$VENV/bin/python" scratch/step7/ogb-manipulation/probe_gate_a.py --out "$SE7_OUT" \
  --cells OGBCube OGBScene > "$SE7_OUT/gate_a.txt" 2>&1; echo "GATE A rc=$?"
tail -12 "$SE7_OUT/gate_a.txt"
echo "=== THE GATE: the three tests/gpu files WHOLE, through the supervisor ==="
mkdir -p "$SE7_OUT/run"
"$VENV/bin/python" -u -m stable_envs.testing.supervisor \
  --rundir "$SE7_OUT/run" --json "$SE7_OUT/run/result.json" \
  -- tests/gpu/test_conformance.py tests/gpu/test_reproducibility.py tests/gpu/test_fast_paths.py \
  -v -rs --durations=25 > "$SE7_OUT/gputests_full.txt" 2>&1
echo "SUITE rc=$?"
echo "--- suite line ---";  grep -E "SUMMARY:" "$SE7_OUT/gputests_full.txt" | tail -1
echo "--- resources ---";   grep -E "resources:" "$SE7_OUT/gputests_full.txt" | tail -1
echo "--- OGB rows, tallied by outcome ---"
grep -E "OGBCube|OGBScene" "$SE7_OUT/gputests_full.txt" | grep -oE " (PASSED|FAILED|ERROR|SKIPPED)" \
  | sort | uniq -c
echo "--- any OGB row not passing (empty means the gate is met) ---"
grep -E "OGBCube|OGBScene" "$SE7_OUT/gputests_full.txt" | grep -E "FAILED|ERROR" || echo "  none"
