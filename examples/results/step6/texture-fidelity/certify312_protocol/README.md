# certify312_protocol — interpreter provenance

This record (job 6365695, gpu2109) was taken in `$SCR/venvs/se-step5` (mujoco 3.12.0, mujoco_warp
3.12.0, dm_control 1.0.45, warp 1.17.0). Its `digest_ours` values for the four DMC visual cells
(`d314a7ea…`, `80eb4938…`, `c32a0829…`, `1801687…`) are the se-step5 renders, which do NOT reproduce
the frozen Step-4 `digest_ours` (`1fc511ea…`, `13b88aad…`, …). The interpreter that reproduces the
frozen corpus byte for byte is `se-step3` (MJWARP-1); in that interpreter the same 3.12 arm gives
`check_resolution_consistency` 0.52223 (Cheetah) and 0.61258 (Walker) on the 8-state protocol, i.e.
three FAILs against the 0.005 margin, where this record shows 0.53385 / 0.62895 and no FAIL
(`examples/results/step6/verify/findings.md`, jobs 6365966/6365967).

Consequence: the 3.12 arm is interpreter-dependent and "no FAIL under 3.12" must not be asserted from
this record alone. The contract judges these cells under the declared 3.11 pair, where every arm
(this workstream's `certify311_protocol`, the verify workstream's cuda and CPU arms) agrees on PASS x4.
The visual-tier BLOCKED verdicts here (version mismatch) are unaffected by the interpreter.
