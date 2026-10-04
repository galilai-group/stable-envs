# Correctness-gate scheduling correction, 2026-10-04

The Rocket CPU-render cost job `6992404_1` reserved an RTX 3090 solely to
match the benchmark host profile. Together with GPU cost job `6992404_3`,
it occupied the two-user-GPU `norm-gpu` allowance and blocked public PushT
acceptance job `6993756`.

Stopped only `6992404_1` after preserving its three completed scenarios
(reset, 25 percent, 50 percent; twelve branches each). The saved
`cpu-b16-before-stop.json` is explicitly a **partial** report, not a completed
benchmark. Original remote outputs remain intact. No timings are extrapolated
or combined with a different host run.

Job `6993756` then started on `gpu2108`. Its unit suite passed 473 tests with
zero skips; independent source comparisons follow in the same job. Passing
the unit suite alone does not promote either registration.

The Rocket GPU job remains running. The identical PushT fallback submission
`6994833` remains queued on the preferred `3090-gcondo` partition. Future
correctness gates take priority over long performance sweeps. The earlier
held tail `6794187_[30-41]` was separately canceled; its completed tasks and
evidence were preserved, and removing held tasks did not free GPU quota.

`priority-decision.json` records the production revision and partial-report
hash. This directory does not modify any previously committed evidence.
