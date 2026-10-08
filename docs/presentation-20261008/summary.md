# Presentation results

Live collection: blank or pending cells are not zero performance. Craftax is excluded.

Speedup is ours FPS divided by source FPS. Medians include only complete, accepted pairs. GPU comparisons below require a real source GPU renderer; CPU-source upload fallbacks are counted separately.

## Saved-state rendering

| Batch | Horizon | CPU complete / 38 | CPU faster | Median CPU speedup | GPU complete / 38 | GPU faster / eligible | Median GPU speedup | CPU fallback pairs | Pending cells |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 37 | 10 | 0.62× | 37 | 5/27 | 0.36× | 10 | 0 |
| 64 | 1 | 37 | 19 | 1.00× | 37 | 25/27 | 5.34× | 10 | 0 |
| 256 | 1 | 37 | 18 | 1.00× | 37 | 25/27 | 7.85× | 10 | 0 |
| 1024 | 1 | 31 | 14 | 0.97× | 37 | 25/27 | 17.25× | 10 | 5 |
| 2048 | 1 | 29 | 13 | 0.93× | 37 | 25/27 | 10.99× | 10 | 7 |

## Online stepping and rendering

| Batch | Horizon | CPU complete / 38 | CPU faster | Median CPU speedup | GPU complete / 38 | GPU faster / eligible | Median GPU speedup | CPU fallback pairs | Pending cells |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 35 | 20 | 1.11× | 35 | 5/24 | 0.59× | 11 | 0 |
| 1 | 8 | 35 | 20 | 1.04× | 32 | 1/21 | 0.46× | 11 | 0 |
| 64 | 1 | 33 | 15 | 0.83× | 32 | 20/21 | 2.60× | 11 | 0 |
| 64 | 8 | 33 | 14 | 0.77× | 32 | 18/21 | 2.29× | 11 | 0 |
| 256 | 1 | 30 | 13 | 0.86× | 28 | 18/19 | 4.86× | 9 | 9 |
| 256 | 8 | 30 | 14 | 0.88× | 25 | 15/16 | 4.48× | 9 | 9 |
| 1024 | 1 | 5 | 1 | 0.91× | 5 | 0/0 | — | 5 | 65 |
| 1024 | 8 | 1 | 1 | 1.39× | 5 | 0/0 | — | 5 | 65 |
| 2048 | 1 | 5 | 1 | 0.95× | 5 | 0/0 | — | 5 | 66 |
| 2048 | 8 | 0 | 0 | — | 0 | 0/0 | — | 0 | 66 |

## Stepping without rendering

| Batch | Horizon | CPU complete / 38 | CPU faster | Median CPU speedup | GPU complete / 38 | GPU faster / eligible | Median GPU speedup | CPU fallback pairs | Pending cells |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 38 | 1 | 0.51× | 38 | 0/0 | — | 38 | 0 |
| 1 | 8 | 38 | 4 | 0.54× | 38 | 0/0 | — | 38 | 0 |
| 64 | 1 | 38 | 21 | 1.16× | 38 | 0/0 | — | 38 | 0 |
| 64 | 8 | 38 | 19 | 0.99× | 38 | 0/0 | — | 38 | 0 |
| 256 | 1 | 33 | 22 | 1.14× | 34 | 0/0 | — | 34 | 9 |
| 256 | 8 | 33 | 21 | 1.28× | 34 | 0/0 | — | 34 | 9 |
| 1024 | 1 | 5 | 0 | 0.15× | 6 | 0/0 | — | 6 | 65 |
| 1024 | 8 | 5 | 0 | 0.08× | 6 | 0/0 | — | 6 | 65 |
| 2048 | 1 | 5 | 0 | 0.12× | 5 | 0/0 | — | 5 | 66 |
| 2048 | 8 | 5 | 0 | 0.09× | 5 | 0/0 | — | 5 | 66 |

All pinned source simulation here is CPU-based. The GPU configuration includes GPU result delivery; it is not source GPU physics. See the detailed table for those rates.

See [measurement definitions](README.md), [the slowdown diagnosis and fix plan](bottlenecks.md), and the detailed CSVs for MSE, standard deviations, resource usage, failed gates and device labels.
