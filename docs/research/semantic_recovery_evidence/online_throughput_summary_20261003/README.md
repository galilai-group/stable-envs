# Measured stepping, prepared-State rendering and online throughput

This reconciles existing, terminal-verified source campaigns; it does not rerun timings. All selected public step cells use B64, T8, no intermediate State history and mandatory final owned State. Public stepping-only requests no frames; combined requests every frame. Render-only uses a prepared full State at B64, including normal validation/projection and output delivery, without rollout or disk I/O. Rates are total environment transitions or emitted images per second, not batches per second. All SDs are sample SDs over six timing samples.

The CSV contains all eleven completed rendered registrations and the three PickAndPlace no-frame registrations. No matching standalone render component was measured in the completed five-classic source campaign, so those component cells remain blank. PickAndPlace render cells remain blank because its upstream rendering changes continuation. Classic rows use their registration-default image dimensions; the six fully measured DMC/Navigation rows below use 64×64.

The raw component columns preserve their actual contracts: DMC task stepping includes IPC/output delivery but no State capture; Navigation task stepping includes its native State result. These raw operations are single-action B64 calls. The public stepping-only column is the comparable eight-action API workload and amortizes dispatch/State overhead differently. Do not treat the raw operation as bare physics or algebraically add its reciprocal to render timing.

DMC GPU-output rows still use CPU physics, with GPU rendering and CUDA outputs. Navigation uses native Torch execution on the selected device. Navigation can finish individual lanes early: TwoRoom, Piecewise and SimplePointMaze have 491, 498 and 505 valid transitions out of 512 requested slots. Stepping rates count valid transitions; all-frame FPS includes the defined held terminal frames. Separate combined valid-step rates are retained in the CSV.

| Environment | Device | Public step-only steps/s | Render-only frames/s | Combined frames/s |
| --- | --- | ---: | ---: | ---: |

| swm/AcrobotDMControl-v0 | cpu | 24,102 ± 2,995 | 857 ± 22 | 782 ± 20 |

| swm/AcrobotDMControl-v0 | cuda | 22,903 ± 170 | 6,898 ± 369 | 4,932 ± 51 |

| swm/PendulumDMControl-v0 | cpu | 32,666 ± 803 | 943 ± 36 | 820 ± 16 |

| swm/PendulumDMControl-v0 | cuda | 28,498 ± 944 | 7,013 ± 404 | 5,325 ± 59 |

| swm/CartpoleDMControl-v0 | cpu | 25,685 ± 964 | 2,313 ± 153 | 1,972 ± 6 |

| swm/CartpoleDMControl-v0 | cuda | 24,575 ± 145 | 9,262 ± 451 | 6,175 ± 148 |

| swm/TwoRoom-v1 | cpu | 44,674 ± 148 | 5,453 ± 361 | 4,600 ± 267 |

| swm/TwoRoom-v1 | cuda | 24,039 ± 219 | 38,994 ± 1,875 | 15,505 ± 121 |

| swm/Piecewise-v0 | cpu | 56,756 ± 301 | 5,257 ± 371 | 4,220 ± 68 |

| swm/Piecewise-v0 | cuda | 34,420 ± 259 | 62,989 ± 549 | 22,584 ± 169 |

| swm/SimplePointMaze-v0 | cpu | 6,352 ± 27 | 1,202 ± 64 | 987 ± 16 |

| swm/SimplePointMaze-v0 | cuda | 21,984 ± 159 | 6,337 ± 82 | 5,167 ± 7 |


Full precision, raw-step/export components, mean/SD, provenance and missing cells are in [throughput.csv](tables/throughput.csv). `inputs.json` hashes every input CSV. Reproduce using `summarize.py --output <new-directory>`. Cube/Scene internal validator controls have a different State-history workload and are deliberately kept in their separate accepted-optimization evidence; they are not substituted for source benchmark rows.
