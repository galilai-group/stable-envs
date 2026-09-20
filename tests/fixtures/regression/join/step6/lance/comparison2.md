# Step 6 — our renderer vs the production 8-worker JPEG-in-Lance pipeline (Oscar)

Every config at **its own certified resolution**. JPEG **q95**, chunk **256**, `examples/bench_lance.py` semantics (its timing functions imported verbatim).

Two break-even columns are kept separate throughout and neither replaces the other:
**(1) frozen conservative** = our fps / one core of *in-memory q90* decode (`certification.md` §5.1/§6.3, taken from the committed shards' `decode` block); **(2) realistic pipeline** = our fps / the 8-worker stored-JPEG pipeline measured here.


## T1 — the stored baseline, per storage arm

Output of every CPU arm is **uint8 CHW tensors in HOST memory**. `cores used` is `sum(worker CPU seconds) / slowest-worker elapsed`, i.e. how many cores the 8 workers actually burned.

| config | frame | kB/f | tmpfs 1w | tmpfs 8w | NFS 1w | NFS 8w | GPFS 1w | GPFS 8w | cores used (8w) | JPEG read MB/s | host-out MB/s (8w) | read frac | 1-proc best-of-8 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Acrobot | 500x500 | 10.6 |       852 |     5,984 |       847 |     6,642 |       847 |     6,718 |   7.91 | 762.64 | 4,487.64 | 1.19% |     1,439 |
| CartPole | 400x600 | 8.1 |       909 |     7,169 |       832 |     6,845 |       890 |     6,223 |   7.91 | 708.45 | 5,161.71 | 1.05% |     1,522 |
| CraftaxClassic | 112x144 | 10.8 |     4,191 |    29,218 |     4,155 |    31,782 |     3,911 |    30,634 |   7.86 | 658.55 | 1,413.67 | 6.77% |     5,089 |
| DMCAcrobotSwingup | 84x84 | 3.1 |    10,052 |    76,941 |     9,963 |    73,217 |    10,125 |    82,069 |   7.80 | 331.48 | 1,628.69 | 8.74% |    11,532 |
| DMCBallInCupCatch | 84x84 | 3.0 |     9,934 |    69,678 |     9,556 |    74,105 |    10,006 |    76,544 |   7.86 | 394.19 | 1,474.94 | 7.61% |    11,332 |
| DMCCartpoleBalance | 84x84 | 2.9 |    10,174 |    78,343 |     9,457 |    69,347 |     8,893 |    71,200 |   7.60 | 330.79 | 1,658.36 | 8.80% |    11,662 |
| DMCCartpoleSwingup | 84x84 | 3.0 |    10,268 |    77,276 |    10,158 |    76,753 |    10,762 |    81,350 |   7.76 | 330.64 | 1,635.77 | 8.98% |    11,988 |
| DMCCheetahRun | 84x84 | 4.6 |     7,823 |    61,453 |     8,331 |    62,996 |     8,381 |    64,669 |   7.72 | 508.88 | 1,300.84 | 7.54% |     9,602 |
| DMCFingerTurnHard | 84x84 | 3.7 |     8,508 |    68,277 |     8,695 |    73,866 |     9,764 |    72,585 |   7.79 | 576.16 | 1,445.28 | 6.17% |    10,439 |
| DMCHopperHop | 84x84 | 4.4 |     9,091 |    68,715 |     8,993 |    68,802 |     9,067 |    68,952 |   7.86 | 816.69 | 1,454.56 | 4.90% |     9,840 |
| DMCHumanoidStand | 84x84 | 3.8 |     9,908 |    73,927 |     9,821 |    70,779 |     9,904 |    75,679 |   7.81 | 738.90 | 1,564.88 | 4.98% |    10,565 |
| DMCPendulumSwingup | 84x84 | 2.5 |    12,169 |    87,027 |    12,005 |    86,032 |    12,162 |    92,218 |   7.41 | 555.02 | 1,842.19 | 5.33% |    12,860 |
| DMCReacherHard | 84x84 | 2.4 |    12,133 |    92,201 |    11,972 |    91,317 |    12,157 |    92,404 |   7.85 | 535.01 | 1,951.72 | 5.40% |    13,042 |
| DMCWalkerWalk | 84x84 | 4.3 |     9,204 |    68,783 |     9,085 |    69,661 |     9,215 |    69,628 |   7.72 | 849.55 | 1,456.01 | 4.66% |     9,971 |
| FetchPickAndPlace | 480x480 | 25.9 |       727 |     5,504 |       729 |     5,721 |       732 |     5,774 |   7.85 | 994.22 | 3,804.59 | 1.92% |     1,101 |
| FetchPickAndPlaceDense | 480x480 | 25.9 |       729 |     5,441 |       704 |     5,340 |       698 |     5,363 |   7.92 | 1,047.30 | 3,760.93 | 1.81% |     1,106 |
| FetchPickAndPlaceDict | 480x480 | 25.9 |       690 |     5,379 |       732 |     5,715 |       724 |     5,737 |   7.90 | 895.69 | 3,717.77 | 2.12% |     1,104 |
| FetchPush | 480x480 | 25.9 |       694 |     5,731 |       728 |     5,697 |       726 |     5,768 |   7.92 | 1,097.24 | 3,961.37 | 1.73% |     1,109 |
| FetchPushDense | 480x480 | 25.9 |       683 |     5,487 |       687 |     5,353 |       709 |     5,198 |   7.90 | 1,110.88 | 3,792.59 | 1.71% |     1,105 |
| FetchPushDict | 480x480 | 25.9 |       696 |     5,467 |       727 |     5,765 |       730 |     5,761 |   7.92 | 977.12 | 3,778.96 | 1.94% |     1,109 |
| FetchReach | 480x480 | 25.5 |       718 |     5,630 |       732 |     5,735 |       733 |     5,784 |   7.92 | 1,000.27 | 3,891.64 | 1.88% |     1,109 |
| FetchReachDense | 480x480 | 25.5 |       707 |     5,216 |       704 |     5,427 |       695 |     5,266 |   7.89 | 985.39 | 3,605.59 | 1.90% |     1,111 |
| FetchReachDict | 480x480 | 25.5 |       709 |     5,195 |       718 |     5,741 |       737 |     5,763 |   7.88 | 994.25 | 3,590.55 | 1.86% |     1,111 |
| FetchSlide | 480x480 | 26.7 |       725 |     5,161 |       725 |     5,516 |       725 |     5,483 |   7.70 | 1,081.15 | 3,567.26 | 1.80% |     1,093 |
| FetchSlideDense | 480x480 | 26.7 |       703 |     5,227 |       694 |     5,118 |       681 |     5,197 |   7.92 | 1,003.35 | 3,612.93 | 1.94% |     1,098 |
| FetchSlideDict | 480x480 | 26.7 |       700 |     5,039 |       708 |     5,524 |       728 |     5,557 |   7.73 | 994.50 | 3,483.13 | 1.94% |     1,097 |
| MountainCar | 400x600 | 13.1 |       862 |     6,385 |       859 |     6,560 |       855 |     6,638 |   7.83 | 833.05 | 4,597.46 | 1.36% |     1,416 |
| MountainCarContinuous | 400x600 | 13.1 |       860 |     6,797 |       818 |     6,453 |       827 |     5,985 |   7.92 | 814.40 | 4,893.67 | 1.40% |     1,410 |
| Pendulum | 500x500 | 8.6 |       822 |     6,581 |       795 |     6,283 |       849 |     6,464 |   7.88 | 681.26 | 4,935.74 | 1.10% |     1,455 |
| Piecewise | 224x224 | 3.2 |     3,460 |    27,120 |     3,581 |    28,340 |     3,611 |    27,170 |   7.69 | 388.58 | 4,082.32 | 2.93% |     4,872 |
| PiecewiseZones | 224x224 | 4.9 |     3,302 |    26,024 |     3,293 |    24,926 |     3,307 |    25,993 |   7.94 | 830.73 | 3,917.30 | 1.95% |     4,049 |
| PointmazeLargeNavigate | 224x224 | 12.3 |     2,464 |    19,184 |     2,411 |    18,660 |     2,421 |    18,927 |   7.92 | 1,219.96 | 2,887.80 | 2.50% |     3,048 |
| PushT | 96x96 | 2.7 |     9,351 |    73,948 |     7,974 |    61,355 |     7,806 |    58,959 |   7.93 | 435.97 | 2,044.53 | 5.63% |     9,958 |
| SimplePointMaze | 500x500 | 12.1 |       841 |     6,502 |       844 |     6,617 |       843 |     6,660 |   7.80 | 772.93 | 4,876.16 | 1.33% |     1,407 |
| TwoRoom | 224x224 | 4.4 |     3,182 |    25,057 |     3,144 |    23,183 |     3,315 |    24,988 |   7.79 | 483.12 | 3,771.85 | 3.02% |     4,619 |
| TwoRoomTarget | 224x224 | 4.8 |     3,129 |    24,067 |     2,943 |    24,103 |     3,203 |    24,972 |   7.91 | 535.54 | 3,622.81 | 2.95% |     4,185 |

## T2 — CPU: our renderer vs the stored pipeline, same node, same job

| config | lance 1 worker | lance 8w | lance 8w / 8 | ours 1 thread | ours 8 threads | ours 8 procs | best 8-way | scaling eff. | **cpu_pipeline_ratio** (8-way / lance 8w) | 1thr / lance 1w | 1thr / per-core |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Acrobot |       852 |     5,984 |       748 |       351 |     1,068 |     2,035 |     2,035 |   0.85 | **  0.34** |   0.41 |   0.47 |
| CartPole |       909 |     7,169 |       896 |       504 |     1,537 |     2,428 |     2,428 |   0.73 | **  0.34** |   0.55 |   0.56 |
| CraftaxClassic |     4,191 |    29,218 |     3,652 |   103,130 |   133,397 |   203,349 |   203,349 |   0.30 | **  6.96** |  24.61 |  28.24 |
| DMCAcrobotSwingup |    10,052 |    76,941 |     9,618 |       256 |     1,525 |     2,180 |     2,180 |   1.18 | **  0.03** |   0.03 |   0.03 |
| DMCBallInCupCatch |     9,934 |    69,678 |     8,710 |        62 |       207 |       393 |       393 |   1.08 | **  0.01** |   0.01 |   0.01 |
| DMCCartpoleBalance |    10,174 |    78,343 |     9,793 |       526 |     2,901 |     3,763 |     3,763 |   0.84 | **  0.05** |   0.05 |   0.05 |
| DMCCartpoleSwingup |    10,268 |    77,276 |     9,659 |       588 |     2,563 |     2,943 |     2,943 |   0.69 | **  0.04** |   0.06 |   0.06 |
| DMCCheetahRun |     7,823 |    61,453 |     7,682 |       182 |       212 |     1,146 |     1,146 |   0.95 | **  0.02** |   0.02 |   0.02 |
| DMCFingerTurnHard |     8,508 |    68,277 |     8,535 |        97 |       480 |       645 |       645 |   1.15 | **  0.01** |   0.01 |   0.01 |
| DMCHopperHop |     9,091 |    68,715 |     8,589 |       216 |       285 |     1,486 |     1,486 |   0.91 | **  0.02** |   0.02 |   0.03 |
| DMCHumanoidStand |     9,908 |    73,927 |     9,241 |        53 |        86 |       387 |       387 |   0.98 | **  0.01** |   0.01 |   0.01 |
| DMCPendulumSwingup |    12,169 |    87,027 |    10,878 |       203 |     1,075 |     1,428 |     1,428 |   0.91 | **  0.02** |   0.02 |   0.02 |
| DMCReacherHard |    12,133 |    92,201 |    11,525 |        96 |       609 |       755 |       755 |   1.11 | **  0.01** |   0.01 |   0.01 |
| DMCWalkerWalk |     9,204 |    68,783 |     8,598 |        68 |       133 |       455 |       455 |   0.92 | **  0.01** |   0.01 |   0.01 |
| FetchPickAndPlace |       727 |     5,504 |       688 |         - |         - |         - |         - |      - | **     -** |      - |      - |
| FetchPickAndPlaceDense |       729 |     5,441 |       680 |         - |         - |         - |         - |      - | **     -** |      - |      - |
| FetchPickAndPlaceDict |       690 |     5,379 |       672 |         - |         - |         - |         - |      - | **     -** |      - |      - |
| FetchPush |       694 |     5,731 |       716 |         - |         - |         - |         - |      - | **     -** |      - |      - |
| FetchPushDense |       683 |     5,487 |       686 |         - |         - |         - |         - |      - | **     -** |      - |      - |
| FetchPushDict |       696 |     5,467 |       683 |         - |         - |         - |         - |      - | **     -** |      - |      - |
| FetchReach |       718 |     5,630 |       704 |         - |         - |         - |         - |      - | **     -** |      - |      - |
| FetchReachDense |       707 |     5,216 |       652 |         - |         - |         - |         - |      - | **     -** |      - |      - |
| FetchReachDict |       709 |     5,195 |       649 |         - |         - |         - |         - |      - | **     -** |      - |      - |
| FetchSlide |       725 |     5,161 |       645 |         - |         - |         - |         - |      - | **     -** |      - |      - |
| FetchSlideDense |       703 |     5,227 |       653 |         - |         - |         - |         - |      - | **     -** |      - |      - |
| FetchSlideDict |       700 |     5,039 |       630 |         - |         - |         - |         - |      - | **     -** |      - |      - |
| MountainCar |       862 |     6,385 |       798 |       516 |     1,817 |     2,794 |     2,794 |   0.80 | **  0.44** |   0.60 |   0.65 |
| MountainCarContinuous |       860 |     6,797 |       850 |       493 |     1,823 |     2,798 |     2,798 |   0.80 | **  0.41** |   0.57 |   0.58 |
| Pendulum |       822 |     6,581 |       823 |       505 |     1,975 |     2,944 |     2,944 |   0.86 | **  0.45** |   0.62 |   0.61 |
| Piecewise |     3,460 |    27,120 |     3,390 |     1,333 |     6,426 |     7,681 |     7,681 |   0.90 | **  0.28** |   0.39 |   0.39 |
| PiecewiseZones |     3,302 |    26,024 |     3,253 |     1,124 |     4,069 |     4,716 |     4,716 |   0.68 | **  0.18** |   0.34 |   0.35 |
| PointmazeLargeNavigate |     2,464 |    19,184 |     2,398 |     1,189 |     7,837 |     8,990 |     8,990 |   0.96 | **  0.47** |   0.48 |   0.50 |
| PushT |     9,351 |    73,948 |     9,244 |     2,218 |     3,011 |     6,187 |     6,187 |   0.51 | **  0.08** |   0.24 |   0.24 |
| SimplePointMaze |       841 |     6,502 |       813 |       277 |     1,160 |     1,890 |     1,890 |   0.95 | **  0.29** |   0.33 |   0.34 |
| TwoRoom |     3,182 |    25,057 |     3,132 |     1,184 |     4,985 |     8,134 |     8,134 |   1.10 | **  0.32** |   0.37 |   0.38 |
| TwoRoomTarget |     3,129 |    24,067 |     3,008 |       585 |     2,212 |     3,345 |     3,345 |   0.92 | **  0.14** |   0.19 |   0.19 |

## T3 — GPU: both sides end with frames ON THE DEVICE

| config | JPEG→device, 8-worker loader + H2D | JPEG→device, nvJPEG | H2D alone (pinned) | ours renderer-only | ours end-to-end (states→device) | **gpu_pipeline_ratio** | vs nvJPEG | renderer-only / loader8+H2D |
|---|---|---|---|---|---|---|---|---|
| Acrobot |     3,719 |     4,931 |    35,039 |   160,041 |   151,860 | ** 40.84** |  30.79 |  43.04 |
| CartPole |     3,884 |     5,968 |    36,301 |   185,798 |   171,559 | ** 44.17** |  28.75 |  47.84 |
| CraftaxClassic |    24,957 |     6,763 |   490,769 | 1,345,537 |   840,567 | ** 33.68** | 124.30 |  53.91 |
| DMCAcrobotSwingup |    53,029 |    13,095 |   884,438 |    36,066 |    36,134 | **  0.68** |   2.76 |   0.68 |
| DMCBallInCupCatch |    51,374 |    13,072 |   875,541 |     9,760 |     9,375 | **  0.18** |   0.72 |   0.19 |
| DMCCartpoleBalance |    49,881 |    13,413 | 1,060,870 |    54,132 |    50,997 | **  1.02** |   3.80 |   1.09 |
| DMCCartpoleSwingup |    52,680 |    13,447 |   902,334 |    51,522 |    51,343 | **  0.97** |   3.82 |   0.98 |
| DMCCheetahRun |    43,450 |    11,173 |   811,822 |    94,570 |    90,248 | **  2.08** |   8.08 |   2.18 |
| DMCFingerTurnHard |    50,725 |    11,624 |   808,032 |    30,451 |    28,485 | **  0.56** |   2.45 |   0.60 |
| DMCHopperHop |    42,807 |    11,199 |   815,424 |    94,822 |    90,139 | **  2.11** |   8.05 |   2.22 |
| DMCHumanoidStand |    48,307 |    12,227 |   738,756 |    79,097 |    79,785 | **  1.65** |   6.53 |   1.64 |
| DMCPendulumSwingup |    54,085 |    14,920 | 1,086,276 |    44,897 |    45,211 | **  0.84** |   3.03 |   0.83 |
| DMCReacherHard |    54,839 |    14,320 |   886,669 |    25,895 |    25,931 | **  0.47** |   1.81 |   0.47 |
| DMCWalkerWalk |    44,508 |    11,087 |   809,007 |    89,310 |    85,140 | **  1.91** |   7.68 |   2.01 |
| FetchPickAndPlace |     3,850 |     2,704 |    38,011 |     5,159 |     5,141 | **  1.34** |   1.90 |   1.34 |
| FetchPickAndPlaceDense |     3,905 |     2,698 |    37,861 |     5,157 |     5,126 | **  1.31** |   1.90 |   1.32 |
| FetchPickAndPlaceDict |     3,718 |     2,703 |    37,871 |     5,124 |     5,128 | **  1.38** |   1.90 |   1.38 |
| FetchPush |     3,692 |     2,708 |    37,822 |     5,223 |     5,190 | **  1.41** |   1.92 |   1.41 |
| FetchPushDense |     3,628 |     2,728 |    37,861 |     5,188 |     5,197 | **  1.43** |   1.91 |   1.43 |
| FetchPushDict |     3,730 |     2,705 |    37,791 |     5,185 |     5,189 | **  1.39** |   1.92 |   1.39 |
| FetchReach |     3,672 |     2,866 |    37,844 |     5,181 |     5,157 | **  1.40** |   1.80 |   1.41 |
| FetchReachDense |     3,698 |     2,725 |    37,766 |     5,177 |     5,158 | **  1.39** |   1.89 |   1.40 |
| FetchReachDict |     3,749 |     2,745 |    37,813 |     5,176 |     5,134 | **  1.37** |   1.87 |   1.38 |
| FetchSlide |     3,794 |     2,664 |    37,804 |     5,289 |     5,255 | **  1.39** |   1.97 |   1.39 |
| FetchSlideDense |     3,697 |     2,653 |    37,803 |     5,252 |     5,244 | **  1.42** |   1.98 |   1.42 |
| FetchSlideDict |     3,579 |     2,666 |    37,803 |     5,257 |     5,159 | **  1.44** |   1.94 |   1.47 |
| MountainCar |     3,701 |     4,904 |    36,289 |   143,876 |   136,950 | ** 37.00** |  27.92 |  38.87 |
| MountainCarContinuous |     3,807 |     4,721 |    36,499 |   143,708 |   135,441 | ** 35.57** |  28.69 |  37.74 |
| Pendulum |     3,578 |     5,691 |    34,850 |   174,171 |   165,347 | ** 46.21** |  29.05 |  48.68 |
| Piecewise |    14,878 |    11,623 |   168,126 |    54,657 |    53,435 | **  3.59** |   4.60 |   3.67 |
| PiecewiseZones |    14,582 |     8,896 |   159,057 |    47,031 |    45,862 | **  3.15** |   5.16 |   3.23 |
| PointmazeLargeNavigate |    12,295 |     4,829 |   167,715 |   196,982 |   177,313 | ** 14.42** |  36.72 |  16.02 |
| PushT |    47,385 |    13,481 |   688,562 |   824,267 |   654,622 | ** 13.81** |  48.56 |  17.40 |
| SimplePointMaze |     3,825 |     4,467 |    34,850 |    41,776 |    41,221 | ** 10.78** |   9.23 |  10.92 |
| TwoRoom |    13,530 |    10,023 |   158,951 |    39,436 |    38,843 | **  2.87** |   3.88 |   2.91 |
| TwoRoomTarget |    13,588 |     9,053 |   163,391 |    23,727 |    23,446 | **  1.73** |   2.59 |   1.75 |

## T4 — the FROZEN conservative column, unchanged (one core of in-memory q90 decode)

This is `certification.md` §5.1/§6.3's `cores_cpu` / `cores`, reproduced from the committed shards' own `decode` block. It is **not** the pipeline ratio and must not be read as one.

| config | one-core q90 decode f/s | q90/q95 decode ratio (measured) | ours CPU 1 thread | **cores_cpu** (frozen) | ours GPU renderer-only | **cores** (frozen) |
|---|---|---|---|---|---|---|
| Acrobot |     1,313 |   1.01 |       351 | **  0.27** |   160,041 | **121.93** |
| CartPole |     1,108 |   1.01 |       504 | **  0.45** |   185,798 | **167.68** |
| CraftaxClassic |     5,459 |   1.00 |   103,130 | ** 18.89** | 1,345,537 | **246.46** |
| DMCAcrobotSwingup |     7,033 |   1.09 |       256 | **  0.04** |    36,066 | **  5.13** |
| DMCBallInCupCatch |     7,730 |   1.05 |        62 | **  0.01** |     9,760 | **  1.26** |
| DMCCartpoleBalance |     7,009 |   1.06 |       526 | **  0.08** |    54,132 | **  7.72** |
| DMCCartpoleSwingup |     7,283 |   1.06 |       588 | **  0.08** |    51,522 | **  7.07** |
| DMCCheetahRun |     7,984 |   1.13 |       182 | **  0.02** |    94,570 | ** 11.84** |
| DMCFingerTurnHard |     6,820 |   1.08 |        97 | **  0.01** |    30,451 | **  4.46** |
| DMCHopperHop |     6,652 |   1.09 |       216 | **  0.03** |    94,822 | ** 14.25** |
| DMCHumanoidStand |     8,357 |   1.09 |        53 | **  0.01** |    79,097 | **  9.46** |
| DMCPendulumSwingup |     8,063 |   1.06 |       203 | **  0.03** |    44,897 | **  5.57** |
| DMCReacherHard |     7,362 |   1.08 |        96 | **  0.01** |    25,895 | **  3.52** |
| DMCWalkerWalk |     7,767 |   1.11 |        68 | **  0.01** |    89,310 | ** 11.50** |
| FetchPickAndPlace |     1,180 |   1.04 |         - | **     -** |     5,159 | **  4.37** |
| FetchPickAndPlaceDense |     1,180 |   0.81 |         - | **     -** |     5,157 | **  4.37** |
| FetchPickAndPlaceDict |       967 |   1.05 |         - | **     -** |     5,124 | **  5.30** |
| FetchPush |     1,186 |   1.03 |         - | **     -** |     5,223 | **  4.40** |
| FetchPushDense |       978 |   1.04 |         - | **     -** |     5,188 | **  5.30** |
| FetchPushDict |       984 |   1.04 |         - | **     -** |     5,185 | **  5.27** |
| FetchReach |     1,244 |   1.04 |         - | **     -** |     5,181 | **  4.16** |
| FetchReachDense |     1,006 |   1.03 |         - | **     -** |     5,177 | **  5.14** |
| FetchReachDict |       891 |   1.04 |         - | **     -** |     5,176 | **  5.81** |
| FetchSlide |       973 |   1.04 |         - | **     -** |     5,289 | **  5.44** |
| FetchSlideDense |       940 |   1.04 |         - | **     -** |     5,252 | **  5.58** |
| FetchSlideDict |       970 |   1.01 |         - | **     -** |     5,257 | **  5.42** |
| MountainCar |     1,065 |   1.01 |       516 | **  0.48** |   143,876 | **135.04** |
| MountainCarContinuous |     1,107 |   1.01 |       493 | **  0.45** |   143,708 | **129.79** |
| Pendulum |     1,341 |   1.03 |       505 | **  0.38** |   174,171 | **129.83** |
| Piecewise |     4,500 |   1.01 |     1,333 | **  0.30** |    54,657 | ** 12.15** |
| PiecewiseZones |     3,198 |   1.01 |     1,124 | **  0.35** |    47,031 | ** 14.71** |
| PointmazeLargeNavigate |     3,376 |   1.08 |     1,189 | **  0.35** |   196,982 | ** 58.35** |
| PushT |     7,197 |   1.04 |     2,218 | **  0.31** |   824,267 | **114.53** |
| SimplePointMaze |     1,024 |   1.00 |       277 | **  0.27** |    41,776 | ** 40.80** |
| TwoRoom |     2,969 |   1.01 |     1,184 | **  0.40** |    39,436 | ** 13.28** |
| TwoRoomTarget |     4,263 |   1.02 |       585 | **  0.14** |    23,727 | **  5.57** |

## T5 — per-cell provenance

| config | our CPU 1-thread cell | our CPU 8-way cell | our GPU renderer-only cell | our GPU end-to-end cell |
|---|---|---|---|---|
| Acrobot | step6/lance/renderer_cpu_1thread_r2_node1956_6367705.json (B=1024, node1956.oscar.ccv.brown.edu) | step6/lance/renderer_cpu_8way_node1956_6367705.json (node1956.oscar.ccv.brown.edu, arm=procs8) | step6/lance/renderer_gpu_after_gpu2104_6366975.json (B=4096, gpu2104.oscar.ccv.brown.edu) | step6/lance/pipeline_gpu_r4_gpu2104_6368790.json (B=4096, gpu2104.oscar.ccv.brown.edu) |
| CartPole | step6/lance/renderer_cpu_1thread_r2_node1956_6367705.json (B=1024, node1956.oscar.ccv.brown.edu) | step6/lance/renderer_cpu_8way_node1956_6367705.json (node1956.oscar.ccv.brown.edu, arm=procs8) | step6/lance/renderer_gpu_after_gpu2104_6366975.json (B=4096, gpu2104.oscar.ccv.brown.edu) | step6/lance/pipeline_gpu_r4_gpu2104_6368790.json (B=4096, gpu2104.oscar.ccv.brown.edu) |
| CraftaxClassic | step6/lance/renderer_cpu_1thread_r2_node1956_6367705.json (B=1024, node1956.oscar.ccv.brown.edu) | step6/lance/renderer_cpu_8way_node1956_6367705.json (node1956.oscar.ccv.brown.edu, arm=procs8) | step6/lance/renderer_gpu_after_gpu2104_6366975.json (B=4096, gpu2104.oscar.ccv.brown.edu) | step6/lance/pipeline_gpu_r4_gpu2104_6368790.json (B=4096, gpu2104.oscar.ccv.brown.edu) |
| DMCAcrobotSwingup | step6/lance/renderer_cpu_1thread_r2_node1956_6367705.json (B=1024, node1956.oscar.ccv.brown.edu) | step6/lance/renderer_cpu_8way_node1956_6367705.json (node1956.oscar.ccv.brown.edu, arm=procs8) | step6/lance/renderer_gpu_after_gpu2104_6366975.json (B=4096, gpu2104.oscar.ccv.brown.edu) | step6/lance/pipeline_gpu_r4_gpu2104_6368790.json (B=4096, gpu2104.oscar.ccv.brown.edu) |
| DMCBallInCupCatch | step6/lance/renderer_cpu_1thread_r2_node1956_6367705.json (B=256, node1956.oscar.ccv.brown.edu) | step6/lance/renderer_cpu_8way_node1956_6367705.json (node1956.oscar.ccv.brown.edu, arm=procs8) | step6/lance/renderer_gpu_after_gpu2104_6366975.json (B=4096, gpu2104.oscar.ccv.brown.edu) | step6/lance/pipeline_gpu_r4_gpu2104_6368790.json (B=4096, gpu2104.oscar.ccv.brown.edu) |
| DMCCartpoleBalance | step6/lance/renderer_cpu_1thread_r2_node1956_6367705.json (B=64, node1956.oscar.ccv.brown.edu) | step6/lance/renderer_cpu_8way_node1956_6367705.json (node1956.oscar.ccv.brown.edu, arm=procs8) | step6/lance/renderer_gpu_after_gpu2104_6366975.json (B=1024, gpu2104.oscar.ccv.brown.edu) | step6/lance/pipeline_gpu_r4_gpu2104_6368790.json (B=1024, gpu2104.oscar.ccv.brown.edu) |
| DMCCartpoleSwingup | step6/lance/renderer_cpu_1thread_r2_node1956_6367705.json (B=64, node1956.oscar.ccv.brown.edu) | step6/lance/renderer_cpu_8way_node1956_6367705.json (node1956.oscar.ccv.brown.edu, arm=procs8) | step6/lance/renderer_gpu_after_gpu2104_6366975.json (B=1024, gpu2104.oscar.ccv.brown.edu) | step6/lance/pipeline_gpu_r4_gpu2104_6368790.json (B=1024, gpu2104.oscar.ccv.brown.edu) |
| DMCCheetahRun | step6/lance/renderer_cpu_1thread_r2_node1956_6367705.json (B=64, node1956.oscar.ccv.brown.edu) | step6/lance/renderer_cpu_8way_node1956_6367705.json (node1956.oscar.ccv.brown.edu, arm=procs8) | step6/lance/renderer_gpu_after_gpu2104_6366975.json (B=4096, gpu2104.oscar.ccv.brown.edu) | step6/lance/pipeline_gpu_r4_gpu2104_6368790.json (B=4096, gpu2104.oscar.ccv.brown.edu) |
| DMCFingerTurnHard | step6/lance/renderer_cpu_1thread_r2_node1956_6367705.json (B=16, node1956.oscar.ccv.brown.edu) | step6/lance/renderer_cpu_8way_node1956_6367705.json (node1956.oscar.ccv.brown.edu, arm=procs8) | step6/lance/renderer_gpu_after_gpu2104_6366975.json (B=4096, gpu2104.oscar.ccv.brown.edu) | step6/lance/pipeline_gpu_r4_gpu2104_6368790.json (B=4096, gpu2104.oscar.ccv.brown.edu) |
| DMCHopperHop | step6/lance/renderer_cpu_1thread_r2_node1956_6367705.json (B=1024, node1956.oscar.ccv.brown.edu) | step6/lance/renderer_cpu_8way_node1956_6367705.json (node1956.oscar.ccv.brown.edu, arm=procs8) | step6/lance/renderer_gpu_after_gpu2104_6366975.json (B=4096, gpu2104.oscar.ccv.brown.edu) | step6/lance/pipeline_gpu_r4_gpu2104_6368790.json (B=4096, gpu2104.oscar.ccv.brown.edu) |
| DMCHumanoidStand | step6/lance/renderer_cpu_1thread_r2_node1956_6367705.json (B=256, node1956.oscar.ccv.brown.edu) | step6/lance/renderer_cpu_8way_node1956_6367705.json (node1956.oscar.ccv.brown.edu, arm=procs8) | step6/lance/renderer_gpu_after_gpu2104_6366975.json (B=4096, gpu2104.oscar.ccv.brown.edu) | step6/lance/pipeline_gpu_r4_gpu2104_6368790.json (B=4096, gpu2104.oscar.ccv.brown.edu) |
| DMCPendulumSwingup | step6/lance/renderer_cpu_1thread_r2_node1956_6367705.json (B=256, node1956.oscar.ccv.brown.edu) | step6/lance/renderer_cpu_8way_node1956_6367705.json (node1956.oscar.ccv.brown.edu, arm=procs8) | step6/lance/renderer_gpu_after_gpu2104_6366975.json (B=1024, gpu2104.oscar.ccv.brown.edu) | step6/lance/pipeline_gpu_r4_gpu2104_6368790.json (B=1024, gpu2104.oscar.ccv.brown.edu) |
| DMCReacherHard | step6/lance/renderer_cpu_1thread_r2_node1956_6367705.json (B=16, node1956.oscar.ccv.brown.edu) | step6/lance/renderer_cpu_8way_node1956_6367705.json (node1956.oscar.ccv.brown.edu, arm=procs8) | step6/lance/renderer_gpu_after_gpu2104_6366975.json (B=4096, gpu2104.oscar.ccv.brown.edu) | step6/lance/pipeline_gpu_r4_gpu2104_6368790.json (B=4096, gpu2104.oscar.ccv.brown.edu) |
| DMCWalkerWalk | step6/lance/renderer_cpu_1thread_r2_node1956_6367705.json (B=256, node1956.oscar.ccv.brown.edu) | step6/lance/renderer_cpu_8way_node1956_6367705.json (node1956.oscar.ccv.brown.edu, arm=procs8) | step6/lance/renderer_gpu_after_gpu2104_6366975.json (B=1024, gpu2104.oscar.ccv.brown.edu) | step6/lance/pipeline_gpu_r4_gpu2104_6368790.json (B=1024, gpu2104.oscar.ccv.brown.edu) |
| FetchPickAndPlace | - | step6/lance/renderer_cpu_8way_node1956_6367705.json (node1956.oscar.ccv.brown.edu, arm=None) | step6/lance/renderer_gpu_after_fetch311_gpu2104_6366975.json (B=4096, gpu2104.oscar.ccv.brown.edu) | step6/lance/pipeline_gpu_r4_fetch311_gpu2104_6368790.json (B=4096, gpu2104.oscar.ccv.brown.edu) |
| FetchPickAndPlaceDense | - | step6/lance/renderer_cpu_8way_node1956_6367705.json (node1956.oscar.ccv.brown.edu, arm=None) | step6/lance/renderer_gpu_after_fetch311_gpu2104_6366975.json (B=4096, gpu2104.oscar.ccv.brown.edu) | step6/lance/pipeline_gpu_r4_fetch311_gpu2104_6368790.json (B=4096, gpu2104.oscar.ccv.brown.edu) |
| FetchPickAndPlaceDict | - | step6/lance/renderer_cpu_8way_node1956_6367705.json (node1956.oscar.ccv.brown.edu, arm=None) | step6/lance/renderer_gpu_after_fetch311_gpu2104_6366975.json (B=4096, gpu2104.oscar.ccv.brown.edu) | step6/lance/pipeline_gpu_r4_fetch311_gpu2104_6368790.json (B=4096, gpu2104.oscar.ccv.brown.edu) |
| FetchPush | - | step6/lance/renderer_cpu_8way_node1956_6367705.json (node1956.oscar.ccv.brown.edu, arm=None) | step6/lance/renderer_gpu_after_fetch311_gpu2104_6366975.json (B=1024, gpu2104.oscar.ccv.brown.edu) | step6/lance/pipeline_gpu_r4_fetch311_gpu2104_6368790.json (B=1024, gpu2104.oscar.ccv.brown.edu) |
| FetchPushDense | - | step6/lance/renderer_cpu_8way_node1956_6367705.json (node1956.oscar.ccv.brown.edu, arm=None) | step6/lance/renderer_gpu_after_fetch311_gpu2104_6366975.json (B=4096, gpu2104.oscar.ccv.brown.edu) | step6/lance/pipeline_gpu_r4_fetch311_gpu2104_6368790.json (B=4096, gpu2104.oscar.ccv.brown.edu) |
| FetchPushDict | - | step6/lance/renderer_cpu_8way_node1956_6367705.json (node1956.oscar.ccv.brown.edu, arm=None) | step6/lance/renderer_gpu_after_fetch311_gpu2104_6366975.json (B=1024, gpu2104.oscar.ccv.brown.edu) | step6/lance/pipeline_gpu_r4_fetch311_gpu2104_6368790.json (B=1024, gpu2104.oscar.ccv.brown.edu) |
| FetchReach | - | step6/lance/renderer_cpu_8way_node1956_6367705.json (node1956.oscar.ccv.brown.edu, arm=None) | step6/lance/renderer_gpu_after_fetch311_gpu2104_6366975.json (B=4096, gpu2104.oscar.ccv.brown.edu) | step6/lance/pipeline_gpu_r4_fetch311_gpu2104_6368790.json (B=4096, gpu2104.oscar.ccv.brown.edu) |
| FetchReachDense | - | step6/lance/renderer_cpu_8way_node1956_6367705.json (node1956.oscar.ccv.brown.edu, arm=None) | step6/lance/renderer_gpu_after_fetch311_gpu2104_6366975.json (B=4096, gpu2104.oscar.ccv.brown.edu) | step6/lance/pipeline_gpu_r4_fetch311_gpu2104_6368790.json (B=4096, gpu2104.oscar.ccv.brown.edu) |
| FetchReachDict | - | step6/lance/renderer_cpu_8way_node1956_6367705.json (node1956.oscar.ccv.brown.edu, arm=None) | step6/lance/renderer_gpu_after_fetch311_gpu2104_6366975.json (B=1024, gpu2104.oscar.ccv.brown.edu) | step6/lance/pipeline_gpu_r4_fetch311_gpu2104_6368790.json (B=1024, gpu2104.oscar.ccv.brown.edu) |
| FetchSlide | - | step6/lance/renderer_cpu_8way_node1956_6367705.json (node1956.oscar.ccv.brown.edu, arm=None) | step6/lance/renderer_gpu_after_fetch311_gpu2104_6366975.json (B=1024, gpu2104.oscar.ccv.brown.edu) | step6/lance/pipeline_gpu_r4_fetch311_gpu2104_6368790.json (B=1024, gpu2104.oscar.ccv.brown.edu) |
| FetchSlideDense | - | step6/lance/renderer_cpu_8way_node1956_6367705.json (node1956.oscar.ccv.brown.edu, arm=None) | step6/lance/renderer_gpu_after_fetch311_gpu2104_6366975.json (B=4096, gpu2104.oscar.ccv.brown.edu) | step6/lance/pipeline_gpu_r4_fetch311_gpu2104_6368790.json (B=4096, gpu2104.oscar.ccv.brown.edu) |
| FetchSlideDict | - | step6/lance/renderer_cpu_8way_node1956_6367705.json (node1956.oscar.ccv.brown.edu, arm=None) | step6/lance/renderer_gpu_after_fetch311_gpu2104_6366975.json (B=4096, gpu2104.oscar.ccv.brown.edu) | step6/lance/pipeline_gpu_r4_fetch311_gpu2104_6368790.json (B=4096, gpu2104.oscar.ccv.brown.edu) |
| MountainCar | step6/lance/renderer_cpu_1thread_r2_node1956_6367705.json (B=256, node1956.oscar.ccv.brown.edu) | step6/lance/renderer_cpu_8way_node1956_6367705.json (node1956.oscar.ccv.brown.edu, arm=procs8) | step6/lance/renderer_gpu_after_gpu2104_6366975.json (B=4096, gpu2104.oscar.ccv.brown.edu) | step6/lance/pipeline_gpu_r4_gpu2104_6368790.json (B=4096, gpu2104.oscar.ccv.brown.edu) |
| MountainCarContinuous | step6/lance/renderer_cpu_1thread_r2_node1956_6367705.json (B=256, node1956.oscar.ccv.brown.edu) | step6/lance/renderer_cpu_8way_node1956_6367705.json (node1956.oscar.ccv.brown.edu, arm=procs8) | step6/lance/renderer_gpu_after_gpu2104_6366975.json (B=4096, gpu2104.oscar.ccv.brown.edu) | step6/lance/pipeline_gpu_r4_gpu2104_6368790.json (B=4096, gpu2104.oscar.ccv.brown.edu) |
| Pendulum | step6/lance/renderer_cpu_1thread_r2_node1956_6367705.json (B=256, node1956.oscar.ccv.brown.edu) | step6/lance/renderer_cpu_8way_node1956_6367705.json (node1956.oscar.ccv.brown.edu, arm=procs8) | step6/lance/renderer_gpu_after_gpu2104_6366975.json (B=4096, gpu2104.oscar.ccv.brown.edu) | step6/lance/pipeline_gpu_r4_gpu2104_6368790.json (B=4096, gpu2104.oscar.ccv.brown.edu) |
| Piecewise | step6/lance/renderer_cpu_1thread_r2_node1956_6367705.json (B=16, node1956.oscar.ccv.brown.edu) | step6/lance/renderer_cpu_8way_node1956_6367705.json (node1956.oscar.ccv.brown.edu, arm=procs8) | step6/lance/renderer_gpu_after_gpu2104_6366975.json (B=1024, gpu2104.oscar.ccv.brown.edu) | step6/lance/pipeline_gpu_r4_gpu2104_6368790.json (B=1024, gpu2104.oscar.ccv.brown.edu) |
| PiecewiseZones | step6/lance/renderer_cpu_1thread_r2_node1956_6367705.json (B=16, node1956.oscar.ccv.brown.edu) | step6/lance/renderer_cpu_8way_node1956_6367705.json (node1956.oscar.ccv.brown.edu, arm=procs8) | step6/lance/renderer_gpu_after_gpu2104_6366975.json (B=1024, gpu2104.oscar.ccv.brown.edu) | step6/lance/pipeline_gpu_r4_gpu2104_6368790.json (B=1024, gpu2104.oscar.ccv.brown.edu) |
| PointmazeLargeNavigate | step6/lance/renderer_cpu_1thread_r2_node1956_6367705.json (B=1024, node1956.oscar.ccv.brown.edu) | step6/lance/renderer_cpu_8way_node1956_6367705.json (node1956.oscar.ccv.brown.edu, arm=procs8) | step6/lance/renderer_gpu_after_gpu2104_6366975.json (B=4096, gpu2104.oscar.ccv.brown.edu) | step6/lance/pipeline_gpu_r4_gpu2104_6368790.json (B=4096, gpu2104.oscar.ccv.brown.edu) |
| PushT | step6/lance/renderer_cpu_1thread_r2_node1956_6367705.json (B=1024, node1956.oscar.ccv.brown.edu) | step6/lance/renderer_cpu_8way_node1956_6367705.json (node1956.oscar.ccv.brown.edu, arm=procs8) | step6/lance/renderer_gpu_after_gpu2104_6366975.json (B=4096, gpu2104.oscar.ccv.brown.edu) | step6/lance/pipeline_gpu_r4_gpu2104_6368790.json (B=4096, gpu2104.oscar.ccv.brown.edu) |
| SimplePointMaze | step6/lance/renderer_cpu_1thread_r2_node1956_6367705.json (B=64, node1956.oscar.ccv.brown.edu) | step6/lance/renderer_cpu_8way_node1956_6367705.json (node1956.oscar.ccv.brown.edu, arm=procs8) | step6/lance/renderer_gpu_after_gpu2104_6366975.json (B=4096, gpu2104.oscar.ccv.brown.edu) | step6/lance/pipeline_gpu_r4_gpu2104_6368790.json (B=4096, gpu2104.oscar.ccv.brown.edu) |
| TwoRoom | step6/lance/renderer_cpu_1thread_r2_node1956_6367705.json (B=16, node1956.oscar.ccv.brown.edu) | step6/lance/renderer_cpu_8way_node1956_6367705.json (node1956.oscar.ccv.brown.edu, arm=procs8) | step6/lance/renderer_gpu_after_gpu2104_6366975.json (B=4096, gpu2104.oscar.ccv.brown.edu) | step6/lance/pipeline_gpu_r4_gpu2104_6368790.json (B=4096, gpu2104.oscar.ccv.brown.edu) |
| TwoRoomTarget | step6/lance/renderer_cpu_1thread_r2_node1956_6367705.json (B=256, node1956.oscar.ccv.brown.edu) | step6/lance/renderer_cpu_8way_node1956_6367705.json (node1956.oscar.ccv.brown.edu, arm=procs8) | step6/lance/renderer_gpu_after_gpu2104_6366975.json (B=4096, gpu2104.oscar.ccv.brown.edu) | step6/lance/pipeline_gpu_r4_gpu2104_6368790.json (B=4096, gpu2104.oscar.ccv.brown.edu) |

## T6 — aggregates over the 36 rows

| quantity | median | min | max | n |
|---|---|---|---|---|
| lance 1 worker (tmpfs, same job as 8-way renderer) | 1,686.79 | 682.58 | 12,168.93 | 36 |
| lance 8 workers (tmpfs, same job as 8-way renderer) | 13,176.76 | 5,039.24 | 92,201.23 | 36 |
| lance 8 workers: cores actually used | 7.86 | 7.41 | 7.94 | 36 |
| lance 1 proc best-of-8 (tmpfs) | 2,285.32 | 1,092.90 | 13,042.00 | 36 |
| lance 8 workers (NFS) | 14,151.52 | 5,328.22 | 101,458.98 | 36 |
| lance 8 workers (GPFS) | 14,310.42 | 5,232.30 | 104,287.33 | 36 |
| JPEG→device, 8w loader + H2D | 8,100.24 | 3,578.04 | 54,839.49 | 36 |
| JPEG→device, nvJPEG | 5,829.54 | 2,652.64 | 14,919.75 | 36 |
| ours CPU 1 thread | 498.44 | 53.23 | 103,130.25 | 24 |
| ours CPU best 8-way | 2,611.18 | 386.56 | 203,349.00 | 24 |
| ours CPU 8-way scaling efficiency | 0.91 | 0.30 | 1.18 | 24 |
| ours GPU renderer-only | 40,605.76 | 5,124.32 | 1,345,537.02 | 36 |
| ours GPU end-to-end | 40,032.01 | 5,125.90 | 840,566.98 | 36 |
| cpu_pipeline_ratio | 0.11 | 0.01 | 6.96 | 24 |
| gpu_pipeline_ratio | 1.44 | 0.18 | 46.21 | 36 |
| gpu_pipeline_ratio (nvJPEG) | 3.42 | 0.72 | 124.30 | 36 |
| cores_cpu (frozen, one core q90) | 0.20 | 0.01 | 18.89 | 24 |
| cores (frozen, GPU vs one core q90) | 6.44 | 1.26 | 246.46 | 36 |

## T7 — who wins against the production pipeline


### CPU: our 8-way renderer beats the 8-worker Lance pipeline

**1/36 win:** CraftaxClassic

**lose:** Acrobot, CartPole, DMCAcrobotSwingup, DMCBallInCupCatch, DMCCartpoleBalance, DMCCartpoleSwingup, DMCCheetahRun, DMCFingerTurnHard, DMCHopperHop, DMCHumanoidStand, DMCPendulumSwingup, DMCReacherHard, DMCWalkerWalk, FetchPickAndPlace, FetchPickAndPlaceDense, FetchPickAndPlaceDict, FetchPush, FetchPushDense, FetchPushDict, FetchReach, FetchReachDense, FetchReachDict, FetchSlide, FetchSlideDense, FetchSlideDict, MountainCar, MountainCarContinuous, Pendulum, Piecewise, PiecewiseZones, PointmazeLargeNavigate, PushT, SimplePointMaze, TwoRoom, TwoRoomTarget


### CPU: our 1 thread beats ONE CORE's share of the 8-worker pipeline

**1/36 win:** CraftaxClassic

**lose:** Acrobot, CartPole, DMCAcrobotSwingup, DMCBallInCupCatch, DMCCartpoleBalance, DMCCartpoleSwingup, DMCCheetahRun, DMCFingerTurnHard, DMCHopperHop, DMCHumanoidStand, DMCPendulumSwingup, DMCReacherHard, DMCWalkerWalk, FetchPickAndPlace, FetchPickAndPlaceDense, FetchPickAndPlaceDict, FetchPush, FetchPushDense, FetchPushDict, FetchReach, FetchReachDense, FetchReachDict, FetchSlide, FetchSlideDense, FetchSlideDict, MountainCar, MountainCarContinuous, Pendulum, Piecewise, PiecewiseZones, PointmazeLargeNavigate, PushT, SimplePointMaze, TwoRoom, TwoRoomTarget


### CPU: our 1 thread beats ONE Lance worker

**1/36 win:** CraftaxClassic

**lose:** Acrobot, CartPole, DMCAcrobotSwingup, DMCBallInCupCatch, DMCCartpoleBalance, DMCCartpoleSwingup, DMCCheetahRun, DMCFingerTurnHard, DMCHopperHop, DMCHumanoidStand, DMCPendulumSwingup, DMCReacherHard, DMCWalkerWalk, FetchPickAndPlace, FetchPickAndPlaceDense, FetchPickAndPlaceDict, FetchPush, FetchPushDense, FetchPushDict, FetchReach, FetchReachDense, FetchReachDict, FetchSlide, FetchSlideDense, FetchSlideDict, MountainCar, MountainCarContinuous, Pendulum, Piecewise, PiecewiseZones, PointmazeLargeNavigate, PushT, SimplePointMaze, TwoRoom, TwoRoomTarget


### GPU: our end-to-end state→device beats 8-worker JPEG→device

**30/36 win:** Acrobot, CartPole, CraftaxClassic, DMCCartpoleBalance, DMCCheetahRun, DMCHopperHop, DMCHumanoidStand, DMCWalkerWalk, FetchPickAndPlace, FetchPickAndPlaceDense, FetchPickAndPlaceDict, FetchPush, FetchPushDense, FetchPushDict, FetchReach, FetchReachDense, FetchReachDict, FetchSlide, FetchSlideDense, FetchSlideDict, MountainCar, MountainCarContinuous, Pendulum, Piecewise, PiecewiseZones, PointmazeLargeNavigate, PushT, SimplePointMaze, TwoRoom, TwoRoomTarget

**lose:** DMCAcrobotSwingup, DMCBallInCupCatch, DMCCartpoleSwingup, DMCFingerTurnHard, DMCPendulumSwingup, DMCReacherHard


### GPU: our end-to-end beats nvJPEG→device

**35/36 win:** Acrobot, CartPole, CraftaxClassic, DMCAcrobotSwingup, DMCCartpoleBalance, DMCCartpoleSwingup, DMCCheetahRun, DMCFingerTurnHard, DMCHopperHop, DMCHumanoidStand, DMCPendulumSwingup, DMCReacherHard, DMCWalkerWalk, FetchPickAndPlace, FetchPickAndPlaceDense, FetchPickAndPlaceDict, FetchPush, FetchPushDense, FetchPushDict, FetchReach, FetchReachDense, FetchReachDict, FetchSlide, FetchSlideDense, FetchSlideDict, MountainCar, MountainCarContinuous, Pendulum, Piecewise, PiecewiseZones, PointmazeLargeNavigate, PushT, SimplePointMaze, TwoRoom, TwoRoomTarget

**lose:** DMCBallInCupCatch


### GPU (renderer-only vs host-memory Lance — the ASYMMETRIC comparison, shown for continuity)

**19/36 win:** Acrobot, CartPole, CraftaxClassic, DMCCheetahRun, DMCHopperHop, DMCHumanoidStand, DMCWalkerWalk, FetchSlide, FetchSlideDense, FetchSlideDict, MountainCar, MountainCarContinuous, Pendulum, Piecewise, PiecewiseZones, PointmazeLargeNavigate, PushT, SimplePointMaze, TwoRoom

**lose:** DMCAcrobotSwingup, DMCBallInCupCatch, DMCCartpoleBalance, DMCCartpoleSwingup, DMCFingerTurnHard, DMCPendulumSwingup, DMCReacherHard, FetchPickAndPlace, FetchPickAndPlaceDense, FetchPickAndPlaceDict, FetchPush, FetchPushDense, FetchPushDict, FetchReach, FetchReachDense, FetchReachDict, TwoRoomTarget


### FROZEN column: cores_cpu ≥ 1 (the Step-2 G4-on-CPU bar)

**1/36 win:** CraftaxClassic

**lose:** Acrobot, CartPole, DMCAcrobotSwingup, DMCBallInCupCatch, DMCCartpoleBalance, DMCCartpoleSwingup, DMCCheetahRun, DMCFingerTurnHard, DMCHopperHop, DMCHumanoidStand, DMCPendulumSwingup, DMCReacherHard, DMCWalkerWalk, FetchPickAndPlace, FetchPickAndPlaceDense, FetchPickAndPlaceDict, FetchPush, FetchPushDense, FetchPushDict, FetchReach, FetchReachDense, FetchReachDict, FetchSlide, FetchSlideDense, FetchSlideDict, MountainCar, MountainCarContinuous, Pendulum, Piecewise, PiecewiseZones, PointmazeLargeNavigate, PushT, SimplePointMaze, TwoRoom, TwoRoomTarget

