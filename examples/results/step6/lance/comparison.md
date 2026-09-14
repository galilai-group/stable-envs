# Our renderer vs the 8-worker JPEG-in-Lance baseline (Oscar)

All Lance columns are this repo's `examples/bench_lance.py`, unmodified, on Oscar.
`local_n` is 8 spawn workers with 1 torch thread each, aggregated from the workers' own
elapsed times. `/8` is that divided by eight, i.e. one core's share.

| config | frame | kB/f | lance local_1 | lance local_n(8) | lance/core | our GPU | GPU/local_n | GPU cores-of-8 | our CPU 1thr | CPU1/local_1 | CPU1/core | our CPU 8thr | CPU8/local_n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Acrobot | 500x500 | 10.6 |     1,439 |     7,639 |       955 |   158,648 |   20.77 |  166.14 |       382 |    0.27 |    0.40 |         - |       - |
| CartPole | 400x600 | 8.1 |     1,522 |     7,624 |       953 |   176,593 |   23.16 |  185.30 |       547 |    0.36 |    0.57 |         - |       - |
| CraftaxClassic | 112x144 | 10.8 |     5,089 |    38,476 |     4,810 | 1,229,787 |   31.96 |  255.70 |   154,086 |   30.28 |   32.04 |         - |       - |
| DMCAcrobotSwingup | 84x84 | 3.1 |    11,532 |    89,050 |    11,131 |    27,808 |    0.31 |    2.50 |       322 |    0.03 |    0.03 |         - |       - |
| DMCBallInCupCatch | 84x84 | 3.0 |    11,332 |    92,184 |    11,523 |    10,112 |    0.11 |    0.88 |        66 |    0.01 |    0.01 |         - |       - |
| DMCCartpoleBalance | 84x84 | 2.9 |    11,662 |    92,829 |    11,604 |    43,642 |    0.47 |    3.76 |       370 |    0.03 |    0.03 |         - |       - |
| DMCCartpoleSwingup | 84x84 | 3.0 |    11,988 |    95,567 |    11,946 |    42,389 |    0.44 |    3.55 |       647 |    0.05 |    0.05 |         - |       - |
| DMCCheetahRun | 84x84 | 4.6 |     9,602 |    77,236 |     9,655 |    76,589 |    0.99 |    7.93 |       116 |    0.01 |    0.01 |         - |       - |
| DMCFingerTurnHard | 84x84 | 3.7 |    10,439 |    84,289 |    10,536 |    22,042 |    0.26 |    2.09 |        57 |    0.01 |    0.01 |         - |       - |
| DMCHopperHop | 84x84 | 4.4 |     9,840 |    77,942 |     9,743 |    76,735 |    0.98 |    7.88 |       188 |    0.02 |    0.02 |         - |       - |
| DMCHumanoidStand | 84x84 | 3.8 |    10,565 |    85,514 |    10,689 |    64,493 |    0.75 |    6.03 |        46 |    0.00 |    0.00 |         - |       - |
| DMCPendulumSwingup | 84x84 | 2.5 |    12,860 |   100,506 |    12,563 |    36,127 |    0.36 |    2.88 |       239 |    0.02 |    0.02 |         - |       - |
| DMCReacherHard | 84x84 | 2.4 |    13,042 |   103,579 |    12,947 |    25,207 |    0.24 |    1.95 |       139 |    0.01 |    0.01 |         - |       - |
| DMCWalkerWalk | 84x84 | 4.3 |     9,971 |    79,561 |     9,945 |    73,503 |    0.92 |    7.39 |        63 |    0.01 |    0.01 |         - |       - |
| FetchPickAndPlace | 480x480 | 25.9 |     1,101 |     5,807 |       726 |     5,171 |    0.89 |    7.12 |         - |       - |       - |         - |       - |
| FetchPickAndPlaceDense | 480x480 | 25.9 |     1,106 |     5,552 |       694 |     5,162 |    0.93 |    7.44 |         - |       - |       - |         - |       - |
| FetchPickAndPlaceDict | 480x480 | 25.9 |     1,104 |     6,482 |       810 |     5,163 |    0.80 |    6.37 |         - |       - |       - |         - |       - |
| FetchPush | 480x480 | 25.9 |     1,109 |     6,421 |       803 |     5,224 |    0.81 |    6.51 |         - |       - |       - |         - |       - |
| FetchPushDense | 480x480 | 25.9 |     1,105 |     6,390 |       799 |     5,221 |    0.82 |    6.54 |         - |       - |       - |         - |       - |
| FetchPushDict | 480x480 | 25.9 |     1,109 |     6,450 |       806 |     5,192 |    0.80 |    6.44 |         - |       - |       - |         - |       - |
| FetchReach | 480x480 | 25.5 |     1,109 |     6,080 |       760 |     5,212 |    0.86 |    6.86 |         - |       - |       - |         - |       - |
| FetchReachDense | 480x480 | 25.5 |     1,111 |     5,721 |       715 |     5,185 |    0.91 |    7.25 |         - |       - |       - |         - |       - |
| FetchReachDict | 480x480 | 25.5 |     1,111 |     5,746 |       718 |     5,184 |    0.90 |    7.22 |         - |       - |       - |         - |       - |
| FetchSlide | 480x480 | 26.7 |     1,093 |     6,346 |       793 |     5,295 |    0.83 |    6.68 |         - |       - |       - |         - |       - |
| FetchSlideDense | 480x480 | 26.7 |     1,098 |     6,311 |       789 |     5,267 |    0.83 |    6.68 |         - |       - |       - |         - |       - |
| FetchSlideDict | 480x480 | 26.7 |     1,097 |     6,224 |       778 |     5,264 |    0.85 |    6.77 |         - |       - |       - |         - |       - |
| MountainCar | 400x600 | 13.1 |     1,416 |     7,599 |       950 |   148,170 |   19.50 |  155.99 |       466 |    0.33 |    0.49 |         - |       - |
| MountainCarContinuous | 400x600 | 13.1 |     1,410 |     7,484 |       935 |   147,355 |   19.69 |  157.52 |       474 |    0.34 |    0.51 |         - |       - |
| Pendulum | 500x500 | 8.6 |     1,455 |     7,742 |       968 |   176,382 |   22.78 |  182.25 |       504 |    0.35 |    0.52 |         - |       - |
| Piecewise | 224x224 | 3.2 |     4,872 |    28,868 |     3,609 |    54,671 |    1.89 |   15.15 |     1,243 |    0.26 |    0.34 |         - |       - |
| PiecewiseZones | 224x224 | 4.9 |     4,049 |    26,465 |     3,308 |    46,922 |    1.77 |   14.18 |     1,205 |    0.30 |    0.36 |         - |       - |
| PointmazeLargeNavigate | 224x224 | 12.3 |     3,048 |    20,163 |     2,520 |    95,962 |    4.76 |   38.07 |     1,460 |    0.48 |    0.58 |         - |       - |
| PushT | 96x96 | 2.7 |     9,958 |    79,220 |     9,902 |   816,785 |   10.31 |   82.48 |     1,981 |    0.20 |    0.20 |         - |       - |
| SimplePointMaze | 500x500 | 12.1 |     1,407 |     7,640 |       955 |    42,305 |    5.54 |   44.30 |       242 |    0.17 |    0.25 |         - |       - |
| TwoRoom | 224x224 | 4.4 |     4,619 |    27,278 |     3,410 |    39,518 |    1.45 |   11.59 |     1,359 |    0.29 |    0.40 |         - |       - |
| TwoRoomTarget | 224x224 | 4.8 |     4,185 |    26,941 |     3,368 |    23,731 |    0.88 |    7.05 |       638 |    0.15 |    0.19 |         - |       - |

## Storage arms -- does the filesystem matter?

| config | local (tmpfs) 1-proc | shared (NFS) 1-proc | jobtmp (GPFS) 1-proc | local 8w | shared 8w | jobtmp 8w | read_frac local | read_frac NFS | read_frac GPFS |
|---|---|---|---|---|---|---|---|---|---|
| Acrobot |     1,439 |     1,418 |     1,420 |     7,639 |     7,594 |     7,678 | 2.14% | 2.53% | 2.21% |
| CartPole |     1,522 |     1,514 |     1,523 |     7,624 |     7,688 |     8,191 | 1.88% | 2.38% | 2.01% |
| CraftaxClassic |     5,089 |     4,785 |     4,675 |    38,476 |    38,207 |    38,148 | 7.18% | 9.04% | 7.63% |
| DMCAcrobotSwingup |    11,532 |     9,444 |    11,184 |    89,050 |    92,140 |    93,271 | 11.46% | 12.54% | 13.55% |
| DMCBallInCupCatch |    11,332 |     9,275 |    11,366 |    92,184 |    92,102 |    93,707 | 12.41% | 13.84% | 11.74% |
| DMCCartpoleBalance |    11,662 |    11,091 |    11,692 |    92,829 |    92,181 |    94,379 | 9.55% | 15.70% | 10.72% |
| DMCCartpoleSwingup |    11,988 |     9,488 |    11,806 |    95,567 |    93,251 |    95,366 | 9.95% | 13.74% | 10.51% |
| DMCCheetahRun |     9,602 |     9,316 |     9,579 |    77,236 |    72,109 |    78,140 | 9.69% | 14.20% | 12.22% |
| DMCFingerTurnHard |    10,439 |     8,435 |    10,357 |    84,289 |    78,924 |    84,428 | 9.74% | 13.02% | 11.12% |
| DMCHopperHop |     9,840 |     7,940 |     8,145 |    77,942 |    78,319 |    77,891 | 9.87% | 12.56% | 10.34% |
| DMCHumanoidStand |    10,565 |     8,542 |    10,806 |    85,514 |    86,017 |    84,879 | 10.48% | 13.59% | 10.92% |
| DMCPendulumSwingup |    12,860 |    10,248 |    12,340 |   100,506 |   101,459 |   102,982 | 11.40% | 15.25% | 15.09% |
| DMCReacherHard |    13,042 |    10,224 |    12,721 |   103,579 |   100,426 |   104,287 | 9.89% | 15.18% | 10.83% |
| DMCWalkerWalk |     9,971 |     9,609 |     9,891 |    79,561 |    78,154 |    79,427 | 9.75% | 14.64% | 10.03% |
| FetchPickAndPlace |     1,101 |     1,097 |     1,109 |     5,807 |     6,150 |     5,232 | 3.01% | 3.54% | 3.00% |
| FetchPickAndPlaceDense |     1,106 |     1,099 |     1,106 |     5,552 |     5,701 |     5,755 | 3.03% | 3.39% | 3.02% |
| FetchPickAndPlaceDict |     1,104 |     1,103 |     1,104 |     6,482 |     6,038 |     6,133 | 3.10% | 3.43% | 2.99% |
| FetchPush |     1,109 |     1,106 |     1,112 |     6,421 |     6,410 |     6,486 | 2.89% | 3.36% | 3.02% |
| FetchPushDense |     1,105 |     1,099 |     1,102 |     6,390 |     6,526 |     6,504 | 3.06% | 3.39% | 3.07% |
| FetchPushDict |     1,109 |     1,100 |     1,105 |     6,450 |     6,453 |     6,501 | 3.03% | 3.39% | 3.18% |
| FetchReach |     1,109 |     1,108 |     1,110 |     6,080 |     5,328 |     6,311 | 2.96% | 3.49% | 3.14% |
| FetchReachDense |     1,111 |     1,110 |     1,109 |     5,721 |     6,123 |     6,716 | 3.07% | 3.25% | 3.13% |
| FetchReachDict |     1,111 |     1,105 |     1,112 |     5,746 |     6,671 |     6,197 | 3.05% | 3.46% | 3.03% |
| FetchSlide |     1,093 |     1,092 |     1,100 |     6,346 |     6,391 |     6,410 | 3.21% | 3.52% | 3.17% |
| FetchSlideDense |     1,098 |     1,095 |     1,095 |     6,311 |     6,330 |     6,206 | 3.12% | 3.49% | 3.22% |
| FetchSlideDict |     1,097 |     1,093 |     1,104 |     6,224 |     6,437 |     6,399 | 3.22% | 3.36% | 3.15% |
| MountainCar |     1,416 |     1,398 |     1,408 |     7,599 |     7,551 |     7,557 | 2.31% | 2.76% | 2.39% |
| MountainCarContinuous |     1,410 |     1,401 |     1,410 |     7,484 |     7,451 |     7,048 | 2.34% | 2.87% | 2.41% |
| Pendulum |     1,455 |     1,466 |     1,455 |     7,742 |     7,802 |     7,765 | 1.95% | 2.15% | 1.95% |
| Piecewise |     4,872 |     4,518 |     4,574 |    28,868 |    28,871 |    27,987 | 4.38% | 5.61% | 4.09% |
| PiecewiseZones |     4,049 |     3,593 |     4,131 |    26,465 |    24,183 |    26,036 | 4.60% | 5.27% | 4.25% |
| PointmazeLargeNavigate |     3,048 |     3,038 |     3,040 |    20,163 |    20,501 |    20,430 | 5.40% | 5.93% | 5.60% |
| PushT |     9,958 |     8,303 |     9,904 |    79,220 |    78,241 |    78,036 | 10.89% | 12.48% | 9.43% |
| SimplePointMaze |     1,407 |     1,431 |     1,409 |     7,640 |     7,686 |     7,403 | 2.29% | 2.54% | 2.30% |
| TwoRoom |     4,619 |     4,260 |     4,337 |    27,278 |    27,483 |    27,332 | 4.54% | 5.58% | 4.86% |
| TwoRoomTarget |     4,185 |     3,639 |     4,179 |    26,941 |    26,888 |    26,632 | 4.61% | 5.01% | 4.85% |

## Worker scaling on the local arm

| config | 1 worker | 4 workers | 8 workers | 8w / 1w | 1-process best-of-8 |
|---|---|---|---|---|---|
| Acrobot |       989 |     3,909 |     7,639 |    7.73 |     1,439 |
| CartPole |     1,015 |     4,160 |     7,624 |    7.51 |     1,522 |
| CraftaxClassic |     4,891 |    19,215 |    38,476 |    7.87 |     5,089 |
| DMCAcrobotSwingup |    11,938 |    46,323 |    89,050 |    7.46 |    11,532 |
| DMCBallInCupCatch |    11,617 |    43,809 |    92,184 |    7.94 |    11,332 |
| DMCCartpoleBalance |    11,899 |    47,348 |    92,829 |    7.80 |    11,662 |
| DMCCartpoleSwingup |    12,159 |    47,444 |    95,567 |    7.86 |    11,988 |
| DMCCheetahRun |     9,974 |    37,702 |    77,236 |    7.74 |     9,602 |
| DMCFingerTurnHard |    10,533 |    41,279 |    84,289 |    8.00 |    10,439 |
| DMCHopperHop |    10,136 |    37,786 |    77,942 |    7.69 |     9,840 |
| DMCHumanoidStand |    10,904 |    41,942 |    85,514 |    7.84 |    10,565 |
| DMCPendulumSwingup |    13,068 |    49,878 |   100,506 |    7.69 |    12,860 |
| DMCReacherHard |    13,363 |    51,534 |   103,579 |    7.75 |    13,042 |
| DMCWalkerWalk |    10,255 |    39,780 |    79,561 |    7.76 |     9,971 |
| FetchPickAndPlace |       848 |     3,215 |     5,807 |    6.85 |     1,101 |
| FetchPickAndPlaceDense |       836 |     3,036 |     5,552 |    6.64 |     1,106 |
| FetchPickAndPlaceDict |       772 |     3,324 |     6,482 |    8.40 |     1,104 |
| FetchPush |       828 |     3,213 |     6,421 |    7.76 |     1,109 |
| FetchPushDense |       854 |     3,258 |     6,390 |    7.48 |     1,105 |
| FetchPushDict |       824 |     3,279 |     6,450 |    7.82 |     1,109 |
| FetchReach |       855 |     3,042 |     6,080 |    7.11 |     1,109 |
| FetchReachDense |       855 |     3,157 |     5,721 |    6.69 |     1,111 |
| FetchReachDict |       780 |     3,382 |     5,746 |    7.37 |     1,111 |
| FetchSlide |       844 |     3,302 |     6,346 |    7.52 |     1,093 |
| FetchSlideDense |       849 |     3,203 |     6,311 |    7.43 |     1,098 |
| FetchSlideDict |       851 |     3,277 |     6,224 |    7.31 |     1,097 |
| MountainCar |     1,005 |     3,831 |     7,599 |    7.56 |     1,416 |
| MountainCarContinuous |     1,008 |     3,843 |     7,484 |    7.43 |     1,410 |
| Pendulum |     1,008 |     4,003 |     7,742 |    7.68 |     1,455 |
| Piecewise |     3,686 |    14,521 |    28,868 |    7.83 |     4,872 |
| PiecewiseZones |     3,426 |    13,331 |    26,465 |    7.73 |     4,049 |
| PointmazeLargeNavigate |     2,638 |    10,327 |    20,163 |    7.64 |     3,048 |
| PushT |    10,256 |    39,278 |    79,220 |    7.72 |     9,958 |
| SimplePointMaze |       985 |     3,882 |     7,640 |    7.75 |     1,407 |
| TwoRoom |     3,518 |    13,812 |    27,278 |    7.75 |     4,619 |
| TwoRoomTarget |     3,463 |    13,299 |    26,941 |    7.78 |     4,185 |

## Oscar vs the committed Randall-cluster run -- how much is hardware, how much is size

The committed `examples/results/storage/lance.json` asked for 256 px frames; the Step-6 rows
are certified at each env's own size. Where an env refused 256 and fell back to native, the
two runs describe the SAME picture and the ratio is pure hardware (Xeon 8558/H200 vs Xeon
8268/3090). Where 256 was accepted, the pictures differ and the ratio is size + hardware.

| config | Oscar frame | Randall frame | same picture? | Oscar kB/f | Randall kB/f | Oscar local_1 | Randall local_1 | Oscar local_n | Randall local_n | Oscar/Randall local_n |
|---|---|---|---|---|---|---|---|---|---|---|
| Acrobot | 500x500 | 500x500 | yes | 10.6 | 10.6 |     1,439 |     1,133 |     7,639 |     7,004 |    1.09 |
| CartPole | 400x600 | 400x600 | yes | 8.1 | 8.1 |     1,522 |     1,239 |     7,624 |     7,287 |    1.05 |
| CraftaxClassic | 112x144 | 112x144 | yes | 10.8 | 10.8 |     5,089 |     5,508 |    38,476 |    40,562 |    0.95 |
| DMCAcrobotSwingup | 84x84 | 256x256 | NO | 3.1 | 23.3 |    11,532 |     2,388 |    89,050 |    14,474 |    6.15 |
| DMCBallInCupCatch | 84x84 | 256x256 | NO | 3.0 | 21.7 |    11,332 |     2,487 |    92,184 |    15,024 |    6.14 |
| DMCCartpoleBalance | 84x84 | 256x256 | NO | 2.9 | 22.2 |    11,662 |     2,402 |    92,829 |    13,489 |    6.88 |
| DMCCartpoleSwingup | 84x84 | 256x256 | NO | 3.0 | 22.4 |    11,988 |     2,452 |    95,567 |    14,542 |    6.57 |
| DMCCheetahRun | 84x84 | 256x256 | NO | 4.6 | 27.2 |     9,602 |     2,193 |    77,236 |    13,289 |    5.81 |
| DMCFingerTurnHard | 84x84 | 256x256 | NO | 3.7 | 23.3 |    10,439 |     2,303 |    84,289 |    14,471 |    5.82 |
| DMCHopperHop | 84x84 | 256x256 | NO | 4.4 | 26.4 |     9,840 |     1,869 |    77,942 |    13,689 |    5.69 |
| DMCHumanoidStand | 84x84 | 256x256 | NO | 3.8 | 16.6 |    10,565 |     2,482 |    85,514 |    16,437 |    5.20 |
| DMCPendulumSwingup | 84x84 | 256x256 | NO | 2.5 | 11.5 |    12,860 |     2,818 |   100,506 |    18,941 |    5.31 |
| DMCReacherHard | 84x84 | 256x256 | NO | 2.4 | 10.3 |    13,042 |     2,873 |   103,579 |    19,070 |    5.43 |
| DMCWalkerWalk | 84x84 | 256x256 | NO | 4.3 | 24.0 |     9,971 |     2,042 |    79,561 |    14,384 |    5.53 |
| FetchPickAndPlace | 480x480 | 256x256 | NO | 25.9 | 11.8 |     1,101 |     2,913 |     5,807 |    19,424 |    0.30 |
| FetchPickAndPlaceDense | 480x480 | 256x256 | NO | 25.9 | 11.8 |     1,106 |     2,905 |     5,552 |    19,360 |    0.29 |
| FetchPickAndPlaceDict | 480x480 | 256x256 | NO | 25.9 | 11.8 |     1,104 |     2,907 |     6,482 |    19,076 |    0.34 |
| FetchPush | 480x480 | 256x256 | NO | 25.9 | 11.8 |     1,109 |     2,930 |     6,421 |    19,623 |    0.33 |
| FetchPushDense | 480x480 | 256x256 | NO | 25.9 | 11.8 |     1,105 |     2,900 |     6,390 |    18,660 |    0.34 |
| FetchPushDict | 480x480 | 256x256 | NO | 25.9 | 11.8 |     1,109 |     2,931 |     6,450 |    19,031 |    0.34 |
| FetchReach | 480x480 | 256x256 | NO | 25.5 | 11.7 |     1,109 |     2,936 |     6,080 |    19,378 |    0.31 |
| FetchReachDense | 480x480 | 256x256 | NO | 25.5 | 11.7 |     1,111 |     2,931 |     5,721 |    19,556 |    0.29 |
| FetchReachDict | 480x480 | 256x256 | NO | 25.5 | 11.7 |     1,111 |     2,931 |     5,746 |    19,113 |    0.30 |
| FetchSlide | 480x480 | 256x256 | NO | 26.7 | 12.7 |     1,093 |     2,851 |     6,346 |    19,043 |    0.33 |
| FetchSlideDense | 480x480 | 256x256 | NO | 26.7 | 12.7 |     1,098 |     2,842 |     6,311 |    19,103 |    0.33 |
| FetchSlideDict | 480x480 | 256x256 | NO | 26.7 | 12.7 |     1,097 |     2,845 |     6,224 |    19,036 |    0.33 |
| MountainCar | 400x600 | 400x600 | yes | 13.1 | 12.9 |     1,416 |     1,386 |     7,599 |     5,961 |    1.27 |
| MountainCarContinuous | 400x600 | 400x600 | yes | 13.1 | 12.9 |     1,410 |     1,388 |     7,484 |     5,998 |    1.25 |
| Pendulum | 500x500 | 500x500 | yes | 8.6 | 8.6 |     1,455 |     1,552 |     7,742 |     5,863 |    1.32 |
| Piecewise | 224x224 | 224x224 | yes | 3.2 | 3.2 |     4,872 |     6,367 |    28,868 |    30,640 |    0.94 |
| PiecewiseZones | 224x224 | 224x224 | yes | 4.9 | 4.9 |     4,049 |     5,697 |    26,465 |    28,654 |    0.92 |
| PointmazeLargeNavigate | 224x224 | 256x256 | NO | 12.3 | 14.9 |     3,048 |     3,316 |    20,163 |    18,153 |    1.11 |
| PushT | 96x96 | 256x256 | NO | 2.7 | 7.5 |     9,958 |     4,584 |    79,220 |    22,538 |    3.51 |
| SimplePointMaze | 500x500 | 500x500 | yes | 12.1 | 12.1 |     1,407 |     1,563 |     7,640 |     5,894 |    1.30 |
| TwoRoom | 224x224 | 224x224 | yes | 4.4 | 4.4 |     4,619 |     5,904 |    27,278 |    29,355 |    0.93 |
| TwoRoomTarget | 224x224 | 224x224 | yes | 4.8 | 4.8 |     4,185 |     5,502 |    26,941 |    28,286 |    0.95 |

On the **11** configs whose frames are the same picture in both runs, Oscar/Randall `local_n` is median **1.05x** (range 0.92-1.32). That factor is the hardware term; anything beyond it on the other rows is frame size.

### GPU cells that are NOT post-Step-6 measurements

These rows carry the Step-3 pre-Step-6 3090 number because no Step-6 shard re-measured them on GPU. Where a Step-6 change touched their backend they are LOWER BOUNDS, not current values.

**23/36**: DMCAcrobotSwingup, DMCCartpoleBalance, DMCCartpoleSwingup, DMCCheetahRun, DMCFingerTurnHard, DMCHopperHop, DMCHumanoidStand, DMCPendulumSwingup, DMCReacherHard, DMCWalkerWalk, FetchPickAndPlace, FetchPickAndPlaceDense, FetchPickAndPlaceDict, FetchPush, FetchPushDense, FetchPushDict, FetchReach, FetchReachDense, FetchReachDict, FetchSlide, FetchSlideDense, FetchSlideDict, PointmazeLargeNavigate

## Per-cell provenance

| config | GPU cell source | CPU 1-thread cell source | CPU 8-thread cell source |
|---|---|---|---|
| Acrobot | step6/gfxdraw/sweep2/gpu_paired_after_gpu2264.json (B=4096, gpu2264.oscar.ccv.brown.edu) | step6/gfxdraw/sweep/cpu_1thread_paired_after_node1829.json (B=64, node1829.oscar.ccv.brown.edu) | - |
| CartPole | step6/gfxdraw/sweep2/gpu_paired_after_gpu2264.json (B=4096, gpu2264.oscar.ccv.brown.edu) | step6/gfxdraw/sweep/cpu_1thread_paired_after_node1829.json (B=64, node1829.oscar.ccv.brown.edu) | - |
| CraftaxClassic | step6/tilemap/sweep/gpu_after_final.json (B=4096, gpu2104.oscar.ccv.brown.edu) | step6/tilemap/sweep/cpu_after_final.json (B=256, node2333.oscar.ccv.brown.edu) | - |
| DMCAcrobotSwingup | step3/gpu/3090_full.json (B=4096, gpu2107.oscar.ccv.brown.edu) **[pre-Step-6]** | step6/raycast3d/sweep/cpu_1thread_after_branch_bounds_on.json (B=64, node2335.oscar.ccv.brown.edu) | - |
| DMCBallInCupCatch | step6/raycast3d/sweep/gpu_3090_after_branch.json (B=4096, gpu2105.oscar.ccv.brown.edu) | step6/raycast3d/sweep/cpu_1thread_after_branch_bounds_on.json (B=64, node2335.oscar.ccv.brown.edu) | - |
| DMCCartpoleBalance | step3/gpu/3090_full.json (B=4096, gpu2107.oscar.ccv.brown.edu) **[pre-Step-6]** | step3/cpu/oscar_x86_1thread.json (B=64, node2349.oscar.ccv.brown.edu) **[pre-Step-6]** | - |
| DMCCartpoleSwingup | step3/gpu/3090_full.json (B=4096, gpu2107.oscar.ccv.brown.edu) **[pre-Step-6]** | step6/raycast3d/sweep/cpu_1thread_after_branch_bounds_on.json (B=64, node2335.oscar.ccv.brown.edu) | - |
| DMCCheetahRun | step3/gpu/3090_full.json (B=4096, gpu2107.oscar.ccv.brown.edu) **[pre-Step-6]** | step6/cpu-3d/sweep/after/cpu_1thread_paired_after_node2333.json (B=64, node2333.oscar.ccv.brown.edu) | - |
| DMCFingerTurnHard | step3/gpu/3090_full.json (B=4096, gpu2107.oscar.ccv.brown.edu) **[pre-Step-6]** | step3/cpu/oscar_x86_1thread.json (B=16, node2349.oscar.ccv.brown.edu) **[pre-Step-6]** | - |
| DMCHopperHop | step3/gpu/3090_full.json (B=4096, gpu2107.oscar.ccv.brown.edu) **[pre-Step-6]** | step6/cpu-3d/sweep/after/cpu_1thread_paired_after_node2333.json (B=256, node2333.oscar.ccv.brown.edu) | - |
| DMCHumanoidStand | step3/gpu/3090_full.json (B=4096, gpu2107.oscar.ccv.brown.edu) **[pre-Step-6]** | step6/cpu-3d/sweep/after/cpu_1thread_paired_after_node2333.json (B=64, node2333.oscar.ccv.brown.edu) | - |
| DMCPendulumSwingup | step3/gpu/3090_full.json (B=4096, gpu2107.oscar.ccv.brown.edu) **[pre-Step-6]** | step6/raycast3d/sweep/cpu_1thread_after_branch_bounds_on.json (B=16, node2335.oscar.ccv.brown.edu) | - |
| DMCReacherHard | step3/gpu/3090_full.json (B=4096, gpu2107.oscar.ccv.brown.edu) **[pre-Step-6]** | step6/raycast3d/sweep/cpu_1thread_after_branch_bounds_on.json (B=16, node2335.oscar.ccv.brown.edu) | - |
| DMCWalkerWalk | step3/gpu/3090_full.json (B=4096, gpu2107.oscar.ccv.brown.edu) **[pre-Step-6]** | step6/cpu-3d/sweep/after/cpu_1thread_paired_after_node2333.json (B=16, node2333.oscar.ccv.brown.edu) | - |
| FetchPickAndPlace | step3/gpu/3090_fetch311.json (B=1024, gpu2101.oscar.ccv.brown.edu) **[pre-Step-6]** | - | - |
| FetchPickAndPlaceDense | step3/gpu/3090_fetch311.json (B=4096, gpu2101.oscar.ccv.brown.edu) **[pre-Step-6]** | - | - |
| FetchPickAndPlaceDict | step3/gpu/3090_fetch311.json (B=4096, gpu2101.oscar.ccv.brown.edu) **[pre-Step-6]** | - | - |
| FetchPush | step3/gpu/3090_fetch311.json (B=4096, gpu2101.oscar.ccv.brown.edu) **[pre-Step-6]** | - | - |
| FetchPushDense | step3/gpu/3090_fetch311.json (B=1024, gpu2101.oscar.ccv.brown.edu) **[pre-Step-6]** | - | - |
| FetchPushDict | step3/gpu/3090_fetch311.json (B=4096, gpu2101.oscar.ccv.brown.edu) **[pre-Step-6]** | - | - |
| FetchReach | step3/gpu/3090_fetch311.json (B=1024, gpu2101.oscar.ccv.brown.edu) **[pre-Step-6]** | - | - |
| FetchReachDense | step3/gpu/3090_fetch311.json (B=4096, gpu2101.oscar.ccv.brown.edu) **[pre-Step-6]** | - | - |
| FetchReachDict | step3/gpu/3090_fetch311.json (B=4096, gpu2101.oscar.ccv.brown.edu) **[pre-Step-6]** | - | - |
| FetchSlide | step3/gpu/3090_fetch311.json (B=1024, gpu2101.oscar.ccv.brown.edu) **[pre-Step-6]** | - | - |
| FetchSlideDense | step3/gpu/3090_fetch311.json (B=4096, gpu2101.oscar.ccv.brown.edu) **[pre-Step-6]** | - | - |
| FetchSlideDict | step3/gpu/3090_fetch311.json (B=4096, gpu2101.oscar.ccv.brown.edu) **[pre-Step-6]** | - | - |
| MountainCar | step6/gfxdraw/sweep2/gpu_paired_after_gpu2264.json (B=4096, gpu2264.oscar.ccv.brown.edu) | step6/gfxdraw/sweep/cpu_1thread_paired_after_node1829.json (B=256, node1829.oscar.ccv.brown.edu) | - |
| MountainCarContinuous | step6/gfxdraw/sweep2/gpu_paired_after_gpu2264.json (B=4096, gpu2264.oscar.ccv.brown.edu) | step6/gfxdraw/sweep/cpu_1thread_paired_after_node1829.json (B=64, node1829.oscar.ccv.brown.edu) | - |
| Pendulum | step6/gfxdraw/sweep2/gpu_paired_after_gpu2264.json (B=4096, gpu2264.oscar.ccv.brown.edu) | step6/gfxdraw/sweep/cpu_1thread_paired_after_node1829.json (B=64, node1829.oscar.ccv.brown.edu) | - |
| Piecewise | step6/raster2d/sweep/gpu_after.json (B=4096, gpu2101.oscar.ccv.brown.edu) | step6/raster2d/sweep/cpu_1thread_after_swm.json (B=16, node2334.oscar.ccv.brown.edu) | - |
| PiecewiseZones | step6/raster2d/sweep/gpu_after.json (B=4096, gpu2101.oscar.ccv.brown.edu) | step6/raster2d/sweep/cpu_1thread_after_swm.json (B=16, node2334.oscar.ccv.brown.edu) | - |
| PointmazeLargeNavigate | step3/gpu/3090_full.json (B=4096, gpu2107.oscar.ccv.brown.edu) **[pre-Step-6]** | step6/raycast3d/sweep/cpu_1thread_after_branch_bounds_on.json (B=64, node2335.oscar.ccv.brown.edu) | - |
| PushT | step6/raster2d/sweep/gpu_after.json (B=4096, gpu2101.oscar.ccv.brown.edu) | step6/raster2d/sweep/cpu_1thread_after_pusht.json (B=1024, node2334.oscar.ccv.brown.edu) | - |
| SimplePointMaze | step6/raster2d/sweep/gpu_after.json (B=4096, gpu2101.oscar.ccv.brown.edu) | step6/raster2d/sweep/cpu_1thread_after_swm.json (B=16, node2334.oscar.ccv.brown.edu) | - |
| TwoRoom | step6/raster2d/sweep/gpu_after.json (B=4096, gpu2101.oscar.ccv.brown.edu) | step6/raster2d/sweep/cpu_1thread_after_swm.json (B=16, node2334.oscar.ccv.brown.edu) | - |
| TwoRoomTarget | step6/raster2d/sweep/gpu_after.json (B=4096, gpu2101.oscar.ccv.brown.edu) | step6/raster2d/sweep/cpu_1thread_after_swm.json (B=1, node2334.oscar.ccv.brown.edu) | - |

## Aggregates

| quantity | median | min | max | n |
|---|---|---|---|---|
| lance local_1 | 2,285.32 | 1,092.90 | 13,042.00 | 36 |
| lance local_n (8 workers) | 13,952.67 | 5,551.96 | 103,579.28 | 36 |
| lance shared_1 (NFS) | 2,276.36 | 1,092.18 | 11,090.89 | 36 |
| lance jobtmp_1 (GPFS) | 2,281.50 | 1,094.87 | 12,720.80 | 36 |
| lance nvJPEG (3090) | 6,041.24 | 2,762.46 | 15,508.26 | 36 |
| our GPU fps | 37,822.10 | 5,161.88 | 1,229,786.61 | 36 |
| our CPU 1-thread fps | 423.94 | 45.82 | 154,085.76 | 24 |
| GPU / local_n | 0.90 | 0.11 | 31.96 | 36 |
| GPU cores-of-8 replaced | 7.17 | 0.88 | 255.70 | 36 |
| CPU1 / local_1 | 0.16 | 0.00 | 30.28 | 24 |
| CPU1 / per-core | 0.19 | 0.00 | 32.04 | 24 |

### Rows whose GPU renderer beats the 8-worker Lance baseline

**12/36**: Acrobot, CartPole, CraftaxClassic, MountainCar, MountainCarContinuous, Pendulum, Piecewise, PiecewiseZones, PointmazeLargeNavigate, PushT, SimplePointMaze, TwoRoom

Not: DMCAcrobotSwingup, DMCBallInCupCatch, DMCCartpoleBalance, DMCCartpoleSwingup, DMCCheetahRun, DMCFingerTurnHard, DMCHopperHop, DMCHumanoidStand, DMCPendulumSwingup, DMCReacherHard, DMCWalkerWalk, FetchPickAndPlace, FetchPickAndPlaceDense, FetchPickAndPlaceDict, FetchPush, FetchPushDense, FetchPushDict, FetchReach, FetchReachDense, FetchReachDict, FetchSlide, FetchSlideDense, FetchSlideDict, TwoRoomTarget

### Rows whose 1-thread CPU renderer beats 1-process Lance

**1/36**: CraftaxClassic

Not: Acrobot, CartPole, DMCAcrobotSwingup, DMCBallInCupCatch, DMCCartpoleBalance, DMCCartpoleSwingup, DMCCheetahRun, DMCFingerTurnHard, DMCHopperHop, DMCHumanoidStand, DMCPendulumSwingup, DMCReacherHard, DMCWalkerWalk, FetchPickAndPlace, FetchPickAndPlaceDense, FetchPickAndPlaceDict, FetchPush, FetchPushDense, FetchPushDict, FetchReach, FetchReachDense, FetchReachDict, FetchSlide, FetchSlideDense, FetchSlideDict, MountainCar, MountainCarContinuous, Pendulum, Piecewise, PiecewiseZones, PointmazeLargeNavigate, PushT, SimplePointMaze, TwoRoom, TwoRoomTarget

### Rows whose 1-thread CPU renderer beats one core of the 8-worker Lance baseline

**1/36**: CraftaxClassic

Not: Acrobot, CartPole, DMCAcrobotSwingup, DMCBallInCupCatch, DMCCartpoleBalance, DMCCartpoleSwingup, DMCCheetahRun, DMCFingerTurnHard, DMCHopperHop, DMCHumanoidStand, DMCPendulumSwingup, DMCReacherHard, DMCWalkerWalk, FetchPickAndPlace, FetchPickAndPlaceDense, FetchPickAndPlaceDict, FetchPush, FetchPushDense, FetchPushDict, FetchReach, FetchReachDense, FetchReachDict, FetchSlide, FetchSlideDense, FetchSlideDict, MountainCar, MountainCarContinuous, Pendulum, Piecewise, PiecewiseZones, PointmazeLargeNavigate, PushT, SimplePointMaze, TwoRoom, TwoRoomTarget

### Rows whose 8-thread CPU renderer beats the 8-worker Lance baseline

**0/36**: none

Not: Acrobot, CartPole, CraftaxClassic, DMCAcrobotSwingup, DMCBallInCupCatch, DMCCartpoleBalance, DMCCartpoleSwingup, DMCCheetahRun, DMCFingerTurnHard, DMCHopperHop, DMCHumanoidStand, DMCPendulumSwingup, DMCReacherHard, DMCWalkerWalk, FetchPickAndPlace, FetchPickAndPlaceDense, FetchPickAndPlaceDict, FetchPush, FetchPushDense, FetchPushDict, FetchReach, FetchReachDense, FetchReachDict, FetchSlide, FetchSlideDense, FetchSlideDict, MountainCar, MountainCarContinuous, Pendulum, Piecewise, PiecewiseZones, PointmazeLargeNavigate, PushT, SimplePointMaze, TwoRoom, TwoRoomTarget
